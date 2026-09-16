"""Fase 4 — orquestração: recuperação -> geração -> verificação -> resposta.

Aqui se juntam as quatro defesas contra alucinação descritas no PLANO.md:

1. Abstenção por limiar, ANTES de chamar o modelo — se o melhor resultado da
   Fase 3 já vem com score de reranking baixo, a Claude API nem é chamada.
   Custo zero e sem chance de o LLM "tentar ser útil" com contexto ruim.
2. Contexto numerado e prompt fechado (`prompt.py`) — na chamada em si.
3. Verificação literal da citação (`verify.py`) — depois que a resposta
   volta, antes de repassá-la ao usuário.
4. Reprocessamento único quando a verificação falha, e abstenção (não uma
   resposta parcialmente inventada) se falhar de novo.

Cada dependência pesada (embedder, scorer, chamada à API) é injetável — o
mesmo padrão das Fases 2 e 3. `tests/test_answer.py` roda o fluxo inteiro
com um `model_caller` falso e determinístico, sem gastar um token de API
real.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from rag.generation.client import ModelCaller, get_default_model_caller
from rag.generation.prompt import SYSTEM_PROMPT, build_retry_prompt, build_user_prompt
from rag.generation.query_log import log_query
from rag.generation.verify import CitationVerification, verify_citations
from rag.ingest.indexer import Embedder
from rag.retrieval.hybrid import search
from rag.retrieval.rerank import RankedChunk, Scorer, rerank
from rag.settings import settings

MAX_RETRIES = 1

# Chamado em cada etapa do pipeline, só para observabilidade externa (a Fase
# 5 usa isso para emitir eventos de progresso via SSE em `/query/stream`).
# Nunca influencia o resultado — se ninguém estiver ouvindo, o padrão
# (`lambda _: None`) faz o hook custar essencialmente nada.
StageHook = Callable[[str], None]


def _noop_stage(_: str) -> None:
    pass


@dataclass
class AnswerResult:
    query: str
    answer: str
    confidence: str
    insufficient_context: bool
    abstained: bool
    abstain_reason: str
    citations: list[CitationVerification] = field(default_factory=list)
    retrieval_top_score: float | None = None
    retries: int = 0
    sources: list[RankedChunk] = field(default_factory=list)

    def to_log_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "confidence": self.confidence,
            "insufficient_context": self.insufficient_context,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "citations": [c.to_dict() for c in self.citations],
            "retrieval_top_score": self.retrieval_top_score,
            "retries": self.retries,
        }


def _abstain(query: str, reason: str, *, top_score: float | None = None) -> AnswerResult:
    return AnswerResult(
        query=query,
        answer="Não há informação suficiente nas fontes disponíveis para responder com confiança.",
        confidence="baixa",
        insufficient_context=True,
        abstained=True,
        abstain_reason=reason,
        retrieval_top_score=top_score,
    )


def answer_question(
    query: str,
    *,
    top_k: int | None = None,
    rerank_top_n: int | None = None,
    min_rerank_score: float | None = None,
    citation_threshold: float | None = None,
    max_retries: int = MAX_RETRIES,
    document_filter: list[str] | None = None,
    embedder: Embedder | None = None,
    scorer: Scorer | None = None,
    model_caller: ModelCaller | None = None,
    client=None,
    bm25=None,
    chunk_ids=None,
    chunk_store: dict[str, dict] | None = None,
    log: bool = True,
    on_stage: StageHook | None = None,
) -> AnswerResult:
    """Ponto de entrada da Fase 4. Devolve sempre um AnswerResult — nunca
    lança por causa de conteúdo do modelo (só por erro de infraestrutura,
    como a Claude API fora do ar).

    `document_filter` (Fase 6): restringe a busca a um subconjunto de
    documentos — o seletor de documentos da UI. `None`/lista vazia busca no
    corpus inteiro, sem mudança de comportamento.
    """
    on_stage = on_stage or _noop_stage
    min_rerank_score = (
        min_rerank_score if min_rerank_score is not None else settings.min_rerank_score
    )

    on_stage("buscando fontes")
    candidates = search(
        query,
        top_k=top_k,
        document_filter=document_filter or None,
        embedder=embedder,
        client=client,
        bm25=bm25,
        chunk_ids=chunk_ids,
        chunk_store=chunk_store,
    )
    ranked = rerank(query, candidates, top_n=rerank_top_n, scorer=scorer)

    if not ranked:
        result = _abstain(query, "a busca não encontrou nenhum chunk candidato")
        if log:
            log_query(result.to_log_dict())
        on_stage("concluído")
        return result

    top_score = ranked[0].rerank_score
    if top_score < min_rerank_score:
        result = _abstain(
            query,
            f"melhor resultado da busca (score={top_score:.3f}) fica abaixo do "
            f"limiar de confiança MIN_RERANK_SCORE={min_rerank_score:.3f}",
            top_score=top_score,
        )
        if log:
            log_query(result.to_log_dict())
        on_stage("concluído (abstenção por limiar)")
        return result

    result = _generate_and_verify(
        query, ranked, top_score, citation_threshold, max_retries, model_caller, on_stage
    )
    if log:
        log_query(result.to_log_dict())
    on_stage("concluído")
    return result


def _generate_and_verify(
    query: str,
    ranked: list[RankedChunk],
    top_score: float,
    citation_threshold: float | None,
    max_retries: int,
    model_caller: ModelCaller | None,
    on_stage: StageHook,
) -> AnswerResult:
    model_caller = model_caller or get_default_model_caller()
    chunks_by_source_id = dict(enumerate(ranked, start=1))
    user_prompt = build_user_prompt(query, ranked)

    on_stage("gerando resposta")
    tool_input = model_caller(SYSTEM_PROMPT, [{"role": "user", "content": user_prompt}])
    retries = 0

    while True:
        insufficient = bool(tool_input.get("insufficient_context"))
        citations_raw = tool_input.get("citations") or []

        if insufficient:
            # O próprio modelo admite que as fontes não bastam — confiamos
            # nessa admissão em vez de forçar uma citação que ele já disse
            # não ter. Não há o que verificar.
            return AnswerResult(
                query=query,
                answer=tool_input.get("answer", ""),
                confidence=tool_input.get("confidence", "baixa"),
                insufficient_context=True,
                abstained=True,
                abstain_reason="o modelo reportou insufficient_context=true",
                retrieval_top_score=top_score,
                retries=retries,
                sources=ranked,
            )

        on_stage("verificando citações")
        verifications = verify_citations(
            citations_raw, chunks_by_source_id, threshold=citation_threshold
        )
        failed = [v for v in verifications if not v.verified]

        if not failed:
            return AnswerResult(
                query=query,
                answer=tool_input.get("answer", ""),
                confidence=tool_input.get("confidence", "baixa"),
                insufficient_context=False,
                abstained=False,
                abstain_reason="",
                citations=verifications,
                retrieval_top_score=top_score,
                retries=retries,
                sources=ranked,
            )

        if not citations_raw:
            # Resposta afirmativa (insufficient_context=false) sem nenhuma
            # citação: viola a regra 2 do prompt. Trata como falha de
            # verificação, não como sucesso vazio.
            failed = [
                CitationVerification(
                    source_id=-1,
                    document="",
                    clause="",
                    page=0,
                    quote="",
                    match_score=0.0,
                    threshold=citation_threshold or settings.citation_match_threshold,
                    verified=False,
                    reason="resposta afirmativa sem nenhuma citação",
                )
            ]

        if retries >= max_retries:
            verified = [v for v in verifications if v.verified]
            if not verified:
                # Nenhuma citação sobreviveu, nem depois de reprocessar —
                # mais seguro abster do que devolver uma resposta sem
                # nenhuma citação confirmada.
                return AnswerResult(
                    query=query,
                    answer=tool_input.get("answer", ""),
                    confidence="baixa",
                    insufficient_context=True,
                    abstained=True,
                    abstain_reason=(
                        "todas as citações falharam na verificação literal, "
                        "mesmo após reprocessamento"
                    ),
                    citations=verifications,
                    retrieval_top_score=top_score,
                    retries=retries,
                    sources=ranked,
                )
            # Resposta parcial: mantém só as citações que passaram, e
            # rebaixa a confiança porque parte do que o modelo alegou não
            # se sustentou.
            return AnswerResult(
                query=query,
                answer=tool_input.get("answer", ""),
                confidence="baixa",
                insufficient_context=False,
                abstained=False,
                abstain_reason="",
                citations=verified,
                retrieval_top_score=top_score,
                retries=retries,
                sources=ranked,
            )

        # Ainda há tentativa disponível: reprocessa com a correção.
        on_stage("reprocessando citação rejeitada")
        retry_prompt = user_prompt + "\n\n" + build_retry_prompt([f.to_dict() for f in failed])
        tool_input = model_caller(SYSTEM_PROMPT, [{"role": "user", "content": retry_prompt}])
        retries += 1
