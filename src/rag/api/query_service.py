"""Fase 5 — ponte entre a Fase 4 (`answer_question`) e a API HTTP.

A conversão de `AnswerResult` (dataclass interno da Fase 4, com objetos
`CitationVerification` e `RankedChunk`) para `QueryResponse` (o contrato
Pydantic público da API) mora num lugar só. Nem `answer.py` deveria saber
que existe uma API HTTP, nem `main.py`/`query_stream.py` deveriam duplicar a
lógica de conversão cada um do seu jeito.
"""

from __future__ import annotations

from rag.api.schemas import CitationResponse, QueryResponse, SourceChunk
from rag.generation.answer import AnswerResult, StageHook, answer_question


def to_query_response(result: AnswerResult) -> QueryResponse:
    return QueryResponse(
        query=result.query,
        answer=result.answer,
        confidence=result.confidence,
        insufficient_context=result.insufficient_context,
        abstained=result.abstained,
        abstain_reason=result.abstain_reason,
        citations=[
            CitationResponse(
                source_id=c.source_id,
                document=c.document,
                clause=c.clause,
                page=c.page,
                quote=c.quote,
                match_score=c.match_score,
                verified=c.verified,
            )
            for c in result.citations
        ],
        # `source_id` numerado a partir de 1, na mesma ordem que
        # `chunks_by_source_id` em `answer.py` — é o que faz uma citação
        # apontar pro chunk certo aqui.
        sources=[
            SourceChunk(
                source_id=i,
                document=ranked.chunk.document,
                clause=ranked.chunk.clause_path or ranked.chunk.clause_title,
                page_start=ranked.chunk.page_start,
                page_end=ranked.chunk.page_end,
                text=ranked.chunk.text,
                rerank_score=ranked.rerank_score,
            )
            for i, ranked in enumerate(result.sources, start=1)
        ],
        retrieval_top_score=result.retrieval_top_score,
        retries=result.retries,
    )


def run_query(
    query: str,
    *,
    top_k: int | None = None,
    rerank_top_n: int | None = None,
    document_filter: list[str] | None = None,
    on_stage: StageHook | None = None,
) -> QueryResponse:
    result = answer_question(
        query,
        top_k=top_k,
        rerank_top_n=rerank_top_n,
        document_filter=document_filter,
        on_stage=on_stage,
    )
    return to_query_response(result)
