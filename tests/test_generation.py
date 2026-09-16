"""Testes da Fase 4 — geração com citação verificada.

Nenhum teste aqui chama a Claude API de verdade: `model_caller` é uma função
falsa e determinística, no mesmo espírito do embedder/scorer falsos da Fase
3. Isso testa a ORQUESTRAÇÃO (quando abstém, quando reprocessa, quando
aceita) sem gastar tokens nem exigir ANTHROPIC_API_KEY para rodar `pytest`.
"""

from __future__ import annotations

import zlib

import pytest

from rag.generation.answer import answer_question
from rag.generation.prompt import build_context, build_retry_prompt, build_user_prompt
from rag.generation.verify import verify_citation, verify_citations
from rag.ingest.indexer import build_dense_index, tokenize
from rag.retrieval.hybrid import RetrievedChunk
from rag.retrieval.rerank import RankedChunk

CHUNKS = [
    {
        "chunk_id": "c1",
        "document": "contrato_a.pdf",
        "doc_type": "contrato_administrativo",
        "clause_path": "CLÁUSULA QUARTA",
        "clause_title": "DAS PENALIDADES",
        "page_start": 3,
        "page_end": 3,
        "token_count": 40,
        "fallback": False,
        "text": (
            "O atraso na entrega sujeitará a CONTRATADA a multa moratória de "
            "0,5% ao dia sobre o valor da parcela em atraso."
        ),
    },
    {
        "chunk_id": "c2",
        "document": "contrato_a.pdf",
        "doc_type": "contrato_administrativo",
        "clause_path": "CLÁUSULA PRIMEIRA",
        "clause_title": "DO OBJETO",
        "page_start": 1,
        "page_end": 1,
        "token_count": 20,
        "fallback": False,
        "text": "O objeto deste contrato é a prestação de serviços de manutenção predial.",
    },
]

CHUNK_STORE = {c["chunk_id"]: c for c in CHUNKS}


def _stable_hash(token: str) -> int:
    """Determinístico entre processos — ver o mesmo helper em test_retrieval.py.

    `hash()` embutido do Python é aleatorizado por processo (PYTHONHASHSEED);
    `zlib.crc32` não é. Usar `hash()` aqui faria este teste passar ou falhar
    dependendo de qual seed o interpretador sorteou naquela execução.
    """
    return zlib.crc32(token.encode("utf-8"))


def fake_embedder(texts: list[str]) -> list[list[float]]:
    dim = 32
    vectors = []
    for text in texts:
        vec = [0.0] * dim
        for tok in tokenize(text):
            vec[_stable_hash(tok) % dim] += 1.0
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        vectors.append([x / norm for x in vec])
    return vectors


@pytest.fixture
def chroma_client(tmp_path):
    import chromadb

    return chromadb.PersistentClient(path=str(tmp_path / "chroma"))


@pytest.fixture
def bm25_index():
    from rank_bm25 import BM25Okapi

    tokenized = [tokenize(c["text"]) for c in CHUNKS]
    return BM25Okapi(tokenized), [c["chunk_id"] for c in CHUNKS]


def _search_kwargs(chroma_client, bm25_index):
    bm25, ids = bm25_index
    build_dense_index(CHUNKS, fake_embedder, client=chroma_client, batch_size=10)
    return {
        "embedder": fake_embedder,
        "client": chroma_client,
        "bm25": bm25,
        "chunk_ids": ids,
        "chunk_store": CHUNK_STORE,
    }


def high_scorer(pairs: list[tuple[str, str]]) -> list[float]:
    return [0.95] * len(pairs)


def low_scorer(pairs: list[tuple[str, str]]) -> list[float]:
    return [0.05] * len(pairs)


# --- prompt.py -------------------------------------------------------------------


def _ranked(chunk_dict: dict, score: float = 0.9) -> RankedChunk:
    rc = RetrievedChunk(
        chunk_id=chunk_dict["chunk_id"],
        text=chunk_dict["text"],
        document=chunk_dict["document"],
        doc_type=chunk_dict["doc_type"],
        clause_path=chunk_dict["clause_path"],
        clause_title=chunk_dict["clause_title"],
        page_start=chunk_dict["page_start"],
        page_end=chunk_dict["page_end"],
        dense_rank=0,
        lexical_rank=0,
        fused_score=1.0,
    )
    return RankedChunk(chunk=rc, rerank_score=score)


def test_build_context_numera_fontes_a_partir_de_1():
    ranked = [_ranked(CHUNKS[0]), _ranked(CHUNKS[1])]
    context = build_context(ranked)
    assert "[1]" in context and "[2]" in context
    assert CHUNKS[0]["text"] in context
    assert CHUNKS[1]["clause_path"] in context


def test_build_user_prompt_inclui_a_pergunta():
    ranked = [_ranked(CHUNKS[0])]
    prompt = build_user_prompt("qual a multa por atraso", ranked)
    assert "qual a multa por atraso" in prompt
    assert "[1]" in prompt


def test_build_retry_prompt_lista_citacoes_rejeitadas():
    rejected = [
        {"source_id": 1, "quote": "coisa inventada", "match_score": 0.2, "threshold": 0.9}
    ]
    retry = build_retry_prompt(rejected)
    assert "coisa inventada" in retry
    assert "[1]" in retry


# --- verify.py ---------------------------------------------------------------------


def test_verify_citation_aceita_quote_literal():
    ranked = _ranked(CHUNKS[0])
    citation = {
        "source_id": 1,
        "document": "contrato_a.pdf",
        "clause": "CLÁUSULA QUARTA",
        "page": 3,
        "quote": "multa moratória de 0,5% ao dia sobre o valor da parcela em atraso",
    }
    result = verify_citation(citation, {1: ranked})
    assert result.verified
    assert result.match_score >= 0.90


def test_verify_citation_rejeita_quote_inventado():
    ranked = _ranked(CHUNKS[0])
    citation = {
        "source_id": 1,
        "document": "contrato_a.pdf",
        "clause": "CLÁUSULA QUARTA",
        "page": 3,
        "quote": "a contratada terá direito a reembolso integral em caso de atraso",
    }
    result = verify_citation(citation, {1: ranked})
    assert not result.verified
    assert result.match_score < 0.90


def test_verify_citation_com_source_id_inexistente():
    citation = {"source_id": 99, "document": "x", "clause": "x", "page": 1, "quote": "x"}
    result = verify_citation(citation, {1: _ranked(CHUNKS[0])})
    assert not result.verified
    assert "não corresponde" in result.reason


def test_verify_citations_processa_lista():
    ranked = _ranked(CHUNKS[0])
    citations = [
        {
            "source_id": 1,
            "document": "contrato_a.pdf",
            "clause": "CLÁUSULA QUARTA",
            "page": 3,
            "quote": "multa moratória de 0,5% ao dia",
        }
    ]
    results = verify_citations(citations, {1: ranked})
    assert len(results) == 1
    assert results[0].verified


# --- answer.py: orquestração completa ------------------------------------------------


def _good_tool_input():
    return {
        "answer": "O atraso na entrega gera multa moratória de 0,5% ao dia.",
        "citations": [
            {
                "source_id": 1,
                "document": "contrato_a.pdf",
                "clause": "CLÁUSULA QUARTA",
                "page": 3,
                "quote": "multa moratória de 0,5% ao dia sobre o valor da parcela em atraso",
            }
        ],
        "confidence": "alta",
        "insufficient_context": False,
    }


def test_abstem_sem_chamar_o_modelo_quando_score_de_rerank_e_baixo(chroma_client, bm25_index):
    calls = []

    def spy_caller(system, messages):
        calls.append((system, messages))
        return _good_tool_input()

    result = answer_question(
        "multa por atraso",
        scorer=low_scorer,
        model_caller=spy_caller,
        min_rerank_score=0.35,
        log=False,
        **_search_kwargs(chroma_client, bm25_index),
    )

    assert result.abstained
    assert "limiar" in result.abstain_reason
    assert calls == [], "o modelo não deveria ser chamado quando o score já é baixo"


def test_aceita_resposta_com_citacao_valida_de_primeira(chroma_client, bm25_index):
    result = answer_question(
        "qual a multa por atraso na entrega",
        scorer=high_scorer,
        model_caller=lambda system, messages: _good_tool_input(),
        log=False,
        **_search_kwargs(chroma_client, bm25_index),
    )

    assert not result.abstained
    assert result.retries == 0
    assert len(result.citations) == 1
    assert result.citations[0].verified


def test_document_filter_restringe_as_fontes_usadas_na_geracao(chroma_client, bm25_index):
    """Fase 6 — seletor de documentos: filtrar para um documento que NÃO tem
    a resposta deve levar à abstenção por falta de candidato, mesmo que o
    corpus inteiro (sem filtro) resolvesse a pergunta perfeitamente.
    """
    result = answer_question(
        "qual a multa por atraso na entrega",
        document_filter=["contrato_b.pdf"],  # a multa (c1) está em contrato_a.pdf
        scorer=high_scorer,
        model_caller=lambda system, messages: _good_tool_input(),
        log=False,
        **_search_kwargs(chroma_client, bm25_index),
    )

    assert result.abstained
    assert "nenhum chunk candidato" in result.abstain_reason


def test_reprocessa_uma_vez_quando_citacao_inicial_falha(chroma_client, bm25_index):
    calls = {"n": 0}

    def flaky_caller(system, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            bad = _good_tool_input()
            bad["citations"][0]["quote"] = "isto não existe em lugar nenhum do texto original"
            return bad
        return _good_tool_input()

    result = answer_question(
        "qual a multa por atraso na entrega",
        scorer=high_scorer,
        model_caller=flaky_caller,
        log=False,
        **_search_kwargs(chroma_client, bm25_index),
    )

    assert calls["n"] == 2
    assert result.retries == 1
    assert not result.abstained
    assert result.citations[0].verified


def test_abstem_quando_citacao_falha_mesmo_apos_reprocessar(chroma_client, bm25_index):
    def always_bad_caller(system, messages):
        bad = _good_tool_input()
        bad["citations"][0]["quote"] = "texto que nunca vai bater com a fonte de jeito nenhum"
        return bad

    result = answer_question(
        "qual a multa por atraso na entrega",
        scorer=high_scorer,
        model_caller=always_bad_caller,
        log=False,
        **_search_kwargs(chroma_client, bm25_index),
    )

    assert result.abstained
    assert "reprocessamento" in result.abstain_reason
    assert result.retries == 1


def test_confia_na_abstencao_declarada_pelo_proprio_modelo(chroma_client, bm25_index):
    calls = []

    def spy_caller(system, messages):
        calls.append(messages)
        return {
            "answer": "As fontes não trazem informação sobre isso.",
            "citations": [],
            "confidence": "baixa",
            "insufficient_context": True,
        }

    result = answer_question(
        "qual a multa por atraso na entrega",
        scorer=high_scorer,
        model_caller=spy_caller,
        log=False,
        **_search_kwargs(chroma_client, bm25_index),
    )

    assert result.abstained
    assert result.insufficient_context
    assert len(calls) == 1, "não deveria reprocessar uma abstenção declarada pelo modelo"
