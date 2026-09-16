"""Testes da Fase 3 — índice híbrido e reranking.

Nenhum teste aqui baixa o BGE-M3 ou o BGE-reranker-v2-m3: o embedder e o
scorer são falsos e determinísticos, injetados nas funções via parâmetro.
Isso valida a integração real com ChromaDB e BM25 (nenhuma delas é mockada) e
a lógica de fusão/reranking — só o modelo de ML é trocado. `make ingest` /
`make search`, na sua máquina, usam os modelos de verdade.
"""

from __future__ import annotations

import zlib

import pytest

from rag.ingest.indexer import (
    build_dense_index,
    build_lexical_index,
    get_collection,
    tokenize,
)
from rag.retrieval.hybrid import (
    RetrievedChunk,
    dense_search,
    lexical_search,
    reciprocal_rank_fusion,
    search,
)
from rag.retrieval.rerank import RankedChunk, rerank

# --- fixtures ------------------------------------------------------------------

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
        "clause_path": "CLÁUSULA DÉCIMA SEGUNDA",
        "clause_title": "DA RESCISÃO",
        "page_start": 8,
        "page_end": 9,
        "token_count": 60,
        "fallback": False,
        "text": (
            "O contrato poderá ser rescindido unilateralmente pelo CONTRATANTE "
            "nas hipóteses previstas no art. 78 da Lei nº 8.666/1993."
        ),
    },
    {
        "chunk_id": "c3",
        "document": "contrato_b.pdf",
        "doc_type": "termo_aditivo",
        "clause_path": "CLÁUSULA PRIMEIRA",
        "clause_title": "DO PRAZO",
        "page_start": 1,
        "page_end": 1,
        "token_count": 25,
        "fallback": False,
        "text": "Fica prorrogado o prazo de vigência do contrato original por mais 12 meses.",
    },
    {
        "chunk_id": "c4",
        "document": "contrato_b.pdf",
        "doc_type": "termo_aditivo",
        "clause_path": "PREÂMBULO",
        "clause_title": "",
        "page_start": 1,
        "page_end": 1,
        "token_count": 20,
        "fallback": False,
        "text": (
            "As partes qualificadas neste instrumento resolvem celebrar "
            "o presente termo aditivo."
        ),
    },
]

CHUNK_STORE = {c["chunk_id"]: c for c in CHUNKS}


def _stable_hash(token: str) -> int:
    """Hash determinístico entre processos e máquinas.

    Bug real encontrado rodando esta suíte fora do sandbox: a versão
    original usava `hash(tok)`, o `hash()` embutido do Python — que para
    strings é ALEATORIZADO a cada processo por padrão (PYTHONHASHSEED),
    justamente por segurança (evita ataques de colisão de hash em serviços
    web). O comentário dizia "determinístico", mas não era: o mesmo teste
    passava no sandbox e falhava na máquina do usuário, porque o balde
    (`hash(tok) % dim`) de cada token mudava a cada nova sessão do
    interpretador. `zlib.crc32` não tem essa aleatorização — mesmo token,
    mesmo balde, sempre, em qualquer processo ou máquina.
    """
    return zlib.crc32(token.encode("utf-8"))


def fake_embedder(texts: list[str]) -> list[list[float]]:
    """Bag-of-words hasheado num vetor pequeno — determinístico, sem ML.

    Não captura semântica de verdade (não sabe que "multa" e "penalidade" são
    parecidos), mas dois textos que compartilham termos ficam próximos, o
    suficiente para testar a integração com o Chroma sem depender de um
    modelo real.
    """
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


# --- RRF: lógica pura, sem I/O ---------------------------------------------------


def test_rrf_favorece_quem_aparece_bem_em_ambas_as_listas():
    dense = ["c1", "c2", "c3"]
    lexical = ["c2", "c1", "c4"]
    scores = reciprocal_rank_fusion([dense, lexical])
    # c1 e c2 aparecem nas duas listas, em posições altas; c3 e c4 só numa.
    assert scores["c1"] > scores["c3"]
    assert scores["c2"] > scores["c4"]


def test_rrf_com_uma_lista_vazia_nao_quebra():
    scores = reciprocal_rank_fusion([["c1", "c2"], []])
    assert scores["c1"] > scores["c2"]


def test_rrf_chunk_ausente_de_uma_lista_nao_e_penalizado_alem_disso():
    only_dense = reciprocal_rank_fusion([["c1"], []])
    both = reciprocal_rank_fusion([["c1"], ["c1"]])
    assert both["c1"] > only_dense["c1"]  # aparecer nas duas soma, não penaliza


# --- BM25 (léxico): integração real, sem mocks -----------------------------------


def test_bm25_encontra_termo_exato_que_nao_aparece_em_outros_chunks():
    from rank_bm25 import BM25Okapi

    tokenized = [tokenize(c["text"]) for c in CHUNKS]
    bm25 = BM25Okapi(tokenized)
    ids = [c["chunk_id"] for c in CHUNKS]

    results = lexical_search("art. 78", 10, bm25=bm25, chunk_ids=ids)
    assert results[0] == "c2"  # é o único chunk que menciona "art. 78"


def test_bm25_query_sem_nenhum_termo_em_comum_devolve_vazio(bm25_index):
    bm25, ids = bm25_index
    results = lexical_search("xilofone marciano inexistente", 10, bm25=bm25, chunk_ids=ids)
    assert results == []


def test_build_lexical_index_grava_e_recarrega_do_disco(tmp_path):
    path = tmp_path / "bm25.pkl"
    vocab_size = build_lexical_index(CHUNKS, path=path)
    assert vocab_size > 0
    assert path.exists()

    import pickle

    with path.open("rb") as fh:
        data = pickle.load(fh)
    assert data["chunk_ids"] == [c["chunk_id"] for c in CHUNKS]
    results = lexical_search("rescindido", 10, bm25=data["bm25"], chunk_ids=data["chunk_ids"])
    assert "c2" in results


# --- ChromaDB (denso): integração real, embedder falso ---------------------------


def test_build_dense_index_e_dense_search_encontram_texto_similar(chroma_client):
    batches, dim = build_dense_index(CHUNKS, fake_embedder, client=chroma_client, batch_size=2)
    assert batches == 2  # 4 chunks, lote de 2
    assert dim == 32

    collection = get_collection(chroma_client)
    assert collection.count() == 4

    # Uma query com as mesmas palavras do chunk c1 deve recuperá-lo primeiro.
    results = dense_search("multa moratória atraso", fake_embedder, 4, client=chroma_client)
    assert results[0] == "c1"


def test_dense_search_em_colecao_vazia_nao_quebra(chroma_client):
    results = dense_search("qualquer coisa", fake_embedder, 5, client=chroma_client)
    assert results == []


# --- busca híbrida ponta a ponta --------------------------------------------------


def test_search_funde_denso_e_lexico(chroma_client, bm25_index):
    build_dense_index(CHUNKS, fake_embedder, client=chroma_client, batch_size=10)
    bm25, ids = bm25_index

    results = search(
        "multa moratória por atraso na entrega",
        top_k=4,
        embedder=fake_embedder,
        client=chroma_client,
        bm25=bm25,
        chunk_ids=ids,
        chunk_store=CHUNK_STORE,
    )

    assert results, "busca híbrida não devolveu nada"
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].chunk_id == "c1"
    assert results[0].clause_path == "CLÁUSULA QUARTA"
    # pelo menos um dos dois rankings encontrou o melhor resultado
    assert results[0].dense_rank is not None or results[0].lexical_rank is not None


def test_search_devolve_no_maximo_top_k(chroma_client, bm25_index):
    build_dense_index(CHUNKS, fake_embedder, client=chroma_client, batch_size=10)
    bm25, ids = bm25_index

    results = search(
        "contrato",
        top_k=2,
        embedder=fake_embedder,
        client=chroma_client,
        bm25=bm25,
        chunk_ids=ids,
        chunk_store=CHUNK_STORE,
    )
    assert len(results) <= 2


# --- filtro por documento (Fase 6 — seletor de documentos da UI) ------------------


def test_search_com_document_filter_so_devolve_chunks_desses_documentos(
    chroma_client, bm25_index
):
    build_dense_index(CHUNKS, fake_embedder, client=chroma_client, batch_size=10)
    bm25, ids = bm25_index

    # "contrato" casa com chunks dos dois documentos — sem filtro, os dois
    # apareceriam. Com o filtro, só contrato_b.pdf pode aparecer.
    results = search(
        "contrato",
        top_k=4,
        document_filter=["contrato_b.pdf"],
        embedder=fake_embedder,
        client=chroma_client,
        bm25=bm25,
        chunk_ids=ids,
        chunk_store=CHUNK_STORE,
    )
    assert results, "filtro não deveria zerar o resultado — contrato_b.pdf tem chunks relevantes"
    assert all(r.document == "contrato_b.pdf" for r in results)


def test_search_com_document_filter_exclui_o_documento_certo_mesmo_sendo_o_mais_relevante(
    chroma_client, bm25_index
):
    build_dense_index(CHUNKS, fake_embedder, client=chroma_client, batch_size=10)
    bm25, ids = bm25_index

    # "multa moratória atraso" casa quase perfeitamente com c1 (contrato_a.pdf)
    # — sem filtro, c1 seria o primeiro colocado disparado. Com o filtro para
    # contrato_b.pdf, c1 (e qualquer outro chunk de contrato_a.pdf) não pode
    # aparecer, mesmo sendo o resultado mais relevante do corpus inteiro.
    results = search(
        "multa moratória atraso",
        top_k=4,
        document_filter=["contrato_b.pdf"],
        embedder=fake_embedder,
        client=chroma_client,
        bm25=bm25,
        chunk_ids=ids,
        chunk_store=CHUNK_STORE,
    )
    assert all(r.document == "contrato_b.pdf" for r in results)
    assert "c1" not in [r.chunk_id for r in results]


def test_search_sem_document_filter_nao_muda_comportamento(chroma_client, bm25_index):
    build_dense_index(CHUNKS, fake_embedder, client=chroma_client, batch_size=10)
    bm25, ids = bm25_index

    kwargs = dict(
        top_k=4,
        embedder=fake_embedder,
        client=chroma_client,
        bm25=bm25,
        chunk_ids=ids,
        chunk_store=CHUNK_STORE,
    )
    sem_filtro = search("contrato", document_filter=None, **kwargs)
    com_filtro_vazio = search("contrato", document_filter=[], **kwargs)
    assert [r.chunk_id for r in sem_filtro] == [r.chunk_id for r in com_filtro_vazio]


# --- reranking --------------------------------------------------------------------


def _chunk(chunk_id: str, text: str, **overrides) -> RetrievedChunk:
    base = dict(
        chunk_id=chunk_id,
        text=text,
        document="doc.pdf",
        doc_type="",
        clause_path="",
        clause_title="",
        page_start=1,
        page_end=1,
        dense_rank=0,
        lexical_rank=0,
        fused_score=1.0,
    )
    base.update(overrides)
    return RetrievedChunk(**base)


def test_rerank_reordena_pelo_score_do_scorer():
    candidates = [
        _chunk("a", "texto pouco relevante"),
        _chunk("b", "texto muito relevante para a pergunta"),
    ]

    def scorer(pairs):
        # inverte a ordem de entrada de propósito, pra provar que o rerank
        # reordena de fato e não só preserva a ordem da busca híbrida.
        return [0.1, 0.9]

    ranked = rerank("pergunta", candidates, top_n=2, scorer=scorer)
    assert [r.chunk.chunk_id for r in ranked] == ["b", "a"]
    assert all(isinstance(r, RankedChunk) for r in ranked)


def test_rerank_corta_em_top_n():
    candidates = [_chunk(str(i), f"texto {i}") for i in range(5)]
    ranked = rerank("q", candidates, top_n=2, scorer=lambda pairs: list(range(len(pairs))))
    assert len(ranked) == 2


def test_rerank_lista_vazia_nao_quebra():
    assert rerank("q", [], top_n=5, scorer=lambda pairs: []) == []
