"""Fase 3 — busca híbrida: denso (ChromaDB) + léxico (BM25), fundidos por RRF.

Por que fundir em vez de escolher um: busca densa captura sinônimo e paráfrase
("penalidade por atraso" encontra "multa moratória"); busca léxica captura
termo exato e não erra o que o denso generaliza demais ("Art. 78, IV" precisa
casar literalmente, não "por perto"). Um contrato administrativo usa os dois
tipos de linguagem na mesma página — abrir mão de qualquer um perde recall.

RECIPROCAL RANK FUSION (RRF): em vez de tentar comparar o score de cosseno do
denso com o BM25 score do léxico (escalas incompatíveis — normalizar entre eles
é arbitrário), a fusão usa só a *posição* de cada chunk em cada ranking:

    rrf_score(chunk) = Σ  1 / (k + rank_da_lista(chunk))

Um chunk que aparece bem posicionado nas duas listas sobe ao topo; um chunk
ausente numa lista não é penalizado além de não somar aquele termo. k=60 é a
constante da literatura original do RRF (Cormack et al., 2009) — suaviza o
peso das posições mais altas sem um motivo para mudar aqui.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import lru_cache

from rag.ingest.indexer import (
    BM25_INDEX_FILENAME,
    Embedder,
    _chroma_client,
    get_collection,
    get_default_embedder,
    load_chunks,
    tokenize,
)
from rag.settings import settings

RRF_K = 60


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    document: str
    doc_type: str
    clause_path: str
    clause_title: str
    page_start: int
    page_end: int
    dense_rank: int | None
    lexical_rank: int | None
    fused_score: float

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "document": self.document,
            "doc_type": self.doc_type,
            "clause_path": self.clause_path,
            "clause_title": self.clause_title,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "dense_rank": self.dense_rank,
            "lexical_rank": self.lexical_rank,
            "fused_score": self.fused_score,
        }


@lru_cache(maxsize=1)
def _chunk_store() -> dict[str, dict]:
    """chunk_id -> chunk completo (texto + metadados), lido uma vez do jsonl."""
    return {c["chunk_id"]: c for c in load_chunks()}


def _load_bm25(path=None):
    path = path or (settings.chroma_dir / BM25_INDEX_FILENAME)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe. Rode `python -m rag.ingest.indexer` primeiro (Fase 3)."
        )
    with path.open("rb") as fh:
        data = pickle.load(fh)
    return data["bm25"], data["chunk_ids"]


def dense_search(query: str, embedder: Embedder, n: int, *, client=None) -> list[str]:
    """Devolve chunk_ids ordenados por similaridade, do mais pro menos parecido."""
    collection = get_collection(client or _chroma_client())
    if collection.count() == 0:
        return []
    query_vector = embedder([query])[0]
    result = collection.query(query_embeddings=[query_vector], n_results=min(n, collection.count()))
    return result["ids"][0]


def lexical_search(query: str, n: int, *, bm25=None, chunk_ids=None) -> list[str]:
    """Devolve chunk_ids ordenados por score BM25, do maior pro menor."""
    if bm25 is None or chunk_ids is None:
        bm25, chunk_ids = _load_bm25()
    if not chunk_ids:
        return []
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    # BM25 dá score 0 pra quem não compartilha nenhum termo — não é "resultado",
    # é ausência. Cortar isso evita que ruído entre na fusão como se fosse sinal.
    return [chunk_ids[i] for i in ranked[:n] if scores[i] > 0]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Funde N listas de chunk_ids ordenados num único score por chunk_id."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


#  Quantas vezes maior que top_k a gente busca quando um filtro de documento
#  está ativo. Sem isso, filtrar DEPOIS de já ter cortado pra top_k=20 quase
#  sempre devolveria lista vazia — o documento escolhido podia estar todo
#  fora da janela de 20 candidatos mais bem rankeados no corpus inteiro.
DOCUMENT_FILTER_FETCH_MULTIPLIER = 10


def search(
    query: str,
    *,
    top_k: int | None = None,
    document_filter: list[str] | None = None,
    embedder: Embedder | None = None,
    client=None,
    bm25=None,
    chunk_ids=None,
    chunk_store: dict[str, dict] | None = None,
) -> list[RetrievedChunk]:
    """Busca híbrida completa: denso + léxico + RRF. O ponto de entrada da Fase 3.

    `document_filter` (Fase 6, seletor de documentos da UI) restringe o
    resultado final a chunks desses documentos. Continua sendo uma busca
    híbrida de verdade, não um filtro cosmético pós-top_k: quando ativo, a
    busca busca mais candidatos (`DOCUMENT_FILTER_FETCH_MULTIPLIER`x) antes de
    filtrar e só então corta em `top_k`, senão um documento pouco citado no
    ranking geral nunca sobreviveria ao corte de 20.

    Os parâmetros embedder/client/bm25/chunk_ids/chunk_store existem para os
    testes injetarem um embedder falso e um Chroma/BM25/corpus temporários —
    em uso normal, todos vêm do disco (índice construído por
    `python -m rag.ingest.indexer`).
    """
    top_k = top_k or settings.retrieve_top_k
    embedder = embedder or get_default_embedder()
    fetch_n = top_k * DOCUMENT_FILTER_FETCH_MULTIPLIER if document_filter else top_k

    dense_ids = dense_search(query, embedder, fetch_n, client=client)
    lexical_ids = lexical_search(query, fetch_n, bm25=bm25, chunk_ids=chunk_ids)

    dense_rank = {cid: i for i, cid in enumerate(dense_ids)}
    lexical_rank = {cid: i for i, cid in enumerate(lexical_ids)}

    fused = reciprocal_rank_fusion([dense_ids, lexical_ids])
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

    store = chunk_store if chunk_store is not None else _chunk_store()
    documents_wanted = set(document_filter) if document_filter else None
    results = []
    for chunk_id, score in ordered:
        if len(results) >= top_k:
            break
        chunk = store.get(chunk_id)
        if chunk is None:
            continue  # índice desatualizado em relação ao chunks.jsonl
        if documents_wanted is not None and chunk["document"] not in documents_wanted:
            continue
        results.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=chunk["text"],
                document=chunk["document"],
                doc_type=chunk.get("doc_type") or "",
                clause_path=chunk.get("clause_path") or "",
                clause_title=chunk.get("clause_title") or "",
                page_start=chunk.get("page_start", 0),
                page_end=chunk.get("page_end", 0),
                dense_rank=dense_rank.get(chunk_id),
                lexical_rank=lexical_rank.get(chunk_id),
                fused_score=score,
            )
        )
    return results
