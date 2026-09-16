"""Fase 3 — construção dos dois índices a partir de data/processed/chunks.jsonl.

Dois índices, propositalmente separados:

  1. Denso (ChromaDB)   — embeddings BGE-M3, captura similaridade semântica.
  2. Léxico (BM25)      — casamento de termo exato, captura o que o denso perde:
                          "Art. 78, IV", "multa moratória", números de cláusula.

A Fase 3 completa (busca) funde os dois por Reciprocal Rank Fusion em
retrieval/hybrid.py. Este módulo só constrói e persiste.

DESIGN: o embedder é um parâmetro, não uma chamada direta a sentence-transformers
dentro da função. Isso existe por um motivo concreto, não estilo: o BGE-M3 pesa
~2GB e leva minutos para baixar na primeira execução. Testar a lógica de
indexação (upsert no Chroma, alinhamento de IDs, paginação em lote) não deveria
depender disso — os testes injetam um embedder falso e determinístico; só a
execução real (`make ingest` / `python -m rag.ingest.indexer`) carrega o BGE-M3
de verdade, via get_default_embedder().

Uso:
    python -m rag.ingest.indexer
    python -m rag.ingest.indexer --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rag.settings import settings

Embedder = Callable[[list[str]], list[list[float]]]

CHROMA_COLLECTION_NAME = "clausula_chunks"
BM25_INDEX_FILENAME = "bm25_index.pkl"

# Chunk maior que isto vira mais de uma "passagem" pro embedder? Não — o BGE-M3
# aceita até 8192 tokens, bem acima do nosso teto de chunk (1200). Não há
# truncamento a fazer aqui; a constante existe só para deixar isso explícito
# caso o teto de chunking mude no futuro.
EMBEDDER_MAX_TOKENS = 8192


@dataclass
class IndexReport:
    chunks_indexed: int = 0
    dense_batches: int = 0
    embedding_dim: int = 0
    bm25_vocabulary_size: int = 0


def load_chunks(path: Path | None = None) -> list[dict]:
    """Lê data/processed/chunks.jsonl (gerado pela Fase 2)."""
    path = path or settings.chunks_file
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe. Rode `python -m rag.ingest.pipeline` primeiro (Fase 2)."
        )
    chunks = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


# --- tokenização compartilhada ------------------------------------------------
#
# A MESMA função tokeniza no momento de indexar (aqui) e no momento de buscar
# (retrieval/hybrid.py importa esta função). Se as duas divergissem, o BM25
# buscaria com um vocabulário diferente do que foi indexado — silenciosamente
# devolvendo menos resultados do que deveria, sem erro nenhum.

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Minúsculas, sem acento, só sequências alfanuméricas.

    Simples de propósito: BM25 quer casamento de termo, não análise
    morfológica. "cláusula" e "clausula" devem ser o mesmo token.
    """
    normalized = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(c for c in normalized if not unicodedata.combining(c))
    return _TOKEN_RE.findall(without_accents)


# --- embedder ------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_default_embedder() -> Embedder:
    """Carrega o BGE-M3 real. Cacheado — só carrega o modelo uma vez por processo.

    Isolado numa função própria (em vez de import no topo do módulo) para que
    importar rag.ingest.indexer não exija ter sentence-transformers instalado
    nem baixe o modelo — só acontece quando esta função é de fato chamada.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embedding_model)

    def embed(texts: list[str]) -> list[list[float]]:
        # normalize_embeddings=True: os vetores saem unitários, então a
        # distância L2 do Chroma vira monótona com a similaridade de cosseno.
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    return embed


# --- índice denso (ChromaDB) ----------------------------------------------------


def _chroma_client():
    import chromadb

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def get_collection(client=None):
    client = client or _chroma_client()
    return client.get_or_create_collection(
        CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _chunk_metadata(chunk: dict) -> dict:
    """Metadados que acompanham o vetor no Chroma.

    O texto vai em `documents`, não aqui — Chroma metadata não aceita bem
    strings longas em todo backend, e não precisamos filtrar por texto.
    """
    return {
        "document": chunk["document"],
        "doc_type": chunk.get("doc_type") or "",
        "clause_path": chunk.get("clause_path") or "",
        "clause_title": chunk.get("clause_title") or "",
        "page_start": chunk.get("page_start", 0),
        "page_end": chunk.get("page_end", 0),
        "token_count": chunk.get("token_count", 0),
        "fallback": bool(chunk.get("fallback", False)),
    }


def build_dense_index(
    chunks: list[dict],
    embedder: Embedder,
    *,
    client=None,
    batch_size: int = 64,
    reset: bool = True,
) -> tuple[int, int]:
    """Embeda os chunks em lote e grava no ChromaDB. Devolve (batches, dim)."""
    client = client or _chroma_client()

    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass  # coleção ainda não existia — tudo bem

    collection = get_collection(client)
    dim = 0
    batches = 0

    for batch in _in_batches(chunks, batch_size):
        vectors = embedder([c["text"] for c in batch])
        dim = len(vectors[0]) if vectors else dim
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=vectors,
            documents=[c["text"] for c in batch],
            metadatas=[_chunk_metadata(c) for c in batch],
        )
        batches += 1

    return batches, dim


def _in_batches(items: list[dict], size: int) -> Iterable[list[dict]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


# --- índice léxico (BM25) -------------------------------------------------------


def build_lexical_index(chunks: list[dict], path: Path | None = None) -> int:
    """Tokeniza e constrói o BM25, persistido em pickle.

    Guarda o objeto BM25Okapi pronto + a lista de chunk_ids na mesma ordem do
    corpus tokenizado — é essa lista que traduz "índice 7 do corpus" de volta
    para "qual chunk_id", na hora da busca.
    """
    from rank_bm25 import BM25Okapi

    path = path or (settings.chroma_dir / BM25_INDEX_FILENAME)
    path.parent.mkdir(parents=True, exist_ok=True)

    chunk_ids = [c["chunk_id"] for c in chunks]
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    vocabulary = {tok for doc in tokenized_corpus for tok in doc}

    with path.open("wb") as fh:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, fh)

    return len(vocabulary)


# --- orquestração ----------------------------------------------------------------


def build_index(
    *,
    chunks: list[dict] | None = None,
    embedder: Embedder | None = None,
    batch_size: int = 64,
) -> IndexReport:
    """Constrói os dois índices a partir dos chunks. Ponto de entrada único.

    `chunks` e `embedder` são parâmetros (não lidos de settings direto) para
    que os testes possam injetar um corpus pequeno e um embedder falso sem
    tocar em disco além de um diretório temporário.
    """
    chunks = chunks if chunks is not None else load_chunks()
    if not chunks:
        raise ValueError("Nenhum chunk para indexar. Rode a Fase 2 primeiro.")

    embedder = embedder or get_default_embedder()

    batches, dim = build_dense_index(chunks, embedder, batch_size=batch_size)
    vocab_size = build_lexical_index(chunks)

    return IndexReport(
        chunks_indexed=len(chunks),
        dense_batches=batches,
        embedding_dim=dim,
        bm25_vocabulary_size=vocab_size,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Fase 3 — constrói os índices denso e léxico")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print(f"Carregando chunks de {settings.chunks_file}...")
    chunks = load_chunks()
    print(f"{len(chunks)} chunks. Carregando o embedder ({settings.embedding_model})...")
    print("(primeira execução baixa o modelo — pode levar alguns minutos)")

    report = build_index(chunks=chunks, embedder=get_default_embedder(), batch_size=args.batch_size)

    print("\n" + "=" * 60)
    print(f"chunks indexados       : {report.chunks_indexed}")
    print(f"dimensão do embedding   : {report.embedding_dim}")
    print(f"lotes processados       : {report.dense_batches}")
    print(f"vocabulário BM25        : {report.bm25_vocabulary_size} termos")
    print(f"índice denso salvo em   : {settings.chroma_dir}")
    print("=" * 60)
    print("\nTeste com: python -m rag.retrieval.cli \"sua pergunta aqui\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
