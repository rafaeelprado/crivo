"""Fase 3 — busca pela linha de comando.

Uso:
    python -m rag.retrieval.cli "quais as penalidades por atraso na entrega"
    python -m rag.retrieval.cli "multa moratória" --no-rerank   # só RRF, pra comparar
    python -m rag.retrieval.cli "art. 78 rescisão" --show-scores
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from rag.retrieval.hybrid import RetrievedChunk, search
from rag.retrieval.rerank import rerank
from rag.settings import settings


def _print_chunk(rank: int, chunk: RetrievedChunk, *, extra_score: float | None, show_scores: bool):
    pages = (
        f"p.{chunk.page_start}"
        if chunk.page_start == chunk.page_end
        else f"p.{chunk.page_start}-{chunk.page_end}"
    )
    print(f"\n[{rank}] {chunk.document}  ({pages})")
    print(f"    cláusula: {chunk.clause_path or '(sem cabeçalho detectado)'}")
    if show_scores:
        scoreline = f"    rrf={chunk.fused_score:.4f}"
        if chunk.dense_rank is not None:
            scoreline += f"  denso#{chunk.dense_rank}"
        if chunk.lexical_rank is not None:
            scoreline += f"  léxico#{chunk.lexical_rank}"
        if extra_score is not None:
            scoreline += f"  rerank={extra_score:.4f}"
        print(scoreline)
    snippet = textwrap.shorten(chunk.text.replace("\n", " "), width=300, placeholder=" [...]")
    print(f"    {snippet}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Busca híbrida — Fase 3")
    parser.add_argument("query", help="a pergunta ou termo de busca")
    parser.add_argument("--top-k", type=int, default=settings.retrieve_top_k)
    parser.add_argument("--top-n", type=int, default=settings.rerank_top_n)
    parser.add_argument(
        "--no-rerank", action="store_true", help="mostra só a fusão RRF, sem cross-encoder"
    )
    parser.add_argument("--show-scores", action="store_true")
    args = parser.parse_args()

    print(f'Buscando: "{args.query}"')
    candidates = search(args.query, top_k=args.top_k)

    if not candidates:
        print("\nNenhum resultado. O índice foi construído? Rode: python -m rag.ingest.indexer")
        return 1

    if args.no_rerank:
        print(f"\n{len(candidates)} candidatos (fusão RRF, sem reranking):")
        for i, chunk in enumerate(candidates[: args.top_n], start=1):
            _print_chunk(i, chunk, extra_score=None, show_scores=args.show_scores)
        return 0

    print(f"{len(candidates)} candidatos recuperados. Aplicando reranking...")
    ranked = rerank(args.query, candidates, top_n=args.top_n)

    print(f"\nTop {len(ranked)} após reranking:")
    for i, item in enumerate(ranked, start=1):
        _print_chunk(i, item.chunk, extra_score=item.rerank_score, show_scores=args.show_scores)

    return 0


if __name__ == "__main__":
    sys.exit(main())
