"""Fase 3 — critério de aceite: denso-puro vs. híbrido+rerank em 10 consultas.

O PLANO.md pede uma comparação ESCRITA, não só "o pipeline roda". Este script
existe pra tornar essa comparação barata de refazer sempre que algo mudar no
índice (novo corpus, outro embedder, teto de chunk diferente) — sem isso, a
tentação é comparar de olho em 2 exemplos e chamar de validado.

Para cada pergunta, mostra lado a lado:
  MODO A — busca densa pura (só embeddings, sem BM25, sem reranking)
  MODO B — pipeline completo (RRF híbrido + reranking cross-encoder)

E grava o resultado em docs/comparacao_fase3.md, pronto pra colar no PLANO.md.

Uso:
    python scripts/compare_retrieval.py
    python scripts/compare_retrieval.py --top-k 3   # menos ruído por query
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.ingest.indexer import get_default_embedder  # noqa: E402
from rag.retrieval.hybrid import _chunk_store, dense_search, search  # noqa: E402
from rag.retrieval.rerank import get_default_scorer, rerank  # noqa: E402

# 10 perguntas cobrindo os tipos de cláusula mais comuns num contrato
# administrativo — as mesmas categorias que a Fase 7 vai usar no golden set,
# só que sem gabarito ainda (aqui é comparação qualitativa, não métrica).
QUERIES = [
    "quais as penalidades por atraso na entrega",
    "qual o prazo de vigência do contrato",
    "como funciona o reajuste de preços",
    "quais as hipóteses de rescisão contratual",
    "qual o valor da garantia contratual exigida",
    "multa moratória por descumprimento",
    "quais as obrigações da contratada",
    "qual o foro eleito para dirimir litígios",
    "quais as sanções administrativas previstas",
    "art. 78",  # controle: sabemos que esta é ambígua (ver diário técnico)
]


def _snippet(text: str, width: int = 110) -> str:
    return textwrap.shorten(text.replace("\n", " "), width=width, placeholder=" [...]")


def _dense_only(query: str, embedder, top_k: int, store) -> list[dict]:
    ids = dense_search(query, embedder, top_k)
    return [store[cid] for cid in ids if cid in store]


def _hybrid_reranked(query: str, embedder, scorer, top_k: int, rerank_top_n: int) -> list[dict]:
    candidates = search(query, top_k=top_k, embedder=embedder)
    ranked = rerank(query, candidates, top_n=rerank_top_n, scorer=scorer)
    return [{"clause_path": r.chunk.clause_path, "document": r.chunk.document,
              "text": r.chunk.text, "score": r.rerank_score} for r in ranked]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compara denso-puro vs híbrido+rerank")
    parser.add_argument("--top-k", type=int, default=5, help="quantos resultados mostrar por modo")
    args = parser.parse_args()

    print("Carregando embedder e reranker (primeira vez pode demorar)...")
    embedder = get_default_embedder()
    scorer = get_default_scorer()
    store = _chunk_store()

    lines = [
        "# Comparação — Fase 3: denso-puro vs. híbrido+rerank",
        "",
        f"Gerado por `scripts/compare_retrieval.py` em "
        f"{datetime.now(UTC):%Y-%m-%d %H:%M UTC}.",
        f"Corpus: {len(store)} chunks. Critério de aceite da Fase 2/3 no `PLANO.md`.",
        "",
    ]

    for i, query in enumerate(QUERIES, start=1):
        print(f"\n[{i}/{len(QUERIES)}] {query}")
        dense_results = _dense_only(query, embedder, args.top_k, store)
        hybrid_results = _hybrid_reranked(query, embedder, scorer, args.top_k, args.top_k)

        lines.append(f"## {i}. \"{query}\"")
        lines.append("")
        lines.append("**A — denso-puro (só embeddings):**")
        lines.append("")
        for chunk in dense_results:
            lines.append(f"- `{chunk['clause_path'] or '(sem cabeçalho)'}` — {chunk['document']}")
            lines.append(f"  > {_snippet(chunk['text'])}")
        lines.append("")
        lines.append("**B — híbrido (RRF) + reranking:**")
        lines.append("")
        for chunk in hybrid_results:
            lines.append(
                f"- `{chunk['clause_path'] or '(sem cabeçalho)'}` — {chunk['document']} "
                f"(score={chunk['score']:.3f})"
            )
            lines.append(f"  > {_snippet(chunk['text'])}")
        lines.append("")

        # Impressão no terminal também, pra acompanhar sem esperar o arquivo.
        print("  A (denso-puro):", [c["clause_path"] or "?" for c in dense_results[:3]])
        print("  B (híbrido+rerank):", [c["clause_path"] or "?" for c in hybrid_results[:3]])

    out_path = Path(__file__).resolve().parents[1] / "docs" / "comparacao_fase3.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nComparação completa gravada em {out_path}")
    print("Leia o arquivo, anote o que notar (onde B corrigiu A, onde os dois empataram)")
    print("e cole aqui na conversa — eu escrevo a análise final no PLANO.md e no diário.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
