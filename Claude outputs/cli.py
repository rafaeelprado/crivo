"""Fase 4 — CLI para testar geração com citação verificada na sua máquina.

Uso:
    python -m rag.generation.cli "quais as penalidades por atraso na entrega"
    python -m rag.generation.cli "art. 78" --show-citations
"""

from __future__ import annotations

import argparse
import sys

from rag.generation.answer import answer_question
from rag.settings import settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Pergunta ao Crivo, com citação verificada")
    parser.add_argument("query", help="Pergunta em linguagem natural")
    parser.add_argument(
        "--show-citations", action="store_true", help="Mostra detalhe de cada citação"
    )
    args = parser.parse_args()

    if not settings.has_api_key:
        print(
            "ANTHROPIC_API_KEY não configurada no .env — a geração não pode rodar.\n"
            "A busca (Fase 3) continua disponível via `python -m rag.retrieval.cli`."
        )
        return 1

    result = answer_question(args.query)

    print(f"\nPergunta: {result.query}")
    print(f"Confiança: {result.confidence}")
    if result.abstained:
        print(f"ABSTENÇÃO — motivo: {result.abstain_reason}")
    print(f"\n{result.answer}\n")

    if not result.abstained and result.citations:
        print(f"Citações verificadas ({len(result.citations)}):")
        for c in result.citations:
            print(f"  [{c.source_id}] {c.document} — {c.clause} (p.{c.page})")
            if args.show_citations:
                print(f"      match={c.match_score:.0%}  \"{c.quote[:120]}\"")

    if result.retries:
        print(f"\n(reprocessado {result.retries}x por falha na verificação de citação)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
