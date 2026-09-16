"""Fase 4 — critério de aceite: 10 perguntas fora do corpus, 10 abstenções.

O PLANO.md pede prova, não promessa: rodar 10 perguntas que o corpus de
contratos públicos não tem como responder e confirmar que o sistema diz "não
sei" nas 10 — nenhuma resposta inventada por educação ou por "tentar ser
útil". Isso testa as quatro defesas juntas (limiar de rerank, prompt
fechado, verificação de citação, abstenção declarada pelo modelo), contra a
Claude API de verdade — por isso este script só roda na sua máquina, com
ANTHROPIC_API_KEY configurada e o índice da Fase 3 já construído
(`python -m rag.ingest.indexer`).

Uso:
    python scripts/test_abstention.py
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.generation.answer import answer_question  # noqa: E402
from rag.settings import settings  # noqa: E402

# Todas de assunto genuinamente fora do domínio do corpus (contratos públicos
# brasileiros) — nenhuma delas tem, nem deveria ter, uma fonte no índice.
QUERIES_FORA_DO_CORPUS = [
    "qual é a capital da Mongólia",
    "quantos átomos de hidrogênio tem uma molécula de glicose",
    "quem foi campeão da Copa do Mundo de 2022",
    "qual é a cotação do bitcoin hoje",
    "como faço um bolo de cenoura com cobertura de chocolate",
    "explique a teoria da relatividade geral de Einstein",
    "qual vai ser o clima em Aracaju amanhã",
    "quantos habitantes tem o Japão atualmente",
    "qual é a fórmula de Bhaskara",
    "como se diz 'bom dia' em francês",
]


def main() -> int:
    if not settings.has_api_key:
        print("ANTHROPIC_API_KEY não configurada no .env — não dá para rodar este teste.")
        return 1

    linhas = [
        "# Fase 4 — critério de aceite: 10 perguntas fora do corpus",
        "",
        f"Gerado por `scripts/test_abstention.py` em {datetime.now(UTC):%Y-%m-%d %H:%M UTC}.",
        "",
        "| # | Pergunta | Absteve? | Motivo |",
        "|---|---|---|---|",
    ]
    abstencoes = 0

    for i, query in enumerate(QUERIES_FORA_DO_CORPUS, start=1):
        print(f"[{i}/10] {query}")
        result = answer_question(query)
        status = "✅ sim" if result.abstained else "❌ NÃO — resposta inventada?"
        motivo = result.abstain_reason or "(nenhum — respondeu como se tivesse fonte)"
        print(f"    absteve={result.abstained}  motivo={motivo}")
        if result.abstained:
            abstencoes += 1
        else:
            print(f"    RESPOSTA DADA: {result.answer[:200]}")
        linhas.append(f"| {i} | {query} | {status} | {motivo} |")

    linhas.append("")
    linhas.append(f"**Resultado: {abstencoes}/10 abstenções.**")
    linhas.append(
        "Critério de aceite da Fase 4 (PLANO.md): 10/10. "
        + ("Atingido." if abstencoes == 10 else "NÃO atingido — investigar antes de fechar a fase.")
    )

    out_path = Path(__file__).resolve().parents[1] / "docs" / "abstencao_fase4.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(linhas), encoding="utf-8")

    print(f"\n{abstencoes}/10 abstenções. Relatório gravado em {out_path}")
    return 0 if abstencoes == 10 else 1


if __name__ == "__main__":
    sys.exit(main())
