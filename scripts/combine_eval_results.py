#!/usr/bin/env python
"""Combina resultados de 3 configuracoes (A, B, C) numa tabela de resumo."""

import argparse
import json
from pathlib import Path


def load_eval_results(json_path: Path) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def compute_metrics_from_config(results: list[dict]) -> dict:
    """Calcula as 4 metricas principais de uma configuracao.

    Todas as chaves do dict retornado sao ASCII puro de proposito, pra evitar
    problemas de normalizacao Unicode (NFC vs NFD) entre acentos digitados
    em momentos diferentes.
    """
    if not results:
        return {}

    sem_resposta = [r for r in results if r.get("categoria") == "sem_resposta"]
    if sem_resposta:
        corretas = sum(1 for r in sem_resposta if r.get('abstenção_correta'))
        abstencao_correta = (corretas / len(sem_resposta)) * 100
    else:
        abstencao_correta = 0

    com_citacoes = [r for r in results if r.get("model_citations_count", 0) > 0]
    if com_citacoes:
        media = sum(r.get("citation_accuracy", 0) for r in com_citacoes) / len(com_citacoes)
        acuracia_citacao = media * 100
    else:
        acuracia_citacao = 0

    deve_responder = [r for r in results if r.get("categoria") != "sem_resposta"]
    if deve_responder:
        indevidas = sum(1 for r in deve_responder if r.get('abstenção_indevida'))
        abstencao_indevida = (indevidas / len(deve_responder)) * 100
    else:
        abstencao_indevida = 0

    total_erros = sum(1 for r in results if r.get("erro"))

    return {
        "abstencao_correta": abstencao_correta,
        "acuracia_citacao": acuracia_citacao,
        "abstencao_indevida": abstencao_indevida,
        "total_erros": total_erros,
        "total_perguntas": len(results),
    }


def build_summary_table(all_metrics: dict) -> str:
    lines = []
    lines.append("# Fase 7 - Resultados da Avaliacao (Golden Set)\n")
    lines.append("## Resumo Agregado - Comparacao das 3 Configuracoes\n")
    lines.append(
        "| Configuracao | Abstencao Correta | Acuracia de Citacao | "
        "Abstencao Indevida | Erros |\n"
    )
    lines.append("|---|---|---|---|---|\n")

    configs_info = {
        "A": "Baseline ingenuo (denso puro, sem reranking)",
        "B": "Chunking estrutural + Hibrido (sem reranking)",
        "C": "Completo: Crivo (estrutural + hibrido + reranking + verificador)",
    }

    for config in ["A", "B", "C"]:
        if config not in all_metrics:
            continue
        m = all_metrics[config]
        lines.append(
            f"| **{config}** {configs_info[config]} | "
            f"{m['abstencao_correta']:.1f}% | "
            f"{m['acuracia_citacao']:.1f}% | "
            f"{m['abstencao_indevida']:.1f}% | "
            f"{m['total_erros']} |\n"
        )

    lines.append("\n## Interpretacao\n\n")
    lines.append(
        "- **Abstencao Correta**: das 10 perguntas sem resposta, "
        "quantas o modelo recusou corretamente? (ideal: 100%)\n"
    )
    lines.append(
        "- **Acuracia de Citacao**: das citacoes feitas, "
        "quantas bateram com texto verificado? (ideal: 100%)\n"
    )
    lines.append(
        "- **Abstencao Indevida**: das 40 perguntas com resposta, "
        "quantas o modelo recusou indevidamente? (ideal: 0%)\n"
    )
    lines.append("- **Erros**: excecoes ou timeouts durante a avaliacao\n")

    lines.append("\n## Ganhos de Cada Componente\n\n")

    if "A" in all_metrics and "B" in all_metrics:
        ganho_ab = all_metrics["B"]["acuracia_citacao"] - all_metrics["A"]["acuracia_citacao"]
        lines.append(f"**A -> B** (Chunking estrutural + Hibrido): +{ganho_ab:.1f}pp de acuracia de citacao\n\n")

    if "B" in all_metrics and "C" in all_metrics:
        ganho_bc = all_metrics["C"]["acuracia_citacao"] - all_metrics["B"]["acuracia_citacao"]
        lines.append(f"**B -> C** (Reranking + Verificador): +{ganho_bc:.1f}pp de acuracia de citacao\n\n")

    if "A" in all_metrics and "C" in all_metrics:
        ganho_ac = all_metrics["C"]["acuracia_citacao"] - all_metrics["A"]["acuracia_citacao"]
        lines.append(f"**A -> C** (Todos os componentes): +{ganho_ac:.1f}pp de acuracia de citacao\n\n")

    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Combina resultados de avaliacao")
    parser.add_argument("inputs", nargs=3, type=Path)
    parser.add_argument("--output", type=Path, default="data/PHASE7_RESULTS.md")
    args = parser.parse_args()

    all_metrics = {}
    configs = ["A", "B", "C"]

    for config, json_path in zip(configs, args.inputs):
        if not json_path.exists():
            print(f"Aviso: {json_path} nao encontrado, ignorando config {config}")
            continue
        print(f"Carregando {config}... {json_path}")
        results = load_eval_results(json_path)
        metrics = compute_metrics_from_config(results)
        all_metrics[config] = metrics
        print(f"  -> {len(results)} perguntas, {metrics['total_erros']} erros")

    markdown = build_summary_table(all_metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"\nRelatorio salvo em: {args.output}")
    print("\nPreview:\n")
    print(markdown)


if __name__ == "__main__":
    main()

