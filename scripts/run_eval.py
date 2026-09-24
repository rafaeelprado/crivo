#!/usr/bin/env python
"""Fase 7 - Avaliacao contra golden set de 50 perguntas."""

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from rag.generation.answer import answer_question
    from rag.settings import Settings
except ImportError as e:
    print(f"Erro ao importar: {e}")
    print("Rode: pip install -e .")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ResultadoAvaliacao:
    """Resultado de uma pergunta em uma configuracao."""
    config: str
    question_id: str
    categoria: str
    pergunta: str
    resposta_esperada_resumo: str
    documento_fonte: str
    clausula_esperada: str

    model_answer: str = ""
    model_abstencao: bool = False
    model_citations_count: int = 0
    model_citations_verified: int = 0
    citation_accuracy: float = 0.0

    abstenção_correta: bool = False
    abstenção_indevida: bool = False
    erro: Optional[str] = None


class AvaliacaoFase7:
    """Avalia o golden set contra 3 configuracoes."""

    def __init__(self, golden_set_path: Path, settings: Settings):
        self.golden_set_path = Path(golden_set_path)
        self.settings = settings

    def carregar_perguntas(self) -> list[dict]:
        """Carrega as 50 perguntas do CSV."""
        perguntas = []
        with open(self.golden_set_path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                perguntas.append(row)
        return perguntas

    def rodar_pergunta(self, config: str, pergunta: dict) -> ResultadoAvaliacao:
        """Roda uma pergunta e retorna o resultado."""
        resultado = ResultadoAvaliacao(
            config=config,
            question_id=pergunta["id"],
            categoria=pergunta["categoria"],
            pergunta=pergunta["pergunta"],
            resposta_esperada_resumo=pergunta["resposta_esperada_resumo"],
            documento_fonte=pergunta["documento_fonte"],
            clausula_esperada=pergunta["clausula_esperada"],
        )

        try:
            logger.info(f"[{config}] {pergunta['id']}: {pergunta['pergunta'][:50]}...")

            # Roda o pipeline
            answer_result = answer_question(
                pergunta["pergunta"],
                disable_reranking=(config in ("A", "B")),
                dense_only=(config == "A"),
            )

            # Extrai resultado
            resultado.model_answer = answer_result.answer or "(vazio)"
            resultado.model_abstencao = answer_result.insufficient_context or not answer_result.answer

            # Metrica: abstenção correta
            if pergunta["categoria"] == "sem_resposta":
                resultado.abstenção_correta = resultado.model_abstencao
            else:
                resultado.abstenção_indevida = resultado.model_abstencao

            # Metrica: acuracia de citacao
            if answer_result.citations:
                verified_count = sum(1 for c in answer_result.citations if getattr(c, "verified", False))
                resultado.model_citations_verified = verified_count
                resultado.model_citations_count = len(answer_result.citations)
                resultado.citation_accuracy = verified_count / len(answer_result.citations)

        except Exception as e:
            logger.error(f"[{config}] {pergunta['id']} falhou: {e}")
            resultado.erro = str(e)

        return resultado

    def avaliar_configuracao(self, config: str) -> list[ResultadoAvaliacao]:
        """Avalia uma configuracao (todas as 50 perguntas)."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Configuracao: {config}")
        logger.info(f"{'='*60}\n")

        perguntas = self.carregar_perguntas()
        resultados = []

        for i, pergunta in enumerate(perguntas, 1):
            resultado = self.rodar_pergunta(config, pergunta)
            resultados.append(resultado)

        return resultados

    def calcular_metricas_agregadas(self, resultados: list[ResultadoAvaliacao]) -> dict:
        """Calcula metricas agregadas de uma configuracao."""
        if not resultados:
            return {}

        # Abstenção correta (% de perguntas "sem_resposta" que foram abstidas)
        sem_resposta = [r for r in resultados if r.categoria == "sem_resposta"]
        corretas = sum(1 for r in sem_resposta if r.abstenção_correta)
        taxa_abstenção_correta = (corretas / len(sem_resposta)) * 100 if sem_resposta else 0

        # Acuracia de citacao
        com_citacoes = [r for r in resultados if r.model_citations_count > 0]
        media_acurácia = sum(r.citation_accuracy for r in com_citacoes) / len(com_citacoes) if com_citacoes else 0

        # Abstenção indevida
        deve_responder = [r for r in resultados if r.categoria != "sem_resposta"]
        indevidas = sum(1 for r in deve_responder if r.abstenção_indevida)
        taxa_abstenção_indevida = (indevidas / len(deve_responder)) * 100 if deve_responder else 0

        return {
            "abstencao_correta_pct": taxa_abstenção_correta,
            "acurácia_citação_pct": media_acurácia * 100,
            "abstenção_indevida_pct": taxa_abstenção_indevida,
            "total_erros": sum(1 for r in resultados if r.erro),
        }

    def salvar_resultados(self, resultados: list[ResultadoAvaliacao], output_path: Path):
        """Salva em JSON e gera relatorio."""
        output_path = Path(output_path)

        # JSON
        with open(output_path, "w") as f:
            json.dump([asdict(r) for r in resultados], f, indent=2, default=str)
        logger.info(f"Salvo: {output_path}")

        # Markdown
        md_path = output_path.parent / f"{output_path.stem}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Resultados - Configuracao {resultados[0].config}\n\n")

            metricas = self.calcular_metricas_agregadas(resultados)
            f.write("## Metricas Agregadas\n\n")
            f.write(f"- Abstenção Correta: {metricas['abstencao_correta_pct']:.1f}%\n")
            f.write(f"- Acuracia de Citação: {metricas['acurácia_citação_pct']:.1f}%\n")
            f.write(f"- Abstenção Indevida: {metricas['abstenção_indevida_pct']:.1f}%\n")
            f.write(f"- Erros: {metricas['total_erros']}\n\n")

            f.write("## Detalhes\n\n")
            f.write("| ID | Pergunta | Respondeu? | Citacoes | Erro? |\n")
            f.write("|---|---|---|---|---|\n")
            for r in resultados:
                respondeu = "Não" if r.model_abstencao else "Sim"
                citacoes = f"{r.model_citations_verified}/{r.model_citations_count}" if r.model_citations_count > 0 else "—"
                erro = f"⚠️ {r.erro[:40]}" if r.erro else ""
                f.write(f"| {r.question_id} | {r.pergunta[:40]}... | {respondeu} | {citacoes} | {erro} |\n")

        logger.info(f"Relatorio: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Fase 7 - Avaliacao Golden Set")
    parser.add_argument("--config", choices=["A", "B", "C"], required=True,
                        help="Qual configuracao avaliar")
    parser.add_argument("--golden-set", type=Path,
                        default=Path(__file__).parent.parent / "data" / "golden_set_fase7.csv",
                        help="Caminho do golden set CSV")
    parser.add_argument("--output", type=Path,
                        default=None,
                        help="Onde salvar resultados (default: data/eval_results_<CONFIG>.json)")

    args = parser.parse_args()

    if not args.output:
        args.output = Path(__file__).parent.parent / "data" / f"eval_results_{args.config}.json"

    settings = Settings()
    avaliador = AvaliacaoFase7(args.golden_set, settings)

    resultados = avaliador.avaliar_configuracao(args.config)
    avaliador.salvar_resultados(resultados, args.output)

    # Resumo
    logger.info("\n" + "="*60)
    metricas = avaliador.calcular_metricas_agregadas(resultados)
    logger.info(f"RESUMO - Configuracao {args.config}")
    logger.info("="*60)
    logger.info(f"Abstenção Correta: {metricas['abstencao_correta_pct']:.1f}%")
    logger.info(f"Acuracia de Citação: {metricas['acurácia_citação_pct']:.1f}%")
    logger.info(f"Abstenção Indevida: {metricas['abstenção_indevida_pct']:.1f}%")
    logger.info(f"Erros: {metricas['total_erros']}")


if __name__ == "__main__":
    main()





