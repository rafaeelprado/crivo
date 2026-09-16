"""Fase 5 — `GET /metrics`: agrega o log estruturado que a Fase 4 já escreve.

Nenhuma métrica nova é calculada em tempo real na consulta — tudo já está em
`data/logs/queries.jsonl` (`rag.generation.query_log.log_query`, Fase 4).
Este módulo só lê e agrega. É um bom exemplo de por que logging estruturado
desde a Fase 4 compensa: a Fase 5 ganha um endpoint de métricas quase de
graça, e a Fase 7 (avaliação) vai reler o mesmo arquivo para métricas bem
mais ricas — nenhuma das duas precisou instrumentar nada de novo.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from rag.api.schemas import MetricsResponse
from rag.settings import settings


def compute_metrics(*, log_file: Path | None = None) -> MetricsResponse:
    """`log_file` é injetável — mesmo padrão de DI do resto do projeto, para
    os testes não dependerem de `data/logs/queries.jsonl` real.
    """
    log_file = log_file or settings.query_log_file
    if not log_file.exists():
        return MetricsResponse(
            total_queries=0,
            abstained_count=0,
            abstention_rate=0.0,
            confidence_counts={},
            avg_retries=0.0,
            avg_retrieval_top_score=None,
        )

    records: list[dict] = []
    with log_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    if total == 0:
        return MetricsResponse(
            total_queries=0,
            abstained_count=0,
            abstention_rate=0.0,
            confidence_counts={},
            avg_retries=0.0,
            avg_retrieval_top_score=None,
        )

    abstained = sum(1 for r in records if r.get("abstained"))
    confidence_counts = Counter(r.get("confidence", "") for r in records)
    retries = [r.get("retries", 0) for r in records]
    scores = [r["retrieval_top_score"] for r in records if r.get("retrieval_top_score") is not None]

    return MetricsResponse(
        total_queries=total,
        abstained_count=abstained,
        abstention_rate=abstained / total,
        confidence_counts=dict(confidence_counts),
        avg_retries=sum(retries) / total,
        avg_retrieval_top_score=(sum(scores) / len(scores)) if scores else None,
    )
