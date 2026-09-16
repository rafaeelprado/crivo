"""Fase 4 — logging estruturado de cada consulta.

Por que log estruturado (JSONL) e não só imprimir no terminal: a Fase 7
(avaliação) precisa reprocessar consultas passadas para medir métricas
agregadas — taxa de abstenção, distribuição de confiança, quantas citações
precisaram de reprocessamento antes de passar na verificação. Isso só é
possível se cada consulta grava um registro que um script relê depois, não
só aparece na tela e some.

Um arquivo por instalação (`data/logs/queries.jsonl`), um objeto JSON por
linha — formato trivial de fazer append (sem reescrever o arquivo inteiro a
cada consulta) e trivial de processar depois com `pandas.read_json(...,
lines=True)` na Fase 7.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from rag.settings import settings


def log_query(record: dict, *, path: Path | None = None) -> None:
    """Acrescenta um registro ao log de consultas. Nunca lança: logging não
    pode ser o motivo de uma consulta falhar para o usuário.
    """
    path = path or settings.query_log_file
    entry = {"timestamp": datetime.now(UTC).isoformat(), **record}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass
