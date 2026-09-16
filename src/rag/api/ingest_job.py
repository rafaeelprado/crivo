"""Fase 5 — `POST /ingest`: reindexação em segundo plano.

Ingestão de verdade (parsear ~20 contratos, gerar embeddings com BGE-M3) leva
minutos, não milissegundos — devolver isso como resposta síncrona de uma
requisição HTTP trava a conexão e arrisca timeout no cliente. A solução mais
simples que resolve o problema de verdade: `BackgroundTasks` do FastAPI roda
o job depois de a resposta HTTP (202 Accepted) já ter voltado, e o cliente
consulta o progresso em `GET /ingest/status`.

**Limitação assumida de propósito, não esquecida:** o estado do job vive em
memória (`_STATE`), num dicionário do processo. Isso significa: reinicia a
API, perde o histórico do último job; rodando mais de um worker (`uvicorn
--workers N`), cada worker tem seu próprio estado, e `/ingest/status` pode
responder de um worker diferente do que recebeu o `/ingest`. Para uma API de
portfólio, single-worker, é suficiente e honesto sobre o trade-off. Um job
store real (Redis, banco) é o próximo passo natural se isto virasse produção
com mais de um processo.
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime

from rag.ingest.indexer import build_index, get_default_embedder, load_chunks
from rag.ingest.pipeline import ingest as run_chunking
from rag.settings import settings


@dataclass
class _IngestState:
    state: str = "idle"  # idle | running | done | error
    started_at: str | None = None
    finished_at: str | None = None
    documents_processed: int | None = None
    chunks_written: int | None = None
    chunks_indexed: int | None = None
    error: str | None = None


_STATE = _IngestState()
_lock = threading.Lock()


def get_status() -> _IngestState:
    return _STATE


def is_running() -> bool:
    return _STATE.state == "running"


def run_ingest_job() -> None:
    """Roda chunking + indexação, na ordem, e atualiza `_STATE` a cada etapa.

    Chamada via `BackgroundTasks.add_task` — o FastAPI/Starlette já executa
    uma função síncrona como esta numa thread separada, então não bloqueia o
    event loop nem a resposta HTTP que já foi enviada.
    """
    with _lock:
        if _STATE.state == "running":
            return
        _STATE.state = "running"
        _STATE.started_at = datetime.now(UTC).isoformat()
        _STATE.finished_at = None
        _STATE.error = None
        _STATE.documents_processed = None
        _STATE.chunks_written = None
        _STATE.chunks_indexed = None

    try:
        pdfs = sorted(settings.raw_dir.glob("*.pdf")) if settings.raw_dir.exists() else []
        _STATE.documents_processed = len(pdfs)

        exit_code = run_chunking()
        if exit_code != 0:
            raise RuntimeError(
                "Chunking (Fase 2) não atingiu o critério de aceite — veja o "
                "console do servidor para o relatório completo."
            )

        chunks = load_chunks()
        _STATE.chunks_written = len(chunks)

        report = build_index(chunks=chunks, embedder=get_default_embedder())
        _STATE.chunks_indexed = report.chunks_indexed

        _STATE.state = "done"
    except Exception as exc:  # noqa: BLE001 — job em background: captura tudo, reporta no status
        _STATE.state = "error"
        _STATE.error = f"{exc}\n{traceback.format_exc(limit=3)}"
    finally:
        _STATE.finished_at = datetime.now(UTC).isoformat()
