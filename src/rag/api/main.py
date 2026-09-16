"""API do Crivo.

Fase 0: /health e /config, para provar que o esqueleto sobe.
Fase 5: acrescenta /documents, /query (síncrono e SSE), /ingest, /metrics.

Este módulo só faz roteamento — cada endpoint delega para um módulo
específico (`documents.py`, `query_service.py`, `query_stream.py`,
`ingest_job.py`, `metrics.py`). Os modelos Pydantic da fronteira ficam todos
em `schemas.py`.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from rag.api import documents as documents_module
from rag.api import ingest_job, query_service
from rag.api import metrics as metrics_module
from rag.api.query_stream import stream_query
from rag.api.schemas import (
    ConfigResponse,
    DocumentsResponse,
    HealthResponse,
    IngestStatusResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
)
from rag.settings import settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
)
logger = logging.getLogger("rag.api")

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Executa na subida do serviço: garante as pastas e avisa o que falta."""
    settings.ensure_dirs()
    logger.info("Crivo no ar — modelo de resposta: %s", settings.answer_model)
    if not settings.has_api_key:
        logger.warning(
            "ANTHROPIC_API_KEY não configurada. A busca funciona; a geração de "
            "respostas não. Copie .env.example para .env e preencha a chave."
        )
    yield


app = FastAPI(
    title="Crivo",
    description=(
        "Assistente de RAG para due diligence contratual. "
        "Responde perguntas sobre contratos citando cláusula, página e trecho literal — "
        "e se recusa a responder quando a resposta não está nos documentos."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["sistema"])
def health() -> HealthResponse:
    """Estado do serviço e do índice. É o critério de aceite da Fase 0."""
    pdf_count = len(list(settings.raw_dir.glob("*.pdf"))) if settings.raw_dir.exists() else 0

    chunk_count = 0
    if settings.chunks_file.exists():
        with settings.chunks_file.open(encoding="utf-8") as fh:
            chunk_count = sum(1 for line in fh if line.strip())

    return HealthResponse(
        status="ok",
        version=app.version,
        api_key_configured=settings.has_api_key,
        documents_indexed=pdf_count,
        chunks_available=chunk_count,
    )


@app.get("/config", response_model=ConfigResponse, tags=["sistema"])
def config() -> ConfigResponse:
    """Parâmetros ativos do pipeline. Útil ao comparar configurações na avaliação."""
    return ConfigResponse(
        answer_model=settings.answer_model,
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
        retrieve_top_k=settings.retrieve_top_k,
        rerank_top_n=settings.rerank_top_n,
        min_rerank_score=settings.min_rerank_score,
        citation_match_threshold=settings.citation_match_threshold,
    )


@app.get("/documents", response_model=DocumentsResponse, tags=["corpus"])
def list_documents() -> DocumentsResponse:
    """Une o manifest da Fase 1 com a contagem real de chunks da Fase 2."""
    return documents_module.list_documents()


@app.post("/query", response_model=QueryResponse, tags=["consulta"])
def query(request: QueryRequest) -> QueryResponse:
    """Pergunta síncrona: busca + geração + verificação de citação (Fase 4),
    numa única resposta JSON. É o endpoint certo para qualquer cliente que
    não precise de feedback de progresso.
    """
    if not settings.has_api_key:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY não configurada — a geração não está disponível.",
        )
    return query_service.run_query(
        request.query,
        top_k=request.top_k,
        rerank_top_n=request.rerank_top_n,
        document_filter=request.documents,
    )


@app.post("/query/stream", tags=["consulta"])
async def query_stream(request: QueryRequest) -> StreamingResponse:
    """A mesma pergunta, como Server-Sent Events de PROGRESSO — não de token.

    Ver o comentário no topo de `query_stream.py` para o porquê exato dessa
    distinção: o contrato de saída forçado por tool use (Fase 4) não permite
    streaming de token de forma significativa. Eventos: `status` (uma etapa
    do pipeline concluída), `result` (a resposta final, mesmo formato de
    `POST /query`), `error` (falha de infraestrutura).
    """
    if not settings.has_api_key:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY não configurada — a geração não está disponível.",
        )
    return StreamingResponse(
        stream_query(
            request.query,
            top_k=request.top_k,
            rerank_top_n=request.rerank_top_n,
            document_filter=request.documents,
        ),
        media_type="text/event-stream",
    )


@app.post("/ingest", response_model=IngestStatusResponse, status_code=202, tags=["corpus"])
def trigger_ingest(background_tasks: BackgroundTasks) -> IngestStatusResponse:
    """Dispara a reindexação (Fases 2 + 3) em segundo plano.

    Não recebe arquivo nenhum — assume que os PDFs já estão em
    `data/raw/` (o fluxo desde a Fase 1: `check_corpus.py` + manifest
    preenchido à mão). Devolve 202 imediatamente; acompanhe o progresso em
    `GET /ingest/status`.
    """
    if ingest_job.is_running():
        return IngestStatusResponse(**ingest_job.get_status().__dict__)
    background_tasks.add_task(ingest_job.run_ingest_job)
    return IngestStatusResponse(state="running")


@app.get("/ingest/status", response_model=IngestStatusResponse, tags=["corpus"])
def ingest_status() -> IngestStatusResponse:
    """Estado do último job de ingestão disparado por este processo.

    Estado em memória — reinicia com o processo, e não é compartilhado entre
    workers (ver o comentário em `ingest_job.py`). Suficiente para uma API
    de instância única; um job store real seria o próximo passo para mais
    de um worker.
    """
    return IngestStatusResponse(**ingest_job.get_status().__dict__)


@app.get("/metrics", response_model=MetricsResponse, tags=["sistema"])
def metrics() -> MetricsResponse:
    """Agrega `data/logs/queries.jsonl` (escrito pela Fase 4 a cada consulta).

    Sem estado próprio: é só uma leitura e agregação do log estruturado que
    já existia. A Fase 7 vai reler o mesmo arquivo para métricas mais ricas.
    """
    return metrics_module.compute_metrics()
