"""Fase 5 — os modelos Pydantic que cruzam a fronteira da API.

Regra do projeto (mesma do `settings.py`): toda forma de dado que atravessa
uma borda do sistema é validada, não confiada. Aqui a borda é HTTP — nenhum
endpoint devolve ou aceita um `dict` solto, sempre um modelo. Isso é o que
faz o `/docs` gerado pelo FastAPI ser útil de verdade (mostra o formato
exato de cada campo) e o que pega erro de integração em validação, não em
produção.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    api_key_configured: bool
    documents_indexed: int
    chunks_available: int


class ConfigResponse(BaseModel):
    answer_model: str
    embedding_model: str
    reranker_model: str
    retrieve_top_k: int
    rerank_top_n: int
    min_rerank_score: float
    citation_match_threshold: float


class DocumentInfo(BaseModel):
    filename: str
    doc_type: str
    source_url: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    documents: list[DocumentInfo]


class QueryRequest(BaseModel):
    query: str = Field(
        ..., min_length=3, max_length=500, description="Pergunta em linguagem natural"
    )
    top_k: int | None = Field(
        default=None, ge=1, le=100, description="Override de RETRIEVE_TOP_K"
    )
    rerank_top_n: int | None = Field(
        default=None, ge=1, le=20, description="Override de RERANK_TOP_N"
    )
    documents: list[str] | None = Field(
        default=None,
        description=(
            "Restringe a busca a estes documentos (filenames). "
            "Omitido/vazio = corpus inteiro."
        ),
    )


class CitationResponse(BaseModel):
    source_id: int
    document: str
    clause: str
    page: int
    quote: str
    match_score: float
    verified: bool


class SourceChunk(BaseModel):
    """O chunk completo por trás de um `source_id` — o "texto-fonte" que a
    Fase 6 mostra quando o usuário clica numa citação (documento, página e o
    texto inteiro do chunk, não só o `quote` curto que o modelo citou).
    """

    source_id: int
    document: str
    clause: str
    page_start: int
    page_end: int
    text: str
    rerank_score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    confidence: str
    insufficient_context: bool
    abstained: bool
    abstain_reason: str
    citations: list[CitationResponse]
    sources: list[SourceChunk]
    retrieval_top_score: float | None
    retries: int


class IngestStatusResponse(BaseModel):
    state: str  # "idle" | "running" | "done" | "error"
    started_at: str | None = None
    finished_at: str | None = None
    documents_processed: int | None = None
    chunks_written: int | None = None
    chunks_indexed: int | None = None
    error: str | None = None


class MetricsResponse(BaseModel):
    total_queries: int
    abstained_count: int
    abstention_rate: float
    confidence_counts: dict[str, int]
    avg_retries: float
    avg_retrieval_top_score: float | None
