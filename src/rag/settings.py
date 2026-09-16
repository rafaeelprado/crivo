"""Configuração central do projeto.

Toda constante ajustável do sistema mora aqui e vem do .env.
Regra do projeto: nenhum outro módulo lê variável de ambiente ou define
número mágico. Se você precisa de um parâmetro novo, ele nasce nesta classe.

Uso:
    from rag.settings import settings
    print(settings.answer_model)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raiz do repositório: .../sistema-rag
ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Configuração do sistema, carregada do .env com validação de tipos."""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Claude API ---
    anthropic_api_key: str = Field(
        default="",
        description="Chave da Claude API. Vazia = modo sem geração (busca ainda funciona).",
    )
    answer_model: str = "claude-haiku-4-5"

    # --- Modelos locais ---
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # --- Recuperação ---
    retrieve_top_k: int = 20
    rerank_top_n: int = 5
    min_rerank_score: float = 0.35
    citation_match_threshold: float = 0.90

    # --- Chunking ---
    max_chunk_tokens: int = 1200

    # --- Servidor ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # --- Caminhos (derivados, não vêm do .env) ---
    @property
    def raw_dir(self) -> Path:
        return ROOT_DIR / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return ROOT_DIR / "data" / "processed"

    @property
    def chroma_dir(self) -> Path:
        return ROOT_DIR / "data" / "chroma"

    @property
    def chunks_file(self) -> Path:
        return self.processed_dir / "chunks.jsonl"

    @property
    def manifest_file(self) -> Path:
        return self.raw_dir / "manifest.csv"

    @property
    def logs_dir(self) -> Path:
        return ROOT_DIR / "data" / "logs"

    @property
    def query_log_file(self) -> Path:
        return self.logs_dir / "queries.jsonl"

    @property
    def has_api_key(self) -> bool:
        """True se há uma chave da Claude API utilizável."""
        return bool(self.anthropic_api_key) and not self.anthropic_api_key.startswith(
            "sk-ant-troque"
        )

    def ensure_dirs(self) -> None:
        """Cria as pastas de dados se ainda não existirem."""
        for path in (self.raw_dir, self.processed_dir, self.chroma_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Instância única, cacheada — evita reler o .env a cada import."""
    return Settings()


settings = get_settings()
