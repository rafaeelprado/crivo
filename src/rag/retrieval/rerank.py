"""Fase 3 — reranking dos candidatos da busca híbrida.

RRF é rápido mas cego ao conteúdo: funde só posições, nunca lê o par
(pergunta, texto do chunk) junto. Um cross-encoder faz exatamente isso — recebe
os dois textos concatenados e produz um score de relevância calibrado para
aquele par específico, muito mais preciso que qualquer fusão de rankings
independentes. O custo é rodar um modelo por candidato, por isso ele entra
DEPOIS da busca híbrida (top-20) e não em lugar dela: rerankear 9990 contratos
seria inviável, rerankear 20 candidatos já filtrados é rápido.

Mesma escolha de design do indexer.py: o `scorer` é injetável. O cross-encoder
BGE-reranker-v2-m3 não precisa ser baixado para testar a lógica de ordenação e
corte em top_n.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from rag.retrieval.hybrid import RetrievedChunk
from rag.settings import settings

# (query, passagem) -> score de relevância. Score maior = mais relevante.
# Sem faixa fixa garantida entre modelos — por isso MIN_RERANK_SCORE (Fase 4)
# precisa ser calibrado empiricamente contra ESTE modelo específico, não
# copiado de outro projeto.
Scorer = Callable[[list[tuple[str, str]]], list[float]]


@dataclass
class RankedChunk:
    chunk: RetrievedChunk
    rerank_score: float

    def to_dict(self) -> dict:
        return {**self.chunk.to_dict(), "rerank_score": self.rerank_score}


@lru_cache(maxsize=1)
def get_default_scorer() -> Scorer:
    """Carrega o BAAI/bge-reranker-v2-m3 real. Isolado por import tardio —
    mesma razão do get_default_embedder: importar este módulo não deveria
    exigir sentence-transformers instalado nem baixar o modelo.
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(settings.reranker_model)

    def score(pairs: list[tuple[str, str]]) -> list[float]:
        return model.predict(list(pairs)).tolist()

    return score


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    *,
    top_n: int | None = None,
    scorer: Scorer | None = None,
) -> list[RankedChunk]:
    """Reordena os candidatos da busca híbrida pelo score do cross-encoder.

    Devolve no máximo top_n itens, do mais para o menos relevante. Não decide
    abstenção — isso é responsabilidade de quem chama (Fase 4), comparando
    rerank_score contra settings.min_rerank_score.
    """
    if not candidates:
        return []

    top_n = top_n or settings.rerank_top_n
    scorer = scorer or get_default_scorer()

    pairs = [(query, c.text) for c in candidates]
    scores = scorer(pairs)

    ranked = sorted(
        (RankedChunk(chunk=c, rerank_score=s) for c, s in zip(candidates, scores, strict=True)),
        key=lambda r: r.rerank_score,
        reverse=True,
    )
    return ranked[:top_n]
