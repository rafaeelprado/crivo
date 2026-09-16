"""Fase 4 — verificação literal da citação: a defesa que mais importa.

As outras três defesas (prompt fechado, contexto numerado, abstenção por
limiar) tentam EVITAR que o modelo alucine. Esta aqui não confia que
funcionou — ela CONFERE, depois que a resposta já voltou, se cada "quote"
que o modelo diz ter tirado da fonte realmente existe, palavra por palavra
(com tolerância a diferenças pequenas), naquele chunk.

Por que fuzzy e não `quote in chunk.text` exato: o modelo pode copiar um
trecho com um espaço a mais, uma reticência a menos, ou normalizar um
travessão — sem ter alucinado nada. Comparação exata rejeitaria isso como
falso positivo. Fuzzy matching (similaridade de string, não semântica —
`rapidfuzz`) tolera esse tipo de ruído de transcrição sem abrir mão de
pegar uma citação de fato inventada, que tende a ter uma similaridade muito
mais baixa que o limiar.

É essa verificação que sustenta a promessa central do projeto: se uma
citação passa aqui, o trecho citado EXISTE no PDF de origem — não é só
"parece que existe".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.retrieval.rerank import RankedChunk
from rag.settings import settings


def _normalize(text: str) -> str:
    """Colapsa espaços/quebras de linha antes de comparar.

    Não normaliza acentos nem caixa — isso mudaria o que conta como "citação
    literal". O objetivo aqui é só absorver diferença de espaçamento, não
    tornar a comparação mais permissiva do que ela deveria ser.
    """
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class CitationVerification:
    source_id: int
    document: str
    clause: str
    page: int
    quote: str
    match_score: float  # 0.0 a 1.0
    threshold: float
    verified: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "document": self.document,
            "clause": self.clause,
            "page": self.page,
            "quote": self.quote,
            "match_score": self.match_score,
            "threshold": self.threshold,
            "verified": self.verified,
            "reason": self.reason,
        }


def verify_citation(
    citation: dict, chunks_by_source_id: dict[int, RankedChunk], *, threshold: float | None = None
) -> CitationVerification:
    """Confere uma citação contra o chunk que ela alega citar.

    Duas formas de falhar, tratadas separadamente porque pedem mensagens de
    erro diferentes: (1) o source_id nem existe entre as fontes fornecidas —
    o modelo inventou um número; (2) o source_id existe, mas o "quote" não
    bate com o texto daquele chunk — o modelo inventou (ou distorceu) a
    citação.
    """
    threshold = threshold if threshold is not None else settings.citation_match_threshold
    source_id = citation.get("source_id")
    quote = citation.get("quote", "")
    document = citation.get("document", "")
    clause = citation.get("clause", "")
    page = citation.get("page", 0)

    chunk = chunks_by_source_id.get(source_id)
    if chunk is None:
        return CitationVerification(
            source_id=source_id,
            document=document,
            clause=clause,
            page=page,
            quote=quote,
            match_score=0.0,
            threshold=threshold,
            verified=False,
            reason=f"source_id {source_id} não corresponde a nenhuma fonte fornecida",
        )

    from rapidfuzz import fuzz

    score = fuzz.partial_ratio(_normalize(quote), _normalize(chunk.chunk.text)) / 100.0
    verified = score >= threshold
    reason = "" if verified else "quote não bate literalmente com o texto da fonte citada"
    return CitationVerification(
        source_id=source_id,
        document=document,
        clause=clause,
        page=page,
        quote=quote,
        match_score=score,
        threshold=threshold,
        verified=verified,
        reason=reason,
    )


def verify_citations(
    citations: list[dict],
    chunks_by_source_id: dict[int, RankedChunk],
    *,
    threshold: float | None = None,
) -> list[CitationVerification]:
    """Verifica cada citação da resposta. Não decide o que fazer com o
    resultado — isso é orquestração (`answer.py`).
    """
    return [verify_citation(c, chunks_by_source_id, threshold=threshold) for c in citations]
