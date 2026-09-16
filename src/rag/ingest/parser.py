"""Fase 2 — extração de texto de PDF preservando a página de origem.

Por que não basta `page.get_text()` e concatenar: o sistema precisa dizer
"página 27" na citação. Então guardamos, junto do texto corrido, o offset de
caractere onde cada página começa. Dado qualquer trecho, sabemos de que página
ele veio.

Uso:
    from rag.ingest.parser import parse_pdf
    doc = parse_pdf(Path("data/raw/contrato.pdf"))
    doc.page_at(1500)   # -> 3
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# Cabeçalho/rodapé repetido em toda página polui o texto e vira ruído no chunk.
# Linhas curtas que aparecem em quase toda página são removidas.
REPEATED_LINE_MIN_PAGES_RATIO = 0.6
REPEATED_LINE_MAX_CHARS = 90


@dataclass
class ParsedDocument:
    """Um PDF já convertido em texto, com o mapa de páginas preservado."""

    filename: str
    text: str
    page_starts: list[int] = field(default_factory=list)
    page_count: int = 0

    def page_at(self, char_index: int) -> int:
        """Número da página (1-based) que contém este offset de caractere."""
        if not self.page_starts:
            return 1
        # Busca binária pela última página que começa antes do índice.
        lo, hi = 0, len(self.page_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.page_starts[mid] <= char_index:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    def page_range(self, start: int, end: int) -> tuple[int, int]:
        """Primeira e última página cobertas pelo intervalo [start, end)."""
        return self.page_at(start), self.page_at(max(start, end - 1))


def _normalize(text: str) -> str:
    """Limpa artefatos típicos de extração de PDF, sem alterar o conteúdo.

    Só mexe em espaçamento, hifenização de quebra de linha e formas Unicode.
    Nada aqui pode alterar uma palavra — a citação precisa continuar literal.
    """
    # NFC, não NFKC. NFKC é destrutivo para texto jurídico: converte "Nº" em
    # "No" e "§" composto em formas equivalentes, quebrando a promessa de que o
    # chunk é literal quando o usuário abre o PDF para conferir a citação.
    text = unicodedata.normalize("NFC", text)
    # Palavra hifenizada quebrada em duas linhas: "resci-\nsão" -> "rescisão"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Espaços não separáveis viram espaço comum.
    text = text.replace("\xa0", " ")
    # Runs de espaço/tab, mas preservando as quebras de linha (o chunker usa).
    text = re.sub(r"[ \t]+", " ", text)
    # Três ou mais quebras viram duas — parágrafo é o máximo de separação.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_repeated_lines(pages: list[str]) -> list[str]:
    """Remove cabeçalhos e rodapés que se repetem na maioria das páginas."""
    if len(pages) < 3:
        return pages

    counts: dict[str, int] = {}
    for page in pages:
        for line in {ln.strip() for ln in page.splitlines() if ln.strip()}:
            if len(line) <= REPEATED_LINE_MAX_CHARS:
                counts[line] = counts.get(line, 0) + 1

    threshold = max(2, int(len(pages) * REPEATED_LINE_MIN_PAGES_RATIO))
    boilerplate = {line for line, n in counts.items() if n >= threshold}
    if not boilerplate:
        return pages

    cleaned = []
    for page in pages:
        kept = [ln for ln in page.splitlines() if ln.strip() not in boilerplate]
        cleaned.append("\n".join(kept))
    return cleaned


def parse_pdf(pdf_path: Path, *, strip_boilerplate: bool = True) -> ParsedDocument:
    """Extrai o texto de um PDF mantendo o mapa de páginas.

    Levanta ValueError se o PDF não tiver camada de texto — nesse caso ele não
    deveria ter passado por scripts/check_corpus.py.
    """
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        raw_pages = [page.get_text("text") for page in doc]

    if strip_boilerplate:
        raw_pages = _strip_repeated_lines(raw_pages)

    pages = [_normalize(p) for p in raw_pages]

    if sum(len(p) for p in pages) == 0:
        raise ValueError(
            f"{pdf_path.name} não tem texto extraível. "
            "Rode scripts/check_corpus.py para triar o corpus."
        )

    chunks: list[str] = []
    page_starts: list[int] = []
    cursor = 0
    for page in pages:
        page_starts.append(cursor)
        block = page + "\n\n"
        chunks.append(block)
        cursor += len(block)

    return ParsedDocument(
        filename=pdf_path.name,
        text="".join(chunks),
        page_starts=page_starts,
        page_count=len(pages),
    )


def parse_text(filename: str, text: str, *, page_starts: list[int] | None = None):
    """Constrói um ParsedDocument a partir de texto puro.

    Existe para os testes e para ingerir Markdown sem passar por PDF.
    """
    normalized = _normalize(text)
    return ParsedDocument(
        filename=filename,
        text=normalized,
        page_starts=page_starts or [0],
        page_count=len(page_starts) if page_starts else 1,
    )
