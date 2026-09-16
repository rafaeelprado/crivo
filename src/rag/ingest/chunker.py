"""Fase 2 — chunking estrutural de documentos jurídicos brasileiros.

O coração técnico do projeto.

Um splitter de 512 tokens corta a cláusula no meio: a penalidade fica num chunk
e a condição que a dispara em outro. O sistema recupera metade da regra e o LLM
completa o resto por conta própria — que é exatamente a alucinação que este
projeto existe para evitar.

Aqui o chunk é a *cláusula*, porque cláusula é ao mesmo tempo a unidade
semântica (a regra completa) e a unidade de citação (o que o advogado escreve
no parecer).

Hierarquia reconhecida:

    CLÁUSULA DÉCIMA SEGUNDA — DA RESCISÃO      nível 1
    Art. 78                                     nível 1
      § 2º / Parágrafo Único                    nível 2
        IV —                                    nível 3  (inciso)
          b)                                    nível 4  (alínea)

INVARIANTE DO PROJETO: o campo `text` de um chunk é sempre uma fatia contígua e
literal do documento — nunca uma concatenação montada por nós. É isso que torna
a verificação de citação da Fase 4 sólida: se o trecho citado casa com o chunk,
ele existe no PDF. A hierarquia viaja nos metadados (`clause_path`), e o prompt
a renderiza junto do texto na hora de montar o contexto.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from rag.ingest.parser import ParsedDocument

# Em português, o tokenizador do Claude e o do BGE-M3 ficam perto de 3,5
# caracteres por token. É estimativa, não medição — serve para decidir corte,
# não para faturar. O nome da função diz isso.
CHARS_PER_TOKEN = 3.5

ROMAN = r"(?=[MDCLXVI])M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"

# Cada padrão é ancorado no início da linha. A ordem importa: o primeiro que
# casar define o nível.
HEADING_PATTERNS: list[tuple[int, str, re.Pattern[str]]] = [
    (
        1,
        "clausula",
        re.compile(
            r"^\s*(CL[ÁA]USULA\s+[\wÀ-Ú]+(?:[-\s]+[\wÀ-Ú]+)?)"
            r"\s*(?:[-–—:.]\s*(?P<title>[^\n]{0,120}))?",
            re.IGNORECASE,
        ),
    ),
    (
        1,
        "artigo",
        re.compile(
            r"^\s*(Art(?:igo)?\.?\s*\d+\s*[ºo°]?(?:\s*-?[A-Z])?)"
            r"\s*(?:[-–—.]\s*(?P<title>[^\n]{0,120}))?",
            re.IGNORECASE,
        ),
    ),
    (
        2,
        "paragrafo",
        re.compile(
            r"^\s*(§\s*\d+\s*[ºo°]?|PAR[ÁA]GRAFO\s+[\wÀ-Ú]+)"
            r"\s*(?:[-–—.]\s*(?P<title>[^\n]{0,120}))?",
            re.IGNORECASE,
        ),
    ),
    (
        3,
        "inciso",
        re.compile(rf"^\s*({ROMAN})\s*[-–—.)]\s"),
    ),
    (
        4,
        "alinea",
        re.compile(r"^\s*([a-z])\s*\)\s"),
    ),
]

# Uma linha só é cabeçalho se for curta o bastante para ser um cabeçalho.
# "CLÁUSULA" no meio de um parágrafo de 400 caracteres é referência, não título.
MAX_HEADING_LINE_CHARS = 200


@dataclass
class Heading:
    offset: int
    level: int
    kind: str
    marker: str
    title: str


@dataclass
class Chunk:
    chunk_id: str
    document: str
    doc_type: str
    clause_path: str
    clause_title: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    token_count: int
    fallback: bool
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChunkingReport:
    """Números que a Fase 2 precisa reportar para o critério de aceite."""

    document: str
    total_chunks: int = 0
    with_clause_path: int = 0
    fallback_chunks: int = 0
    oversized_chunks: int = 0
    max_tokens_seen: int = 0
    headings_found: int = 0
    kinds: dict[str, int] = field(default_factory=dict)

    @property
    def clause_path_rate(self) -> float:
        return self.with_clause_path / self.total_chunks if self.total_chunks else 0.0


def estimate_tokens(text: str) -> int:
    """Estimativa de tokens a partir do número de caracteres."""
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def _clean_marker(marker: str) -> str:
    return re.sub(r"\s+", " ", marker).strip(" .:-–—")


# Caracteres que, no fim da linha anterior, indicam frase encerrada (ponto,
# ponto-e-vírgula, dois-pontos, fecha-parênteses/aspas). Usado só para o
# padrão "artigo" — ver _artigo_heading_context_ok.
SENTENCE_END_CHARS = ".;:)\"'”’"


def _artigo_heading_context_ok(prev_stripped: str, is_first_line: bool) -> bool:
    """Um "Art. N" só é cabeçalho de verdade se abre parágrafo novo.

    Bug real encontrado no corpus: uma citação como "conforme o disposto no
    art. 137 da Lei 14.133/2021" pode quebrar de linha exatamente em "art.
    137", por causa do reflow do PDF — e aí a linha começa igual a um
    cabeçalho de artigo de verdade. A diferença está na linha ANTERIOR: um
    cabeçalho real vem depois de linha em branco (fim do bloco anterior) ou
    do fim de uma frase; uma citação quebrada continua uma frase em
    andamento, então a linha anterior termina sem pontuação de fechamento
    (ex.: "...disposto no"). Sem essa checagem, "art. 137 da" virava
    clause_path — um cabeçalho fantasma que nunca existiu no contrato.
    """
    if is_first_line or not prev_stripped:
        return True
    return prev_stripped[-1] in SENTENCE_END_CHARS


def find_headings(text: str) -> list[Heading]:
    """Varre o texto linha a linha e devolve os cabeçalhos encontrados."""
    headings: list[Heading] = []
    offset = 0
    prev_stripped = ""
    is_first_line = True

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped and len(stripped) <= MAX_HEADING_LINE_CHARS:
            for level, kind, pattern in HEADING_PATTERNS:
                match = pattern.match(line)
                if not match:
                    continue
                if kind == "artigo" and not _artigo_heading_context_ok(
                    prev_stripped, is_first_line
                ):
                    continue
                title = ""
                if "title" in pattern.groupindex:
                    title = (match.group("title") or "").strip(" .:-–—")
                headings.append(
                    Heading(
                        offset=offset + (len(line) - len(line.lstrip())),
                        level=level,
                        kind=kind,
                        marker=_clean_marker(match.group(1)),
                        title=title,
                    )
                )
                break
        offset += len(line)
        prev_stripped = stripped
        is_first_line = False

    return headings


def _build_spans(headings: list[Heading], text_len: int) -> list[tuple[Heading, int, str]]:
    """Para cada cabeçalho, calcula onde termina o seu bloco e o caminho completo.

    O bloco de um cabeçalho vai até o próximo cabeçalho de nível igual ou
    superior. O caminho é a pilha de ancestrais: "CLÁUSULA XII > § 2º > IV".
    """
    spans: list[tuple[Heading, int, str]] = []
    stack: list[Heading] = []

    for i, heading in enumerate(headings):
        while stack and stack[-1].level >= heading.level:
            stack.pop()

        end = text_len
        for later in headings[i + 1 :]:
            if later.level <= heading.level:
                end = later.offset
                break

        path_parts = [h.marker for h in stack] + [heading.marker]
        spans.append((heading, end, " > ".join(path_parts)))
        stack.append(heading)

    return spans


def _split_long_text(text: str, base_offset: int, max_tokens: int) -> list[tuple[int, str]]:
    """Divide um bloco grande demais em fatias contíguas, cortando em parágrafo.

    Devolve (offset_absoluto, texto). Nenhuma fatia é remontada — cada uma
    continua sendo um pedaço literal do documento.
    """
    budget_chars = int(max_tokens * CHARS_PER_TOKEN)
    parts: list[tuple[int, str]] = []

    # Corta em parágrafo; se um único parágrafo estourar, corta em frase.
    blocks: list[tuple[int, str]] = []
    cursor = 0
    for para in re.split(r"(\n\n)", text):
        if para == "\n\n":
            cursor += 2
            continue
        if para:
            blocks.append((cursor, para))
            cursor += len(para)

    current_start: int | None = None
    current_end = 0
    for offset, block in blocks:
        if current_start is None:
            current_start, current_end = offset, offset + len(block)
            continue
        if (offset + len(block)) - current_start <= budget_chars:
            current_end = offset + len(block)
        else:
            parts.append((base_offset + current_start, text[current_start:current_end]))
            current_start, current_end = offset, offset + len(block)

    if current_start is not None:
        parts.append((base_offset + current_start, text[current_start:current_end]))

    # Último recurso: fatia dura, quando um parágrafo sozinho estoura o teto.
    final: list[tuple[int, str]] = []
    for offset, part in parts:
        if estimate_tokens(part) <= max_tokens:
            final.append((offset, part))
            continue
        for i in range(0, len(part), budget_chars):
            final.append((offset + i, part[i : i + budget_chars]))

    return [(o, t) for o, t in final if t.strip()]


def _fallback_chunks(
    doc: ParsedDocument, doc_type: str, max_tokens: int
) -> tuple[list[Chunk], ChunkingReport]:
    """Quando nenhuma hierarquia é detectada, cai para janelas de parágrafo.

    Não é o caminho feliz. A taxa de fallback é medida e reportada de propósito:
    é ela que diz se o detector precisa de mais padrões.
    """
    report = ChunkingReport(document=doc.filename)
    chunks: list[Chunk] = []

    for i, (offset, text) in enumerate(_split_long_text(doc.text, 0, max_tokens)):
        page_start, page_end = doc.page_range(offset, offset + len(text))
        tokens = estimate_tokens(text)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.filename}::fb{i:04d}",
                document=doc.filename,
                doc_type=doc_type,
                clause_path="",
                clause_title="",
                page_start=page_start,
                page_end=page_end,
                char_start=offset,
                char_end=offset + len(text),
                token_count=tokens,
                fallback=True,
                text=text,
            )
        )
        report.total_chunks += 1
        report.fallback_chunks += 1
        report.max_tokens_seen = max(report.max_tokens_seen, tokens)
        if tokens > max_tokens:
            report.oversized_chunks += 1

    return chunks, report


def chunk_document(
    doc: ParsedDocument,
    *,
    doc_type: str = "",
    max_tokens: int = 1200,
) -> tuple[list[Chunk], ChunkingReport]:
    """Converte um documento parseado em chunks por cláusula.

    Devolve os chunks e um relatório com os números do critério de aceite:
    taxa de clause_path, taxa de fallback e maior chunk gerado.
    """
    headings = find_headings(doc.text)

    if not headings:
        return _fallback_chunks(doc, doc_type, max_tokens)

    report = ChunkingReport(document=doc.filename, headings_found=len(headings))
    for h in headings:
        report.kinds[h.kind] = report.kinds.get(h.kind, 0) + 1

    spans = _build_spans(headings, len(doc.text))
    chunks: list[Chunk] = []

    # Preâmbulo: o que vem antes do primeiro cabeçalho (qualificação das partes,
    # preço, prazo). Tem conteúdo consultável e não pode ser descartado.
    if spans[0][0].offset > 0:
        preamble = doc.text[: spans[0][0].offset].strip()
        if estimate_tokens(preamble) > 20:
            for offset, part in _split_long_text(preamble, 0, max_tokens):
                chunks.append(
                    _make_chunk(
                        doc, doc_type, "PREÂMBULO", "", offset, part, len(chunks), False
                    )
                )

    # Só os cabeçalhos de nível 1 abrem chunk; os de nível 2+ só entram quando o
    # bloco do pai não cabe no teto de tokens.
    for index, (heading, end, path) in enumerate(spans):
        if heading.level != 1:
            continue

        block = doc.text[heading.offset : end]
        if estimate_tokens(block) <= max_tokens:
            chunks.append(
                _make_chunk(
                    doc, doc_type, path, heading.title, heading.offset, block, len(chunks), False
                )
            )
            continue

        # Bloco grande: desce um nível. Cada filho vira um chunk próprio,
        # e o texto do pai antes do primeiro filho também.
        children = [
            (h, e, p) for (h, e, p) in spans[index + 1 :] if h.offset < end and h.level == 2
        ]

        if not children:
            for offset, part in _split_long_text(block, heading.offset, max_tokens):
                chunks.append(
                    _make_chunk(
                        doc, doc_type, path, heading.title, offset, part, len(chunks), False
                    )
                )
            continue

        head_text = doc.text[heading.offset : children[0][0].offset]
        if estimate_tokens(head_text) > 20:
            # head_text pode ser tão grande quanto o bloco inteiro (quando o
            # primeiro filho só aparece bem depois do início da cláusula) —
            # por isso passa pelo mesmo splitter que os filhos, nunca vira um
            # chunk único sem checar o teto.
            for offset, part in _split_long_text(head_text, heading.offset, max_tokens):
                chunks.append(
                    _make_chunk(
                        doc, doc_type, path, heading.title, offset, part, len(chunks), False
                    )
                )

        for child, child_end, child_path in children:
            child_block = doc.text[child.offset : min(child_end, end)]
            for offset, part in _split_long_text(child_block, child.offset, max_tokens):
                chunks.append(
                    _make_chunk(
                        doc, doc_type, child_path, heading.title, offset,
                        part, len(chunks), False,
                    )
                )

    for chunk in chunks:
        report.total_chunks += 1
        if chunk.clause_path:
            report.with_clause_path += 1
        report.max_tokens_seen = max(report.max_tokens_seen, chunk.token_count)
        if chunk.token_count > max_tokens:
            report.oversized_chunks += 1

    return chunks, report


def _make_chunk(
    doc: ParsedDocument,
    doc_type: str,
    path: str,
    title: str,
    offset: int,
    text: str,
    index: int,
    fallback: bool,
) -> Chunk:
    stripped = text.strip()
    lead = len(text) - len(text.lstrip())
    start = offset + lead
    end = start + len(stripped)
    page_start, page_end = doc.page_range(start, end)
    return Chunk(
        chunk_id=f"{doc.filename}::{index:04d}",
        document=doc.filename,
        doc_type=doc_type,
        clause_path=path,
        clause_title=title,
        page_start=page_start,
        page_end=page_end,
        char_start=start,
        char_end=end,
        token_count=estimate_tokens(stripped),
        fallback=fallback,
        text=stripped,
    )
