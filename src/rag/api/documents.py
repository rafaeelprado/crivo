"""Fase 5 — `GET /documents`: junta o manifest da Fase 1 com a contagem real
de chunks da Fase 2, por documento.

Por que duas fontes e não uma: o `manifest.csv` tem os metadados que só um
humano sabe preencher (`doc_type`, `source_url` — ver Fase 1); o
`chunks.jsonl` tem o resultado real do chunking. Um documento pode estar no
manifest e falhar no parsing (0 chunks) — isso é informação, não deveria
sumir juntando as duas fontes de qualquer jeito.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from rag.api.schemas import DocumentInfo, DocumentsResponse
from rag.settings import settings


def _chunk_counts_by_document(chunks_file: Path) -> Counter[str]:
    """Conta quantos chunks cada documento tem em chunks.jsonl, sem carregar
    o texto inteiro na memória — só o campo "document" de cada linha.
    """
    counts: Counter[str] = Counter()
    if not chunks_file.exists():
        return counts

    with chunks_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line).get("document", "")
            if doc:
                counts[doc] += 1
    return counts


def list_documents(
    *, chunks_file: Path | None = None, manifest_file: Path | None = None
) -> DocumentsResponse:
    """`chunks_file`/`manifest_file` são injetáveis — mesmo padrão de DI do
    resto do projeto (Embedder, Scorer, ModelCaller): os testes passam
    caminhos de arquivo temporários em vez de depender de `data/` real.
    """
    chunks_file = chunks_file or settings.chunks_file
    manifest_file = manifest_file or settings.manifest_file

    chunk_counts = _chunk_counts_by_document(chunks_file)

    manifest_rows: dict[str, dict] = {}
    if manifest_file.exists():
        with manifest_file.open(encoding="utf-8", newline="") as fh:
            manifest_rows = {row["filename"]: row for row in csv.DictReader(fh)}

    # União dos dois: um documento pode estar só no manifest (ainda não
    # chunkado) ou só nos chunks (manifest desatualizado) — nenhum dos dois
    # casos deveria desaparecer silenciosamente.
    all_filenames = sorted(set(manifest_rows) | set(chunk_counts))

    documents = [
        DocumentInfo(
            filename=name,
            doc_type=manifest_rows.get(name, {}).get("doc_type", ""),
            source_url=manifest_rows.get(name, {}).get("source_url", ""),
            chunk_count=chunk_counts.get(name, 0),
        )
        for name in all_filenames
    ]

    return DocumentsResponse(
        total_documents=len(documents),
        total_chunks=sum(chunk_counts.values()),
        documents=documents,
    )
