"""Fase 2 — pipeline de ingestão.

Lê os PDFs de data/raw, aplica parser + chunker e grava data/processed/chunks.jsonl.
Ao final imprime o relatório que decide se a Fase 2 passou no critério de aceite.

Uso:
    python -m rag.ingest.pipeline
    python -m rag.ingest.pipeline --limit 3 --verbose
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

from rag.ingest.chunker import ChunkingReport, chunk_document
from rag.ingest.parser import parse_pdf
from rag.settings import settings

# Critérios de aceite da Fase 2, definidos no plano.
MIN_CLAUSE_PATH_RATE = 0.90
MAX_FALLBACK_RATE = 0.20


def load_doc_types() -> dict[str, str]:
    """Lê o doc_type de cada arquivo do manifest.csv, se ele existir."""
    if not settings.manifest_file.exists():
        return {}
    with settings.manifest_file.open(encoding="utf-8", newline="") as fh:
        return {row["filename"]: row.get("doc_type", "") for row in csv.DictReader(fh)}


def ingest(limit: int | None = None, verbose: bool = False) -> int:
    settings.ensure_dirs()
    pdfs = sorted(settings.raw_dir.glob("*.pdf"))
    if limit:
        pdfs = pdfs[:limit]

    if not pdfs:
        print(f"Nenhum PDF em {settings.raw_dir}. Complete a Fase 1 primeiro.")
        return 1

    doc_types = load_doc_types()
    reports: list[ChunkingReport] = []
    total_written = 0

    with settings.chunks_file.open("w", encoding="utf-8") as out:
        for pdf in pdfs:
            try:
                doc = parse_pdf(pdf)
            except Exception as exc:
                print(f"  ! {pdf.name}: {exc}")
                continue

            chunks, report = chunk_document(
                doc,
                doc_type=doc_types.get(pdf.name, ""),
                max_tokens=settings.max_chunk_tokens,
            )
            reports.append(report)

            for chunk in chunks:
                out.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
                total_written += 1

            flag = " " if report.clause_path_rate >= MIN_CLAUSE_PATH_RATE else "!"
            print(
                f"{flag} {pdf.name[:48]:<48} "
                f"{doc.page_count:>4}p  {report.total_chunks:>4} chunks  "
                f"path {report.clause_path_rate:>4.0%}  "
                f"maior {report.max_tokens_seen:>5} tok"
            )
            if verbose and report.kinds:
                print(f"      hierarquia detectada: {report.kinds}")

    return summarize(reports, total_written)


def summarize(reports: list[ChunkingReport], total_written: int) -> int:
    """Imprime o veredito da Fase 2 e devolve o código de saída."""
    if not reports:
        print("\nNenhum documento processado.")
        return 1

    total_chunks = sum(r.total_chunks for r in reports)
    with_path = sum(r.with_clause_path for r in reports)
    fallback = sum(r.fallback_chunks for r in reports)
    oversized = sum(r.oversized_chunks for r in reports)

    path_rate = with_path / total_chunks if total_chunks else 0.0
    fallback_rate = fallback / total_chunks if total_chunks else 0.0

    print("\n" + "=" * 72)
    print(f"{len(reports)} documentos · {total_written} chunks · {settings.chunks_file}")
    print(f"  clause_path detectado : {path_rate:.1%}  (meta ≥ {MIN_CLAUSE_PATH_RATE:.0%})")
    print(f"  chunks em fallback    : {fallback_rate:.1%}  (meta ≤ {MAX_FALLBACK_RATE:.0%})")
    print(f"  chunks acima do teto  : {oversized}  (meta 0)")

    piores = sorted(reports, key=lambda r: r.clause_path_rate)[:5]
    ruins = [r for r in piores if r.clause_path_rate < MIN_CLAUSE_PATH_RATE]
    if ruins:
        print("\nDocumentos em que o detector foi pior — olhe estes primeiro:")
        for r in ruins:
            print(f"  {r.clause_path_rate:>5.0%}  {r.document}")
        print("\n  Abra um deles e veja qual padrão de cabeçalho o chunker não reconhece.")
        print("  A correção é acrescentar o padrão em HEADING_PATTERNS, não baixar a meta.")

    ok = (
        path_rate >= MIN_CLAUSE_PATH_RATE
        and fallback_rate <= MAX_FALLBACK_RATE
        and oversized == 0
    )
    print("\n" + ("Critério de aceite da Fase 2 atingido." if ok else "Fase 2 ainda não passou."))
    print("=" * 72)
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingestão: PDFs -> chunks.jsonl")
    parser.add_argument("--limit", type=int, help="processa só os N primeiros PDFs")
    parser.add_argument("--verbose", action="store_true", help="mostra a hierarquia detectada")
    args = parser.parse_args()
    return ingest(limit=args.limit, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
