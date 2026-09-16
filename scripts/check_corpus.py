"""Fase 1 — triagem do corpus.

Percorre data/raw/*.pdf, rejeita o que não serve e gera o manifest.csv.

Um PDF só entra no corpus se tiver camada de texto real. PDF escaneado sem OCR
vira ruído: o parser extrai zero caracteres e o chunk correspondente polui o
índice com vazio. Melhor descobrir isso agora que na Fase 3.

Uso:
    python scripts/check_corpus.py
    python scripts/check_corpus.py --min-chars 500
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.settings import settings  # noqa: E402

# Um PDF com menos que isto por página quase certamente é digitalizado sem OCR.
DEFAULT_MIN_CHARS_PER_PAGE = 200

COLUMNS = [
    "filename",
    "pages",
    "chars",
    "chars_per_page",
    "has_text_layer",
    "status",
    "doc_type",
    "source_url",
    "notes",
]


def analyze(pdf_path: Path, min_chars_per_page: int) -> dict:
    """Abre um PDF e mede se ele tem texto extraível suficiente."""
    import pymupdf

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:  # PDF corrompido ou protegido
        return {
            "filename": pdf_path.name,
            "pages": 0,
            "chars": 0,
            "chars_per_page": 0,
            "has_text_layer": False,
            "status": "erro",
            "doc_type": "",
            "source_url": "",
            "notes": f"não foi possível abrir: {exc}",
        }

    with doc:
        pages = doc.page_count
        chars = sum(len(page.get_text("text")) for page in doc)

    per_page = round(chars / pages) if pages else 0
    ok = per_page >= min_chars_per_page

    return {
        "filename": pdf_path.name,
        "pages": pages,
        "chars": chars,
        "chars_per_page": per_page,
        "has_text_layer": ok,
        "status": "aceito" if ok else "rejeitado",
        "doc_type": "",  # preencha à mão: contrato_administrativo, termo_aditivo, dispensa...
        "source_url": "",  # opcional — preencha se tiver o link, ignore se não tiver
        "notes": "" if ok else "pouco texto por página — provavelmente escaneado sem OCR",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Tria os PDFs do corpus e gera o manifest.csv")
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS_PER_PAGE,
        help=f"mínimo de caracteres por página (padrão: {DEFAULT_MIN_CHARS_PER_PAGE})",
    )
    args = parser.parse_args()

    settings.ensure_dirs()
    pdfs = sorted(settings.raw_dir.glob("*.pdf"))

    if not pdfs:
        print(f"Nenhum PDF em {settings.raw_dir}")
        print("Coloque os contratos lá e rode de novo. Meta da Fase 1: 20 a 30 documentos.")
        return 1

    rows = [analyze(p, args.min_chars) for p in pdfs]

    # Preserva doc_type / source_url já preenchidos à mão num manifest anterior.
    if settings.manifest_file.exists():
        with settings.manifest_file.open(encoding="utf-8", newline="") as fh:
            previous = {r["filename"]: r for r in csv.DictReader(fh)}
        for row in rows:
            old = previous.get(row["filename"])
            if old:
                row["doc_type"] = old.get("doc_type", "") or row["doc_type"]
                row["source_url"] = old.get("source_url", "") or row["source_url"]

    with settings.manifest_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    accepted = [r for r in rows if r["status"] == "aceito"]
    rejected = [r for r in rows if r["status"] != "aceito"]
    total_pages = sum(r["pages"] for r in accepted)

    print(f"\n{len(rows)} PDFs analisados — {len(accepted)} aceitos, {len(rejected)} rejeitados")
    print(f"{total_pages} páginas úteis no corpus")
    print(f"manifest: {settings.manifest_file}")

    if rejected:
        print("\nRejeitados:")
        for r in rejected:
            print(f"  - {r['filename']}: {r['notes']}")

    sem_tipo = [r["filename"] for r in accepted if not r["doc_type"]]
    if sem_tipo:
        print(f"\nFalta preencher doc_type de {len(sem_tipo)} documentos")
        print("Abra o manifest.csv e complete — a Fase 7 agrupa os resultados por esse campo.")
        print(
            "(source_url é opcional: só rastreabilidade, nenhum código do pipeline lê esse "
            "valor — preencha se ainda tiver o link à mão, ignore se não tiver.)"
        )

    if len(accepted) < 20:
        print(f"\nMeta da Fase 1: 20 documentos. Faltam {20 - len(accepted)}.")
        return 1

    print("\nCritério de aceite da Fase 1 atingido.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
