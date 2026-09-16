"""Fase 1 — preenchimento automático de doc_type no manifest.csv.

Encontrar de novo a URL exata de onde cada PDF foi baixado é trabalho perdido
se você não guardou o link na hora. Este script resolve a parte que dá para
automatizar: ler o nome do arquivo e o texto do PDF e inferir o tipo de
documento, para você só revisar em vez de digitar 20 vezes do zero.

O que ele NÃO faz: preencher source_url. Esse campo é só rastreabilidade — o
pipeline de ingestão não lê esse valor. Se você tiver o link, ótimo; se não
tiver, deixe em branco. Não é bloqueante para nenhuma fase.

Regra de segurança: nunca sobrescreve um doc_type que você já preencheu à mão.
Só toca nas linhas em que a coluna está vazia.

Uso:
    python scripts/autofill_manifest.py
    python scripts/autofill_manifest.py --dry-run   # mostra sem gravar
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.settings import settings  # noqa: E402

# Ordem importa: o primeiro padrão que casar (no nome do arquivo OU no texto
# das primeiras páginas) decide o tipo. Do mais específico para o mais geral.
RULES: list[tuple[str, re.Pattern[str]]] = [
    ("termo_aditivo", re.compile(r"aditiv", re.IGNORECASE)),
    ("ata_registro_precos", re.compile(r"\bata\b.{0,20}registro.{0,20}pre[çc]o", re.IGNORECASE)),
    ("dispensa", re.compile(r"dispensa\s+de\s+licita|\bdispensa\b", re.IGNORECASE)),
    ("inexigibilidade", re.compile(r"inexigibilidade|\binex\b", re.IGNORECASE)),
    ("pregao", re.compile(r"preg[ãa]o", re.IGNORECASE)),
    ("concorrencia", re.compile(r"concorr[êe]ncia", re.IGNORECASE)),
    ("contrato_concessao", re.compile(r"concess[ãa]o", re.IGNORECASE)),
]

DEFAULT_TYPE = "contrato_administrativo"

# Quantas páginas iniciais ler para procurar pistas no texto — não precisa do
# PDF inteiro, o cabeçalho quase sempre entrega o tipo de instrumento.
PAGES_TO_SCAN = 2


def guess_doc_type(pdf_path: Path) -> tuple[str, str]:
    """Devolve (doc_type, pista) — a pista explica de onde veio o palpite."""
    for doc_type, pattern in RULES:
        if pattern.search(pdf_path.stem):
            return doc_type, f"nome do arquivo casou com '{pattern.pattern}'"

    try:
        import pymupdf

        with pymupdf.open(pdf_path) as doc:
            text = "".join(page.get_text("text") for page in doc[:PAGES_TO_SCAN])
    except Exception:
        text = ""

    for doc_type, pattern in RULES:
        if pattern.search(text):
            return doc_type, f"conteúdo das primeiras páginas casou com '{pattern.pattern}'"

    return DEFAULT_TYPE, "nenhum padrão específico encontrado — padrão"


def main() -> int:
    parser = argparse.ArgumentParser(description="Preenche doc_type automaticamente")
    parser.add_argument("--dry-run", action="store_true", help="mostra o resultado sem gravar")
    args = parser.parse_args()

    if not settings.manifest_file.exists():
        print(f"{settings.manifest_file} não existe. Rode scripts/check_corpus.py primeiro.")
        return 1

    with settings.manifest_file.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
        fieldnames = list(rows[0].keys()) if rows else []

    filled = 0
    for row in rows:
        if row.get("doc_type", "").strip():
            continue  # já preenchido à mão — nunca sobrescreve

        pdf_path = settings.raw_dir / row["filename"]
        if not pdf_path.exists():
            continue

        doc_type, hint = guess_doc_type(pdf_path)
        row["doc_type"] = doc_type
        filled += 1
        print(f"  {row['filename'][:55]:<55} -> {doc_type:<22} ({hint})")

    if filled == 0:
        print("Nenhuma linha vazia — todo doc_type já estava preenchido.")
        return 0

    print(f"\n{filled} linhas preenchidas automaticamente.")

    if args.dry_run:
        print("(--dry-run: nada foi gravado)")
        return 0

    with settings.manifest_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"gravado em {settings.manifest_file}")
    print(
        "\nRevise as linhas marcadas com 'padrão' acima — são as que o script não "
        "teve pista nenhuma e caiu no valor default. As outras você pode confiar."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
