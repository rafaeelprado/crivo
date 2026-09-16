"""Ingestão — Fase 2.

parser.py    PDF -> texto com número de página e coordenadas (PyMuPDF)
chunker.py   texto -> chunks por cláusula, com clause_path nos metadados
indexer.py   chunks -> embeddings BGE-M3 + ChromaDB + índice BM25
pipeline.py  orquestra os três acima (`make ingest`)
"""
