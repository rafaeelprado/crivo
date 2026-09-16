.PHONY: help install dev api ui test lint ingest search eval up down clean

help:  ## Mostra os comandos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Instala o projeto em modo editável
	pip install -e ".[dev]"

dev:  ## Instala tudo, incluindo as libs de avaliação
	pip install -e ".[dev,eval]"

api:  ## Sobe a API em modo recarregável
	uvicorn rag.api.main:app --reload --host 0.0.0.0 --port 8000

ui:  ## Sobe a interface Streamlit
	streamlit run ui/app.py

test:  ## Roda os testes
	pytest -q

lint:  ## Verifica e formata o código
	ruff check --fix src tests ui scripts
	ruff format src tests ui scripts

corpus:  ## Fase 1 — tria os PDFs de data/raw e gera o manifest.csv
	python scripts/check_corpus.py

ingest:  ## Fase 2/3 — extrai, faz chunking e indexa
	python -m rag.ingest.pipeline

search:  ## Fase 3 — busca pela linha de comando. Ex: make search Q="multa por atraso"
	python -m rag.retrieval.cli "$(Q)"

eval:  ## Fase 7 — roda a suíte de avaliação
	python -m rag.eval.run_eval

up:  ## Sobe tudo no Docker
	docker compose up --build

down:  ## Derruba os containers
	docker compose down

clean:  ## Apaga índice e chunks (não apaga os PDFs)
	rm -rf data/chroma/* data/processed/*
	touch data/chroma/.gitkeep data/processed/.gitkeep
