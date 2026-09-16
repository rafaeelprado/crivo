"""Testes da Fase 5 — API completa.

Nada aqui chama a Claude API, o BGE-M3 ou o reranker de verdade: os
endpoints que dependem da Fase 4 (`/query`, `/query/stream`) têm
`answer_question` trocado por um monkeypatch; o job de ingestão
(`/ingest`) tem `run_chunking`/`load_chunks`/`build_index` trocados da
mesma forma. `/documents` e `/metrics` são testados como funções puras,
com caminhos de arquivo injetados — nenhum dos dois toca `data/` real.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from rag.api import documents as documents_module
from rag.api import ingest_job
from rag.api import metrics as metrics_module
from rag.api.main import app
from rag.generation.answer import AnswerResult
from rag.generation.verify import CitationVerification
from rag.retrieval.hybrid import RetrievedChunk
from rag.retrieval.rerank import RankedChunk
from rag.settings import settings


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_ingest_state():
    """O estado do job de ingestão é um singleton de módulo — sem isolar
    entre testes, um teste vaza estado pro próximo.
    """
    ingest_job._STATE.__init__()
    yield
    ingest_job._STATE.__init__()


# --- /health e /config (sanidade rápida, cobertura completa em test_health.py) ---


def test_health_ainda_responde_200(client):
    assert client.get("/health").status_code == 200


# --- /documents ------------------------------------------------------------------


def test_list_documents_une_manifest_e_chunks(tmp_path):
    chunks_file = tmp_path / "chunks.jsonl"
    chunks_file.write_text(
        "\n".join(
            json.dumps({"document": doc})
            for doc in ["a.pdf", "a.pdf", "b.pdf"]
        ),
        encoding="utf-8",
    )

    manifest_file = tmp_path / "manifest.csv"
    manifest_file.write_text(
        "filename,doc_type,source_url\n"
        "a.pdf,contrato_administrativo,http://exemplo.com/a\n"
        "c.pdf,termo_aditivo,\n",
        encoding="utf-8",
    )

    result = documents_module.list_documents(chunks_file=chunks_file, manifest_file=manifest_file)

    assert result.total_documents == 3  # a, b (só em chunks), c (só em manifest)
    assert result.total_chunks == 3
    by_name = {d.filename: d for d in result.documents}
    assert by_name["a.pdf"].chunk_count == 2
    assert by_name["a.pdf"].doc_type == "contrato_administrativo"
    assert by_name["b.pdf"].chunk_count == 1
    assert by_name["b.pdf"].doc_type == ""  # não está no manifest
    assert by_name["c.pdf"].chunk_count == 0  # está no manifest, mas não foi chunkado


def test_list_documents_sem_nenhum_arquivo(tmp_path):
    result = documents_module.list_documents(
        chunks_file=tmp_path / "nao_existe.jsonl", manifest_file=tmp_path / "nao_existe.csv"
    )
    assert result.total_documents == 0
    assert result.total_chunks == 0


def test_endpoint_documents_reflete_a_funcao(client, monkeypatch, tmp_path):
    from rag.api.schemas import DocumentsResponse

    fake = DocumentsResponse(total_documents=1, total_chunks=5, documents=[])
    monkeypatch.setattr(documents_module, "list_documents", lambda: fake)
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json()["total_documents"] == 1


# --- /metrics ----------------------------------------------------------------------


def test_compute_metrics_sem_log_usa_default_das_settings(monkeypatch, tmp_path):
    """log_file=None cai no default de settings.query_log_file — se esse
    arquivo não existir (caso comum num ambiente de teste limpo), o
    resultado deve ser o mesmo "vazio" de test_compute_metrics_com_log_inexistente.
    """
    monkeypatch.setattr(
        type(settings), "query_log_file", property(lambda self: tmp_path / "nao_existe.jsonl")
    )
    result = metrics_module.compute_metrics()
    assert result.total_queries == 0
    assert result.abstention_rate == 0.0
    assert result.avg_retrieval_top_score is None


def test_compute_metrics_com_log_inexistente(tmp_path):
    result = metrics_module.compute_metrics(log_file=tmp_path / "nao_existe.jsonl")
    assert result.total_queries == 0
    assert result.abstention_rate == 0.0
    assert result.avg_retrieval_top_score is None


def test_compute_metrics_agrega_registros(tmp_path):
    log_file = tmp_path / "queries.jsonl"
    registros = [
        {"abstained": True, "confidence": "baixa", "retries": 0, "retrieval_top_score": 0.02},
        {"abstained": False, "confidence": "alta", "retries": 0, "retrieval_top_score": 0.91},
        {"abstained": False, "confidence": "alta", "retries": 1, "retrieval_top_score": 0.78},
    ]
    log_file.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in registros), encoding="utf-8"
    )

    result = metrics_module.compute_metrics(log_file=log_file)
    assert result.total_queries == 3
    assert result.abstained_count == 1
    assert result.abstention_rate == pytest.approx(1 / 3)
    assert result.confidence_counts == {"baixa": 1, "alta": 2}
    assert result.avg_retries == pytest.approx((0 + 0 + 1) / 3)
    assert result.avg_retrieval_top_score == pytest.approx((0.02 + 0.91 + 0.78) / 3)


# --- /query e /query/stream --------------------------------------------------------


def _fake_ranked_chunk() -> RankedChunk:
    chunk = RetrievedChunk(
        chunk_id="c1",
        text=(
            "O atraso na entrega sujeitará a CONTRATADA a multa moratória de "
            "0,5% ao dia sobre o valor da parcela em atraso."
        ),
        document="contrato_a.pdf",
        doc_type="contrato_administrativo",
        clause_path="CLÁUSULA QUARTA",
        clause_title="DAS PENALIDADES",
        page_start=3,
        page_end=3,
        dense_rank=0,
        lexical_rank=0,
        fused_score=1.0,
    )
    return RankedChunk(chunk=chunk, rerank_score=0.95)


def _fake_answer_result(query: str) -> AnswerResult:
    return AnswerResult(
        query=query,
        answer="A multa é de 0,5% ao dia.",
        confidence="alta",
        insufficient_context=False,
        abstained=False,
        abstain_reason="",
        citations=[
            CitationVerification(
                source_id=1,
                document="contrato_a.pdf",
                clause="CLÁUSULA QUARTA",
                page=3,
                quote="multa moratória de 0,5% ao dia",
                match_score=0.98,
                threshold=0.90,
                verified=True,
            )
        ],
        retrieval_top_score=0.91,
        retries=0,
        sources=[_fake_ranked_chunk()],
    )


def test_query_endpoint_recusa_sem_api_key(client, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    response = client.post("/query", json={"query": "qual a multa por atraso"})
    assert response.status_code == 503


def test_query_endpoint_devolve_resposta_com_citacao(client, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-teste-valido")
    monkeypatch.setattr(
        "rag.api.query_service.answer_question",
        lambda query, **kwargs: _fake_answer_result(query),
    )

    response = client.post("/query", json={"query": "qual a multa por atraso na entrega"})

    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is False
    assert len(body["citations"]) == 1
    assert body["citations"][0]["verified"] is True
    # "texto-fonte" (Fase 6): o chunk inteiro por trás da citação, não só o
    # `quote` curto — é o que a UI mostra quando o usuário clica na citação.
    assert len(body["sources"]) == 1
    assert body["sources"][0]["source_id"] == 1
    assert body["sources"][0]["document"] == "contrato_a.pdf"
    assert "multa moratória" in body["sources"][0]["text"]


def test_query_endpoint_valida_pergunta_muito_curta(client, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-teste-valido")
    response = client.post("/query", json={"query": "oi"})
    assert response.status_code == 422


def test_query_endpoint_repassa_o_filtro_de_documentos(client, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-teste-valido")
    received = {}

    def spy_answer_question(query, **kwargs):
        received.update(kwargs)
        return _fake_answer_result(query)

    monkeypatch.setattr("rag.api.query_service.answer_question", spy_answer_question)

    response = client.post(
        "/query",
        json={"query": "qual a multa por atraso na entrega", "documents": ["contrato_a.pdf"]},
    )

    assert response.status_code == 200
    assert received["document_filter"] == ["contrato_a.pdf"]


def test_query_stream_emite_status_e_result(client, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-teste-valido")

    def fake_answer_question(query, *, top_k=None, rerank_top_n=None, on_stage=None, **kwargs):
        if on_stage:
            on_stage("buscando fontes")
            on_stage("gerando resposta")
        return _fake_answer_result(query)

    monkeypatch.setattr("rag.generation.answer.answer_question", fake_answer_question)
    monkeypatch.setattr("rag.api.query_stream.answer_question", fake_answer_question)

    with client.stream(
        "POST", "/query/stream", json={"query": "qual a multa por atraso na entrega"}
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "event: status" in body
    assert "event: result" in body
    assert "multa" in body.lower()


# --- /ingest e /ingest/status --------------------------------------------------------


def test_ingest_dispara_job_e_reporta_sucesso(client, monkeypatch):
    monkeypatch.setattr(ingest_job, "run_chunking", lambda: 0)
    monkeypatch.setattr(ingest_job, "load_chunks", lambda: [{"chunk_id": "c1"}, {"chunk_id": "c2"}])

    class FakeReport:
        chunks_indexed = 2

    monkeypatch.setattr(ingest_job, "build_index", lambda **kwargs: FakeReport())
    monkeypatch.setattr(ingest_job, "get_default_embedder", lambda: (lambda texts: [[0.0]]))

    response = client.post("/ingest")
    assert response.status_code == 202

    status = client.get("/ingest/status").json()
    assert status["state"] == "done"
    assert status["chunks_written"] == 2
    assert status["chunks_indexed"] == 2


def test_ingest_reporta_erro_quando_chunking_falha(monkeypatch):
    monkeypatch.setattr(ingest_job, "run_chunking", lambda: 1)  # Fase 2 não passou

    ingest_job.run_ingest_job()

    status = ingest_job.get_status()
    assert status.state == "error"
    assert "Chunking" in status.error


def test_ingest_status_comeca_idle():
    assert ingest_job.get_status().state == "idle"
