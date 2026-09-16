"""Testes da Fase 0.

Critério de aceite: o serviço sobe e /health responde 200 com o formato esperado.
"""

from fastapi.testclient import TestClient

from rag.api.main import app

client = TestClient(app)


def test_health_responde_200():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_tem_o_formato_esperado():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert isinstance(body["documents_indexed"], int)
    assert isinstance(body["chunks_available"], int)
    assert isinstance(body["api_key_configured"], bool)


def test_config_expoe_os_parametros_do_pipeline():
    body = client.get("/config").json()
    assert body["rerank_top_n"] <= body["retrieve_top_k"]
    assert 0.0 < body["citation_match_threshold"] <= 1.0
