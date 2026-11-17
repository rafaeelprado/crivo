"""Testes do servidor MCP do Crivo. A API é simulada com httpx.MockTransport,
então não precisa de modelo, índice, chave de API nem do SDK do MCP."""

from __future__ import annotations

import json

import httpx
import pytest

from rag import mcp_server
from rag.mcp_server import (
    CrivoError,
    format_answer,
    listar_documentos,
    responder_com_citacao,
    status_do_crivo,
)

RESPOSTA = {
    "query": "Qual a vigência?",
    "answer": "Até 31 de dezembro de 2026.",
    "confidence": "alta",
    "insufficient_context": False,
    "abstained": False,
    "abstain_reason": "",
    "citations": [
        {
            "source_id": 1,
            "document": "contrato_237.pdf",
            "clause": "CLÁUSULA SEXTA",
            "page": 3,
            "quote": "vigência até 31 de dezembro de 2026",
            "match_score": 1.0,
            "verified": True,
        }
    ],
    "sources": [],
    "retrieval_top_score": 0.9,
    "retries": 0,
}

ABSTENCAO = {
    **RESPOSTA,
    "answer": "Não há informação suficiente nas fontes disponíveis para responder com confiança.",
    "abstained": True,
    "abstain_reason": "melhor resultado da busca fica abaixo do limiar",
    "citations": [],
}


def _client(handler) -> httpx.Client:
    return httpx.Client(base_url="http://crivo.test", transport=httpx.MockTransport(handler))


def test_resposta_traz_citacao_com_documento_clausula_e_pagina():
    texto = format_answer(RESPOSTA)
    assert "Até 31 de dezembro de 2026." in texto
    assert "contrato_237.pdf · CLÁUSULA SEXTA · p. 3" in texto
    assert "verificada literalmente" in texto


def test_citacao_nao_verificada_e_sinalizada():
    data = {**RESPOSTA, "citations": [{**RESPOSTA["citations"][0], "verified": False}]}
    assert "NÃO verificada" in format_answer(data)


def test_abstencao_nunca_parece_resposta():
    texto = format_answer(ABSTENCAO)
    assert "se absteve" in texto
    assert "abaixo do limiar" in texto
    assert "Não complete com suposições" in texto
    assert "Citações:" not in texto


def test_responder_envia_pergunta_e_filtro_de_documentos():
    capturado = {}

    def handler(request: httpx.Request) -> httpx.Response:
        capturado["path"] = request.url.path
        capturado["body"] = json.loads(request.content)
        return httpx.Response(200, json=RESPOSTA)

    with _client(handler) as c:
        out = responder_com_citacao(c, "  Qual a vigência?  ", ["contrato_237.pdf"])
    assert capturado["path"] == "/query"
    assert capturado["body"] == {"query": "Qual a vigência?", "documents": ["contrato_237.pdf"]}
    assert "contrato_237.pdf" in out


def test_pergunta_curta_demais_nao_chega_na_api():
    def handler(request):  # pragma: no cover - não deve ser chamado
        raise AssertionError("não deveria chamar a API")

    with _client(handler) as c, pytest.raises(CrivoError, match="ao menos"):
        responder_com_citacao(c, "ok")


def test_pergunta_longa_e_cortada_no_limite_da_api():
    capturado = {}

    def handler(request):
        capturado["body"] = json.loads(request.content)
        return httpx.Response(200, json=RESPOSTA)

    with _client(handler) as c:
        responder_com_citacao(c, "a" * 800)
    assert len(capturado["body"]["query"]) == 500


def test_api_fora_do_ar_da_mensagem_acionavel():
    def handler(request):
        raise httpx.ConnectError("recusou")

    with _client(handler) as c, pytest.raises(CrivoError, match="uvicorn rag.api.main:app"):
        responder_com_citacao(c, "Qual a vigência?")


def test_503_vira_mensagem_sobre_a_chave():
    with _client(lambda r: httpx.Response(503, json={"detail": "sem chave"})) as c:
        with pytest.raises(CrivoError, match="ANTHROPIC_API_KEY"):
            responder_com_citacao(c, "Qual a vigência?")


def test_outros_erros_http_mostram_o_detalhe():
    with _client(lambda r: httpx.Response(422, json={"detail": "query muito curta"})) as c:
        with pytest.raises(CrivoError, match="422.*query muito curta"):
            responder_com_citacao(c, "Qual a vigência?")


def test_listar_documentos():
    data = {
        "total_documents": 2,
        "total_chunks": 30,
        "documents": [
            {"filename": "a.pdf", "doc_type": "contrato", "source_url": "", "chunk_count": 10},
            {"filename": "b.pdf", "doc_type": "aditivo", "source_url": "", "chunk_count": 20},
        ],
    }
    with _client(lambda r: httpx.Response(200, json=data)) as c:
        out = listar_documentos(c)
    assert "2 documentos, 30 trechos" in out
    assert "- a.pdf (contrato, 10 trechos)" in out


def test_status_junta_health_e_config():
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "version": "0.1.0",
                    "api_key_configured": False,
                    "documents_indexed": 21,
                    "chunks_available": 326,
                },
            )
        return httpx.Response(
            200,
            json={
                "answer_model": "claude-haiku-4-5",
                "embedding_model": "x",
                "reranker_model": "y",
                "retrieve_top_k": 20,
                "rerank_top_n": 5,
                "min_rerank_score": 0.35,
                "citation_match_threshold": 0.9,
            },
        )

    with _client(handler) as c:
        out = status_do_crivo(c)
    assert "326" in out and "NÃO configurada" in out and "claude-haiku-4-5" in out


def test_url_da_api_vem_do_ambiente(monkeypatch):
    monkeypatch.setenv("CRIVO_API_URL", "http://exemplo:9000/")
    assert mcp_server.api_url() == "http://exemplo:9000"


def test_servidor_registra_as_tres_ferramentas():
    pytest.importorskip("mcp")  # só roda se o extra [mcp] estiver instalado
    import asyncio
    import inspect

    server = mcp_server.build_server(_client(lambda r: httpx.Response(200, json=RESPOSTA)))
    tools = server.list_tools()
    if inspect.isawaitable(tools):
        tools = asyncio.run(tools)
    tools = getattr(tools, "tools", tools)  # SDK 2.x devolve um objeto com .tools
    nomes = {t.name for t in tools}
    assert nomes == {"responder_com_citacao", "listar_documentos", "status_do_crivo"}
