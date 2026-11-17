"""Servidor MCP do Crivo.

Expõe a API do Crivo (a que já roda em :8000) como ferramentas MCP, para que um
agente, como o Claude Desktop, consulte os contratos e receba a resposta já com
citação verificada.

Decisão de projeto: este servidor é um cliente HTTP fino da API, e não importa
o pipeline. Assim ele não carrega BGE-M3 nem o reranker na memória, não duplica
regra nenhuma (a abstenção e a verificação literal de citação continuam
morando na API), e o que o agente vê é exatamente o contrato público `/query`.

Ferramentas:
  responder_com_citacao  pergunta -> resposta com documento, cláusula, página e trecho
  listar_documentos      quais contratos estão indexados
  status_do_crivo        a API está no ar? há chave? quantos trechos?

Uso (stdio, o padrão do MCP):
  python -m rag.mcp_server

Variável de ambiente: CRIVO_API_URL (padrão http://localhost:8000).
"""

from __future__ import annotations

import os
import sys

import httpx

DEFAULT_URL = "http://localhost:8000"
# A geração com reranking pode levar dezenas de segundos.
TIMEOUT_S = 120.0
MIN_QUERY_CHARS = 3  # o mesmo mínimo do QueryRequest da API


class CrivoError(Exception):
    """Falha esperada (API fora do ar, sem chave...), com mensagem para o agente."""


def api_url() -> str:
    return os.environ.get("CRIVO_API_URL", DEFAULT_URL).rstrip("/")


def make_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    return httpx.Client(base_url=api_url(), timeout=TIMEOUT_S, transport=transport)


def _request(client: httpx.Client, method: str, path: str, **kwargs) -> dict:
    try:
        resp = client.request(method, path, **kwargs)
    except httpx.ConnectError as exc:
        raise CrivoError(
            f"Não consegui falar com a API do Crivo em {api_url()}. "
            "Ela precisa estar rodando: uvicorn rag.api.main:app"
        ) from exc
    except httpx.TimeoutException as exc:
        raise CrivoError(
            f"A API do Crivo demorou mais de {TIMEOUT_S:.0f} s para responder. "
            "Tente de novo ou use uma pergunta mais específica."
        ) from exc
    if resp.status_code == 503:
        raise CrivoError(
            "A API do Crivo está no ar, mas sem ANTHROPIC_API_KEY configurada: "
            "a busca funciona, a geração de respostas não."
        )
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise CrivoError(f"A API do Crivo respondeu {resp.status_code}: {detail}")
    return resp.json()


# ---------------------------------------------------------------------------
# Lógica das ferramentas (funções puras sobre um httpx.Client: fáceis de testar)
# ---------------------------------------------------------------------------


def format_answer(data: dict) -> str:
    """Resposta do `/query` em texto para o agente, sem esconder a abstenção."""
    answer = (data.get("answer") or "").strip()
    if data.get("abstained"):
        reason = data.get("abstain_reason") or "motivo não informado"
        return (
            f"{answer}\n\n"
            f"O Crivo se absteve de responder. Motivo: {reason}.\n"
            "Isso quer dizer que a resposta não está, de forma verificável, nos "
            "documentos indexados. Não complete com suposições."
        )

    lines = [answer, "", f"Confiança: {data.get('confidence', 'n/d')}"]
    citations = data.get("citations") or []
    if citations:
        lines.append("Citações:")
        for c in citations:
            status = "verificada literalmente" if c.get("verified") else "NÃO verificada"
            clause = c.get("clause") or "cláusula não identificada"
            lines.append(
                f"[{c.get('source_id')}] {c.get('document')} · {clause} · p. {c.get('page')}"
            )
            lines.append(f"    «{c.get('quote', '').strip()}» ({status})")
    else:
        lines.append("Sem citações verificadas.")
    if data.get("retries"):
        lines.append(
            f"(A resposta precisou de {data['retries']} reprocessamento(s) "
            "para passar na verificação.)"
        )
    return "\n".join(lines)


def responder_com_citacao(
    client: httpx.Client, pergunta: str, documentos: list[str] | None = None
) -> str:
    pergunta = (pergunta or "").strip()
    if len(pergunta) < MIN_QUERY_CHARS:
        raise CrivoError(f"A pergunta precisa ter ao menos {MIN_QUERY_CHARS} caracteres.")
    payload: dict = {"query": pergunta[:500]}
    if documentos:
        payload["documents"] = documentos
    return format_answer(_request(client, "POST", "/query", json=payload))


def listar_documentos(client: httpx.Client) -> str:
    data = _request(client, "GET", "/documents")
    docs = data.get("documents") or []
    lines = [
        f"{data.get('total_documents', len(docs))} documentos, "
        f"{data.get('total_chunks', 0)} trechos indexados."
    ]
    for d in docs:
        tipo = d.get("doc_type") or "tipo n/d"
        lines.append(f"- {d.get('filename')} ({tipo}, {d.get('chunk_count', 0)} trechos)")
    return "\n".join(lines)


def status_do_crivo(client: httpx.Client) -> str:
    health = _request(client, "GET", "/health")
    config = _request(client, "GET", "/config")
    key = (
        "configurada"
        if health.get("api_key_configured")
        else "NÃO configurada (a geração não funciona)"
    )
    return (
        f"Crivo {health.get('version')} no ar.\n"
        f"Documentos: {health.get('documents_indexed')} · "
        f"trechos: {health.get('chunks_available')}\n"
        f"Chave da Claude API: {key}\n"
        f"Modelo de resposta: {config.get('answer_model')} · "
        f"limiar de relevância: {config.get('min_rerank_score')} · "
        f"limiar de citação: {config.get('citation_match_threshold')}"
    )


# ---------------------------------------------------------------------------
# Registro no MCP (import tardio: o módulo importa sem o SDK instalado, o que
# deixa os testes da lógica acima rodarem no CI sem a dependência opcional)
# ---------------------------------------------------------------------------


def build_server(client: httpx.Client | None = None):
    try:  # SDK 2.x renomeou FastMCP para MCPServer; a API do decorador é a mesma.
        from mcp.server.mcpserver import MCPServer as _Server
    except ImportError:  # SDK 1.x
        from mcp.server.fastmcp import FastMCP as _Server

    http = client or make_client()
    mcp = _Server("crivo")

    def _safe(fn, *args):
        try:
            return fn(http, *args)
        except CrivoError as exc:
            return f"Erro: {exc}"

    @mcp.tool(name="responder_com_citacao")
    def responder_com_citacao_tool(pergunta: str, documentos: list[str] | None = None) -> str:
        """Responde uma pergunta sobre contratos públicos indexados no Crivo, citando
        documento, cláusula, página e trecho literal. Se a resposta não estiver nos
        documentos, o Crivo se abstém em vez de inventar. `documentos` (opcional) restringe
        a busca a esses nomes de arquivo; use listar_documentos para ver os nomes."""
        return _safe(responder_com_citacao, pergunta, documentos)

    @mcp.tool(name="listar_documentos")
    def listar_documentos_tool() -> str:
        """Lista os contratos indexados no Crivo, com tipo e quantidade de trechos."""
        return _safe(listar_documentos)

    @mcp.tool(name="status_do_crivo")
    def status_do_crivo_tool() -> str:
        """Diz se a API do Crivo está no ar, se há chave da Claude API e quantos trechos existem."""
        return _safe(status_do_crivo)

    return mcp


def main() -> None:
    try:
        server = build_server()
    except ImportError:
        sys.stderr.write('O SDK do MCP não está instalado. Rode: pip install -e ".[mcp]"\n')
        raise SystemExit(1) from None
    server.run()  # stdio


if __name__ == "__main__":
    main()
