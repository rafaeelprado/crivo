"""Interface do Crivo — Fase 6.

Dois painéis: conversa (esquerda) e fontes (direita). Uma citação na resposta é
um botão — clicar nele abre, no painel de fontes, o texto inteiro do chunk por
trás daquela citação (documento, página, e o trecho citado destacado dentro do
texto completo), não só o `quote` curto que o modelo alegou ter usado. É a
leitura literal do critério de aceite da Fase 6: "clicar numa citação exibe o
texto-fonte com documento e página".

A pergunta é enviada via `POST /query/stream` (Fase 5) em vez de `POST /query`:
consumir o SSE do lado do cliente permite mostrar o progresso real do pipeline
("buscando fontes" -> "gerando resposta" -> "verificando citações") em vez de
travar a tela por alguns segundos atrás de um spinner genérico.

Cor e tipografia vêm de `docs/IDENTIDADE.md` — via `.streamlit/config.toml`
(paleta base) e o CSS logo abaixo (tipografia e componentes). As três cores
semânticas (verde-verificado, âmbar-rejeitada, tijolo-abstenção) são as únicas
cores vivas da tela, e cada uma só aparece quando aquele estado é verdade.
"""

from __future__ import annotations

import html
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 120.0

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

# Cores semânticas — espelham docs/IDENTIDADE.md, seção 2.
VERDE_VERIFICADO = "#2C6E49"
AMBAR_REJEITADA = "#8F5209"
TIJOLO_ABSTENCAO = "#8C2F1E"
CINZA_PEDRA = "#5A6472"  # não é semântico: é a ausência de estado (falha externa)
COR_POR_CONFIANCA = {
    "alta": VERDE_VERIFICADO,
    "media": AMBAR_REJEITADA,
    "média": AMBAR_REJEITADA,
    "baixa": TIJOLO_ABSTENCAO,
}

st.set_page_config(
    page_title="Crivo",
    page_icon=str(ASSETS / "favicon.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- identidade visual -----------------------------------------------------------


@st.cache_data
def _svg(nome: str) -> str:
    """Lê um SVG de assets/ para injeção inline.

    Inline, e não `st.image`: assim o SVG herda o tamanho do CSS e renderiza
    como vetor de verdade em qualquer versão do Streamlit.
    """
    return (ASSETS / nome).read_text(encoding="utf-8")


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600&display=swap');

/* Serifa em tudo que é título — o sinal tipográfico de "documento sério". */
h1, h2, h3, h4,
[data-testid="stHeading"] h1,
[data-testid="stHeading"] h2,
[data-testid="stHeading"] h3 {
    font-family: 'Lora', Georgia, 'Times New Roman', serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.1px;
    color: #14243D;
}

/* Mono em tudo que é medida: dígitos alinham em coluna e viram comparáveis. */
.dado {
    font-family: 'Source Code Pro', ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.78rem;
    color: #5A6472;
    letter-spacing: -0.1px;
}

.marca svg { height: 46px; width: auto; display: block; }

/* O padrão do Streamlit reserva espaço para a toolbar; com ela em modo
   mínimo, sobra um vazio grande acima da marca. */
.block-container { padding-top: 2.6rem; }

.tagline {
    color: #5A6472;
    font-size: 0.93rem;
    margin: 0.35rem 0 0 0;
}

/* Régua de estado — fica no topo, fora da sidebar, para latência e nº de
   chunks continuarem visíveis mesmo com a sidebar recolhida. */
.regua {
    border-top: 1px solid #DDD9D0;
    border-bottom: 1px solid #DDD9D0;
    padding: 0.42rem 0;
    margin: 0.9rem 0 1.3rem 0;
}

.badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--c);
    border: 1px solid var(--c);
    border-radius: 999px;
    padding: 0.08rem 0.55rem;
    background: color-mix(in srgb, var(--c) 8%, transparent);
    white-space: nowrap;
}

/* Texto-fonte: o chunk literal do contrato. Serifa aqui é intencional —
   é a única parte da tela que é o DOCUMENTO, não a interface. */
.fonte-texto {
    font-family: 'Lora', Georgia, serif;
    font-size: 0.92rem;
    line-height: 1.62;
    color: #22262B;
    /* Sem justificar: numa coluna estreita e sem hifenização, texto
       justificado abre "rios" de espaço em branco e piora a leitura. */
    text-align: left;
}

/* O trecho que o modelo citou e que passou na verificação literal.
   Marca-texto, como alguém faria num contrato impresso. */
.fonte-texto mark {
    background: rgba(156, 123, 77, 0.24);
    color: #22262B;
    padding: 0.02em 0.06em;
    border-radius: 2px;
}

/* Citações são referências, não ações: peso de link, não de botão. */
[data-testid="stExpander"] summary { font-size: 0.88rem; }

/* Abstenção: barra tijolo à esquerda, impossível de confundir com resposta. */
.aviso {
    border-left: 3px solid var(--c);
    background: color-mix(in srgb, var(--c) 6%, transparent);
    padding: 0.75rem 0.95rem;
    border-radius: 0 4px 4px 0;
}
.aviso-titulo {
    color: var(--c);
    font-weight: 600;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# --- cliente da API --------------------------------------------------------------


def _get(path: str) -> dict:
    return httpx.get(f"{API_URL}{path}", timeout=10).json()


def mensagem_de_falha(exc: Exception) -> str:
    """Traduz a exceção crua num motivo curto e acionável.

    Falha de infraestrutura não é abstenção — são coisas conceitualmente
    opostas ("não sei responder com as fontes que tenho" vs. "não consegui
    nem tentar"), e a interface não deveria deixar o usuário confundir as
    duas. As três primeiras traduções abaixo vieram de erros reais que
    apareceram colocando o sistema no ar pela primeira vez.
    """
    texto = str(exc)

    if "credit balance is too low" in texto:
        return (
            "Créditos da Claude API esgotados. Recarregue em console.anthropic.com "
            "→ Plans & Billing."
        )
    if "not scoped to a workspace" in texto:
        return (
            "A chave da Claude API não está vinculada a um workspace. Gere uma chave "
            "dentro de um workspace específico no Console e atualize o .env da API."
        )
    if "authentication_error" in texto or "invalid x-api-key" in texto:
        return "A chave da Claude API foi recusada. Confira a ANTHROPIC_API_KEY no .env da API."
    if "rate_limit" in texto:
        return "Limite de requisições da Claude API atingido. Tente de novo em alguns segundos."

    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response.status_code == 503:
            return (
                "A API respondeu 503: a geração está indisponível. Normalmente é a "
                "ANTHROPIC_API_KEY faltando no .env da API."
            )
        return f"A API respondeu {exc.response.status_code}."
    if isinstance(exc, httpx.ConnectError):
        return f"Não foi possível conectar em {API_URL}. A API está no ar?"
    if isinstance(exc, httpx.ReadTimeout):
        return "A API demorou demais para responder."

    return texto


def stream_query(query: str, documents: list[str] | None, on_stage) -> dict[str, Any]:
    """Consome `POST /query/stream` e devolve o payload do evento `result`.

    `on_stage` é chamado a cada evento `status` com o texto da etapa —
    exatamente o que o hook `on_stage` de `answer_question` (Fase 4) emite, só
    que consumido aqui do lado do cliente via SSE em vez de direto em processo.
    """
    payload: dict[str, Any] = {"query": query}
    if documents:
        payload["documents"] = documents

    event = None
    with httpx.stream(
        "POST", f"{API_URL}/query/stream", json=payload, timeout=REQUEST_TIMEOUT
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                # maxsplit=1 é essencial: o JSON do payload também tem ":"
                # dentro, e um split sem limite fatiaria o conteúdo ao meio.
                data = json.loads(line.split(":", 1)[1].strip())
                if event == "status":
                    on_stage(data["stage"])
                elif event == "result":
                    return data
                elif event == "error":
                    raise RuntimeError(data["message"])
    raise RuntimeError("o servidor encerrou o stream sem devolver um resultado")


# --- estado ----------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_source_id" not in st.session_state:
    st.session_state.selected_source_id = None
if "active_result" not in st.session_state:
    st.session_state.active_result = None
if "last_latency" not in st.session_state:
    st.session_state.last_latency = None


# --- sidebar: saúde do sistema + seletor de documentos ---------------------------

with st.sidebar:
    try:
        health = _get("/health")
        docs_response = _get("/documents")
    except Exception as exc:  # noqa: BLE001 — qualquer falha de rede vira a mesma mensagem
        st.error(f"API fora do ar em {API_URL}. Suba com `make api`.\n\n{exc}")
        st.stop()

    if not health["api_key_configured"]:
        st.warning("`ANTHROPIC_API_KEY` não configurada na API — a geração não vai funcionar.")

    st.subheader("Seletor de documentos")
    doc_names = sorted(d["filename"] for d in docs_response["documents"])
    selected_docs = st.multiselect(
        "Restringir a busca a",
        options=doc_names,
        default=[],
        placeholder="Todos os documentos",
        help=(
            "Vazio = busca no corpus inteiro. Com documentos selecionados, a busca "
            "híbrida (Fase 3) é refeita só entre eles — não é um filtro cosmético "
            "sobre um resultado já pronto."
        ),
    )

    with st.expander(f"Corpus — {len(doc_names)} documentos"):
        for doc in docs_response["documents"]:
            st.markdown(
                f'<div class="dado">{html.escape(doc["filename"])} · '
                f'{html.escape(doc["doc_type"] or "sem tipo")} · '
                f'{doc["chunk_count"]} chunks</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.caption("As quatro defesas contra alucinação")
    st.markdown(
        '<div class="dado">1. abstenção por limiar de relevância<br>'
        "2. contexto numerado e prompt fechado<br>"
        "3. verificação literal da citação<br>"
        "4. rastro de origem auditável</div>",
        unsafe_allow_html=True,
    )


# --- cabeçalho -------------------------------------------------------------------

st.markdown(f'<div class="marca">{_svg("logo.svg")}</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="tagline">Due diligence contratual com citações verificadas contra o '
    "texto de origem.</p>",
    unsafe_allow_html=True,
)

_latencia = (
    f"{st.session_state.last_latency:.1f} s" if st.session_state.last_latency else "—"
)
_chunks_usados = (
    len(st.session_state.active_result["sources"]) if st.session_state.active_result else "—"
)
st.markdown(
    f'<div class="regua"><span class="dado">'
    f'corpus: {health["documents_indexed"]} documentos · '
    f'{health["chunks_available"]} chunks indexados &nbsp;&nbsp;|&nbsp;&nbsp; '
    f"última consulta: {_latencia} · {_chunks_usados} chunks usados"
    f"</span></div>",
    unsafe_allow_html=True,
)


# --- layout principal: conversa | fontes -----------------------------------------

col_chat, col_sources = st.columns([3, 2], gap="large")

with col_chat:
    st.subheader("Conversa")

    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user", avatar=str(ASSETS / "avatar-usuario.png")):
                st.markdown(msg["content"])
            continue

        with st.chat_message("assistant", avatar=str(ASSETS / "favicon.png")):
            if msg["abstained"]:
                # Abstenção visualmente distinta de resposta fundamentada —
                # metade do critério de aceite da Fase 6. E falha de
                # infraestrutura distinta de abstenção: uma é o sistema
                # decidindo não responder, a outra é ele não ter conseguido
                # nem tentar. Cores e rótulos diferentes, de propósito.
                if msg.get("falha"):
                    cor, rotulo, prefixo = CINZA_PEDRA, "Falha de conexão", "detalhe"
                else:
                    cor, rotulo, prefixo = TIJOLO_ABSTENCAO, "Abstenção", "motivo"
                st.markdown(
                    f'<div class="aviso" style="--c:{cor}">'
                    f'<div class="aviso-titulo">{rotulo}</div>'
                    f'{html.escape(msg["content"])}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="dado" style="margin-top:0.5rem">'
                    f'{prefixo}: {html.escape(msg["abstain_reason"])}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(msg["content"])
                cor = COR_POR_CONFIANCA.get(msg["confidence"], AMBAR_REJEITADA)
                st.markdown(
                    f'<span class="badge" style="--c:{cor}">'
                    f'confiança {html.escape(msg["confidence"])}</span>',
                    unsafe_allow_html=True,
                )

            if msg["citations"]:
                st.caption("Fontes citadas — clique para abrir o texto completo")
                # Mais colunas que citações: os botões ficam estreitos, com
                # peso de referência, não de ação principal.
                cols = st.columns(max(len(msg["citations"]), 6))
                for col, citation in zip(cols, msg["citations"], strict=False):
                    marca = "✓" if citation["verified"] else "✕"
                    ajuda = (
                        f"{citation['document']} — {citation['clause']}, "
                        f"pág. {citation['page']} · "
                        f"casamento literal {citation['match_score']:.2f}"
                    )
                    if col.button(
                        f"[{citation['source_id']}] {marca}",
                        key=f"cit-{i}-{citation['source_id']}",
                        help=ajuda,
                        use_container_width=True,
                    ):
                        st.session_state.selected_source_id = citation["source_id"]
                        st.session_state.active_result = msg["result"]
                        st.rerun()

            if msg["latency"] is not None:
                extra = f" · {msg['retries']} reprocessamento" if msg.get("retries") else ""
                st.markdown(
                    f'<div class="dado" style="margin-top:0.55rem">'
                    f'{msg["latency"]:.1f} s · {len(msg["sources"])} chunks na geração'
                    f"{extra}</div>",
                    unsafe_allow_html=True,
                )

    prompt = st.chat_input("Pergunte sobre os contratos...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        try:
            with st.status("Buscando fontes...", expanded=False) as status:

                def on_stage(stage: str, _status=status) -> None:
                    _status.update(label=stage.capitalize())

                result = stream_query(prompt, selected_docs, on_stage)
                status.update(label="Concluído", state="complete")
        except Exception as exc:  # noqa: BLE001 — infraestrutura, não conteúdo do modelo
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "abstained": True,
                    "falha": True,
                    "content": "Não foi possível consultar a API.",
                    "abstain_reason": mensagem_de_falha(exc),
                    "confidence": "baixa",
                    "citations": [],
                    "sources": [],
                    "result": None,
                    "latency": None,
                    "retries": 0,
                }
            )
        else:
            latency = time.perf_counter() - start
            st.session_state.last_latency = latency
            st.session_state.active_result = result
            st.session_state.selected_source_id = (
                result["citations"][0]["source_id"] if result["citations"] else None
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "abstained": result["abstained"],
                    "content": result["answer"],
                    "abstain_reason": result["abstain_reason"],
                    "confidence": result["confidence"],
                    "citations": result["citations"],
                    "sources": result["sources"],
                    "result": result,
                    "latency": latency,
                    "retries": result["retries"],
                }
            )
        st.rerun()

with col_sources:
    st.subheader("Fontes")
    active_result = st.session_state.active_result

    if not active_result or not active_result["sources"]:
        st.caption(
            "O texto-fonte de cada citação aparece aqui — documento, página e o "
            "trecho citado destacado dentro do chunk inteiro."
        )
    else:
        quote_by_source_id = {c["source_id"]: c["quote"] for c in active_result["citations"]}
        verificada = {c["source_id"]: c["verified"] for c in active_result["citations"]}

        for source in active_result["sources"]:
            sid = source["source_id"]
            pages = (
                str(source["page_start"])
                if source["page_start"] == source["page_end"]
                else f"{source['page_start']}–{source['page_end']}"
            )
            clause = source["clause"] or "sem cláusula identificada"
            citada = sid in quote_by_source_id
            header = f"{'✓ ' if citada and verificada.get(sid) else ''}[{sid}] {clause}"

            with st.expander(header, expanded=(sid == st.session_state.selected_source_id)):
                st.markdown(
                    f'<div class="dado">{html.escape(source["document"])} · '
                    f"página {pages} · rerank {source['rerank_score']:.3f}</div>",
                    unsafe_allow_html=True,
                )

                # Escapa ANTES de destacar: texto de contrato extraído de PDF
                # pode conter "<", ">" ou "&", e o painel renderiza HTML.
                texto = html.escape(source["text"])
                quote = quote_by_source_id.get(sid)
                if quote:
                    quote_escapado = html.escape(quote)
                    if quote_escapado in texto:
                        texto = texto.replace(
                            quote_escapado, f"<mark>{quote_escapado}</mark>", 1
                        )
                st.markdown(f'<div class="fonte-texto">{texto}</div>', unsafe_allow_html=True)
