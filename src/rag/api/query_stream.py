"""Fase 5 — `POST /query/stream`: a mesma pergunta da Fase 4, como SSE.

**O que "streaming" significa aqui, com todas as letras:** eventos de
PROGRESSO enquanto o pipeline roda ("buscando fontes" -> "gerando resposta"
-> "verificando citações"), não token a token da resposta do modelo. Isso
não é uma limitação escondida — é uma consequência direta de uma decisão de
design da Fase 4: a saída é forçada via `tool_choice`, e a Claude API não
expõe o conteúdo de um bloco `tool_use` incrementalmente de um jeito útil de
consumir antes do bloco fechar (ele só faz sentido completo, como JSON
validado). Um endpoint que fingisse streaming de token aqui estaria
mentindo sobre o que a arquitetura permite.

Ainda assim, streaming de PROGRESSO tem valor real numa UI: sem ele, o
cliente vê uma tela travada por alguns segundos (busca + reranking +
chamada à API, às vezes com reprocessamento) sem nenhum feedback. Com SSE,
a interface pode mostrar "buscando fontes...", depois "gerando
resposta...", em vez de um spinner genérico.

**Mecanismo:** `answer_question` roda numa thread separada (é uma função
síncrona e bloqueante — embeddings, reranking e a chamada à Claude API não
são async); o hook `on_stage` que ela chama a cada etapa empurra eventos
numa fila; a corrotina deste gerador lê da fila e produz o SSE. `on_stage`
roda na thread de trabalho, então usa `loop.call_soon_threadsafe` para
cruzar de volta pro event loop com segurança — chamar `asyncio.Queue.put`
direto de outra thread não é seguro.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from collections.abc import AsyncIterator

from rag.api.query_service import to_query_response
from rag.generation.answer import answer_question


def _format_sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def stream_query(
    query: str,
    *,
    top_k: int | None = None,
    rerank_top_n: int | None = None,
    document_filter: list[str] | None = None,
) -> AsyncIterator[str]:
    loop = asyncio.get_event_loop()
    events: queue.Queue = queue.Queue()

    def on_stage(stage: str) -> None:
        loop.call_soon_threadsafe(events.put_nowait, ("status", {"stage": stage}))

    def worker() -> None:
        try:
            result = answer_question(
                query,
                top_k=top_k,
                rerank_top_n=rerank_top_n,
                document_filter=document_filter,
                on_stage=on_stage,
            )
            response = to_query_response(result)
            loop.call_soon_threadsafe(events.put_nowait, ("result", response.model_dump()))
        except Exception as exc:  # noqa: BLE001 — reporta qualquer erro como evento SSE
            loop.call_soon_threadsafe(events.put_nowait, ("error", {"message": str(exc)}))
        finally:
            loop.call_soon_threadsafe(events.put_nowait, None)

    threading.Thread(target=worker, daemon=True).start()

    while True:
        item = await loop.run_in_executor(None, events.get)
        if item is None:
            break
        event, payload = item
        yield _format_sse(event, payload)
