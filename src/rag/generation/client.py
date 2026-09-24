"""Fase 4 — a única função que fala com a Claude API.

Mesmo padrão de injeção de dependência da Fase 3 (`Embedder`, `Scorer`): o
resto da Fase 4 (`answer.py`) nunca importa `anthropic` nem sabe que existe
uma chave de API — ele recebe um `ModelCaller` (uma função) e chama. Em
produção, `get_default_model_caller()` cria essa função de verdade, com
`tool_choice` forçando o schema de `schema.py`. Em teste, um `ModelCaller`
falso e determinístico substitui a API real — sem gastar tokens, sem rede,
sem precisar de `ANTHROPIC_API_KEY` para rodar `pytest`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from rag.generation.schema import ANSWER_TOOL, ANSWER_TOOL_NAME
from rag.settings import settings

logger = logging.getLogger("rag.generation.client")

Message = dict[str, Any]
# (system_prompt, histórico_de_mensagens) -> input já validado da tool_use.
ModelCaller = Callable[[str, list[Message]], dict]

MAX_TOKENS = 1536
API_TIMEOUT = 60.0  # segundos — reranking + geração pode levar tempo
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # segundos


@lru_cache(maxsize=1)
def get_default_model_caller() -> ModelCaller:
    """Cria o caller real, contra a Claude API. Import tardio: importar este
    módulo não deveria exigir `anthropic` instalado nem uma chave configurada
    — só chamar a função devolvida é que exige.

    Com retry logic e timeout explícito — chamadas à Claude podem levar
    tempo (especialmente com reranking antes), e falhas de rede precisam
    de retry com backoff.
    """
    import anthropic

    client = anthropic.Anthropic(
        api_key=settings.anthropic_api_key,
        timeout=API_TIMEOUT,
    )

    def call(system: str, messages: list[Message]) -> dict:
        """Chama a Claude API com retry exponencial em falhas de rede."""
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.debug(
                    "Claude API: chamada %d/%d — modelo %s, %d tokens max",
                    attempt,
                    MAX_RETRIES,
                    settings.answer_model,
                    MAX_TOKENS,
                )
                start_time = time.time()

                response = client.messages.create(
                    model=settings.answer_model,
                    max_tokens=MAX_TOKENS,
                    system=system,
                    tools=[ANSWER_TOOL],
                    tool_choice={"type": "tool", "name": ANSWER_TOOL_NAME},
                    messages=messages,
                )

                elapsed = time.time() - start_time
                logger.debug("Claude API respondeu em %.1f segundos", elapsed)

                for block in response.content:
                    if block.type == "tool_use":
                        return block.input

                # tool_choice força a ferramenta — chegar aqui indica resposta
                # malformada da API, não uma escolha do modelo.
                raise RuntimeError(
                    f"Resposta da Claude API não trouxe um bloco tool_use para {ANSWER_TOOL_NAME}"
                )

            except Exception as e:
                last_error = e

                # Se foi a última tentativa ou erro irrecuperável, lança.
                if attempt == MAX_RETRIES:
                    logger.error(
                        "Claude API falhou após %d tentativas: %s",
                        MAX_RETRIES,
                        str(e),
                    )
                    raise

                # Senão, aguarda com backoff e tenta de novo.
                wait_time = RETRY_BACKOFF ** (attempt - 1)
                logger.warning(
                    "Claude API falhou na tentativa %d/%d (%s). "
                    "Aguardando %.1f segundos antes de retry...",
                    attempt,
                    MAX_RETRIES,
                    type(e).__name__,
                    wait_time,
                )
                time.sleep(wait_time)

        # Nunca chega aqui, mas mypy quer uma saída.
        raise last_error or RuntimeError("Erro desconhecido na Claude API")

    return call
