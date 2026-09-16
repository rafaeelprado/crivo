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

from collections.abc import Callable
from functools import lru_cache
from typing import Any

from rag.generation.schema import ANSWER_TOOL, ANSWER_TOOL_NAME
from rag.settings import settings

Message = dict[str, Any]
# (system_prompt, histórico_de_mensagens) -> input já validado da tool_use.
ModelCaller = Callable[[str, list[Message]], dict]

MAX_TOKENS = 1536


@lru_cache(maxsize=1)
def get_default_model_caller() -> ModelCaller:
    """Cria o caller real, contra a Claude API. Import tardio: importar este
    módulo não deveria exigir `anthropic` instalado nem uma chave configurada
    — só chamar a função devolvida é que exige.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def call(system: str, messages: list[Message]) -> dict:
        response = client.messages.create(
            model=settings.answer_model,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=[ANSWER_TOOL],
            tool_choice={"type": "tool", "name": ANSWER_TOOL_NAME},
            messages=messages,
        )
        for block in response.content:
            if block.type == "tool_use":
                return block.input
        # tool_choice força a ferramenta — chegar aqui indica resposta
        # malformada da API, não uma escolha do modelo.
        raise RuntimeError(
            f"Resposta da Claude API não trouxe um bloco tool_use para {ANSWER_TOOL_NAME}"
        )

    return call
