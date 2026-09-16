"""Fase 4 — o contrato de saída do modelo, como schema de tool use.

Por que tool use e não "peça JSON no prompt e faça `json.loads`": um prompt
pedindo JSON ainda pode voltar com texto antes/depois, aspas erradas ou um
campo faltando — e aí o parser quebra silenciosamente ou você escreve um
regex frágil pra extrair o bloco. Tool use faz a Claude API validar a forma
da resposta contra um JSON Schema antes de ela chegar aqui: o campo
"citations" É uma lista de objetos com os campos certos, não "geralmente é".

Isto não elimina a alucinação de CONTEÚDO (o modelo ainda pode citar uma
fonte errada, ou um "quote" que não existe literalmente no texto) — só
elimina a alucinação de FORMATO. A verificação de conteúdo é o trabalho de
`verify.py`, depois que a resposta já chegou estruturada.
"""

from __future__ import annotations

ANSWER_TOOL_NAME = "emitir_resposta"

ANSWER_TOOL = {
    "name": ANSWER_TOOL_NAME,
    "description": (
        "Emite a resposta estruturada à pergunta de due diligence contratual, "
        "com citações rastreáveis às fontes numeradas fornecidas no prompt."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": (
                    "Resposta em linguagem natural, baseada só nas fontes "
                    "numeradas. Se insufficient_context for true, explica o "
                    "que falta em vez de responder a pergunta."
                ),
            },
            "citations": {
                "type": "array",
                "description": (
                    "Uma entrada por afirmação factual da resposta. Vazio "
                    "quando insufficient_context é true."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "source_id": {
                            "type": "integer",
                            "description": "O número [N] da fonte citada.",
                        },
                        "document": {
                            "type": "string",
                            "description": "Nome do arquivo do contrato, copiado da fonte.",
                        },
                        "clause": {
                            "type": "string",
                            "description": "clause_path da fonte, copiado exatamente.",
                        },
                        "page": {
                            "type": "integer",
                            "description": "Página inicial da fonte citada.",
                        },
                        "quote": {
                            "type": "string",
                            "description": (
                                "Trecho LITERAL copiado da fonte — nunca "
                                "parafraseado. É contra este campo que o "
                                "sistema verifica a citação."
                            ),
                        },
                    },
                    "required": ["source_id", "document", "clause", "page", "quote"],
                },
            },
            "confidence": {
                "type": "string",
                "enum": ["alta", "media", "baixa"],
                "description": "Confiança do modelo na resposta, dado o que as fontes cobrem.",
            },
            "insufficient_context": {
                "type": "boolean",
                "description": (
                    "true quando as fontes numeradas não bastam para "
                    "responder com confiança — nesse caso, citations deve "
                    "vir vazio e answer explica o que falta."
                ),
            },
        },
        "required": ["answer", "citations", "confidence", "insufficient_context"],
    },
}
