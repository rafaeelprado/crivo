"""Fase 4 — construção do prompt: contexto numerado e instrução fechada.

Duas das quatro defesas contra alucinação vivem aqui, antes de qualquer
chamada à API:

1. **Contexto numerado.** Cada chunk recuperado vira uma fonte com um número
   `[N]`, documento, cláusula e página. O modelo cita pelo número — é o que
   torna a citação um dado estruturado (Fase 4.3) em vez de uma alegação solta
   no meio da resposta.
2. **Prompt fechado.** A instrução deixa claro que só as fontes numeradas
   existem — o modelo não tem "conhecimento geral sobre contratos públicos"
   disponível para preencher lacuna. Se as fontes não bastam, a instrução é
   dizer isso (`insufficient_context`), não adivinhar.

Este módulo só monta texto — não chama a API e não decide nada. Isso separa
"o que perguntamos" (aqui) de "como decidimos se a resposta é confiável"
(`verify.py`) e "como orquestramos a chamada" (`answer.py`).
"""

from __future__ import annotations

from rag.retrieval.rerank import RankedChunk

SYSTEM_PROMPT = """\
Você é um assistente de due diligence contratual. Sua única fonte de \
verdade são os trechos de contrato numerados que aparecem abaixo, entre \
[1] e [N]. Você não tem acesso a nenhum outro contrato, lei ou conhecimento \
sobre este caso além do que está nessas fontes.

Regras, sem exceção:

1. Responda SOMENTE com base no que está nas fontes numeradas. Se a resposta \
não está lá, não complete com conhecimento geral sobre contratos \
administrativos brasileiros — mesmo que pareça óbvio ou provável.
2. Toda afirmação factual da resposta precisa vir acompanhada de uma \
citação apontando pra fonte numerada correspondente.
3. O campo "quote" de cada citação precisa ser uma cópia LITERAL de um \
trecho da fonte citada — copie e cole o texto, nunca parafraseie ou resuma \
dentro do "quote". Paráfrase vai no campo "answer", nunca no "quote".
4. Se as fontes não contêm informação suficiente para responder a pergunta \
com confiança, defina "insufficient_context" como true, deixe "citations" \
vazio e explique em "answer" o que falta — não tente adivinhar ou \
"ser útil" inventando uma resposta plausível.
5. Se só parte da pergunta tem resposta nas fontes, responda só essa parte \
e marque "confidence" como "baixa" ou "media", nunca "alta".

Use a ferramenta fornecida para estruturar sua resposta. Não responda em \
texto livre.\
"""

RETRY_INSTRUCTION_TEMPLATE = """\
Sua resposta anterior tinha citações que NÃO batem literalmente com o texto \
da fonte indicada. Isto quebra uma regra obrigatória do sistema: o campo \
"quote" tem que ser um trecho LITERAL da fonte, não uma paráfrase.

Citações rejeitadas:
{rejected}

Tente de novo. Para cada citação, copie o trecho exatamente como aparece na \
fonte numerada — sem resumir, sem ajustar pontuação, sem parafrasear. Se, \
revisando, você perceber que a fonte não sustenta aquela afirmação, remova \
a citação e ajuste a resposta (ou marque insufficient_context) em vez de \
tentar forçar uma citação que não existe literalmente no texto.\
"""


def format_source(index: int, chunk: RankedChunk) -> str:
    """Uma fonte numerada, no formato que o modelo vê e que verify.py espera
    poder rastrear de volta (pelo índice) até o RankedChunk original.
    """
    c = chunk.chunk
    localizacao = c.clause_path or "(sem cabeçalho identificado)"
    pagina = f"p.{c.page_start}" if c.page_start == c.page_end else f"p.{c.page_start}-{c.page_end}"
    return (
        f"[{index}] Documento: {c.document} | Cláusula: {localizacao} | {pagina}\n"
        f"{c.text}"
    )


def build_context(chunks: list[RankedChunk]) -> str:
    """Concatena as fontes numeradas — a ordem da lista define o número."""
    return "\n\n".join(format_source(i, c) for i, c in enumerate(chunks, start=1))


def build_user_prompt(query: str, chunks: list[RankedChunk]) -> str:
    context = build_context(chunks)
    return (
        f"Fontes disponíveis:\n\n{context}\n\n"
        f"---\n\nPergunta: {query}"
    )


def build_retry_prompt(rejected: list[dict]) -> str:
    """Mensagem de correção enviada na 2ª tentativa (defesa de reprocessamento).

    `rejected` é a lista de citações que falharam na verificação fuzzy —
    cada item já vem formatado por verify.py com o motivo da rejeição.
    """
    linhas = "\n".join(
        f"- Fonte [{r['source_id']}]: quote enviado = {r['quote']!r} "
        f"(similaridade {r['match_score']:.0%}, mínimo exigido "
        f"{r['threshold']:.0%})"
        for r in rejected
    )
    return RETRY_INSTRUCTION_TEMPLATE.format(rejected=linhas)
