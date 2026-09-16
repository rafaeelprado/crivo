# Fase 4 — critério de aceite: 10 perguntas fora do corpus

Gerado por `scripts/test_abstention.py` em 2026-09-16 14:23 UTC.

## Análise

**10/10 abstenções — critério de aceite atingido.** Todos os scores de reranking ficaram entre 0,000 e 0,004, muito abaixo do `MIN_RERANK_SCORE=0,35` — sem ambiguidade nenhuma, ao contrário da consulta de controle "art. 78" da Fase 3 (que chegava a 0,18).

**Achado importante: nenhuma das 10 chegou a chamar a Claude API.** Toda abstenção aqui veio da **defesa 1** (o corte por score de reranking, antes de qualquer geração) — não das defesas 2–4 (prompt fechado, verificação de citação, reprocessamento). Isso prova que a camada de recuperação (Fases 2 e 3) já filtra sozinha o caso mais óbvio de "pergunta sem nenhuma relação com o corpus", com custo zero de tokens de geração.

O que este teste NÃO prova, e por isso não é o fim da história: como o sistema se comporta quando a busca encontra *algum* chunk plausível (score moderado) mas o conteúdo não sustenta uma resposta completa — o caso em que as defesas 2–4 (prompt fechado, verificação literal da citação, reprocessamento) precisam entrar em ação de verdade. Isso exige perguntas relacionadas ao domínio do corpus mas sem resposta *naquele* corpus específico (ex.: "qual a multa por atraso num contrato de fornecimento de merenda escolar" quando nenhum contrato do corpus é sobre merenda escolar) — o tipo de pergunta "quase certa" que o golden set de 50 perguntas da Fase 7 vai incluir de propósito (10 das 50 são projetadas para não ter resposta no corpus, mas de um jeito mais sutil que estas 10).

**Conclusão prática:** a Fase 4 fecha o critério de aceite formal (10/10, zero respostas inventadas), mas a validação completa das defesas 2–4 fica para a Fase 7, com perguntas desenhadas para chegar mais perto da fronteira em vez de estarem obviamente fora do domínio.

| # | Pergunta | Absteve? | Motivo |
|---|---|---|---|
| 1 | qual é a capital da Mongólia | ✅ sim | melhor resultado da busca (score=0.000) fica abaixo do limiar de confiança MIN_RERANK_SCORE=0.350 |
| 2 | quantos átomos de hidrogênio tem uma molécula de glicose | ✅ sim | melhor resultado da busca (score=0.004) fica abaixo do limiar de confiança MIN_RERANK_SCORE=0.350 |
| 3 | quem foi campeão da Copa do Mundo de 2022 | ✅ sim | melhor resultado da busca (score=0.002) fica abaixo do limiar de confiança MIN_RERANK_SCORE=0.350 |
| 4 | qual é a cotação do bitcoin hoje | ✅ sim | melhor resultado da busca (score=0.001) fica abaixo do limiar de confiança MIN_RERANK_SCORE=0.350 |
| 5 | como faço um bolo de cenoura com cobertura de chocolate | ✅ sim | melhor resultado da busca (score=0.001) fica abaixo do limiar de confiança MIN_RERANK_SCORE=0.350 |
| 6 | explique a teoria da relatividade geral de Einstein | ✅ sim | melhor resultado da busca (score=0.002) fica abaixo do limiar de confiança MIN_RERANK_SCORE=0.350 |
| 7 | qual vai ser o clima em Aracaju amanhã | ✅ sim | melhor resultado da busca (score=0.000) fica abaixo do limiar de confiança MIN_RERANK_SCORE=0.350 |
| 8 | quantos habitantes tem o Japão atualmente | ✅ sim | melhor resultado da busca (score=0.000) fica abaixo do limiar de confiança MIN_RERANK_SCORE=0.350 |
| 9 | qual é a fórmula de Bhaskara | ✅ sim | melhor resultado da busca (score=0.001) fica abaixo do limiar de confiança MIN_RERANK_SCORE=0.350 |
| 10 | como se diz 'bom dia' em francês | ✅ sim | melhor resultado da busca (score=0.002) fica abaixo do limiar de confiança MIN_RERANK_SCORE=0.350 |

**Resultado: 10/10 abstenções.**
Critério de aceite da Fase 4 (PLANO.md): 10/10. Atingido.