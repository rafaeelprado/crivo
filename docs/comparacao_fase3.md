# Comparação — Fase 3: denso-puro vs. híbrido+rerank

Gerado por `scripts/compare_retrieval.py` em 2026-09-09 20:45 UTC (reindexado após a correção do bug do chunker).
Corpus: 326 chunks (334 antes da correção — a diferença são os 8 chunks fantasma que citações de lei quebradas geravam). Critério de aceite da Fase 2/3 no `PLANO.md`.

## Análise escrita (critério de aceite da Fase 3)

**Confirmação da correção do bug:** depois de reindexar com o `chunker.py` corrigido, `'art. 137 d'` e `'art. 94 d'` desapareceram por completo dos resultados — nas consultas 6 ("multa moratória") e 10 ("art. 78"), onde apareciam antes, agora só há `clause_path` reais. O texto dessas citações de lei passou a fazer parte do chunk da cláusula em que estava inserido, como deveria desde o início.

**Resultado agregado das 10 consultas (já com o corpus corrigido):**

| # | Consulta | A vs. B | Leitura |
|---|---|---|---|
| 1 | penalidades por atraso | B melhora | Mesmo conjunto de cláusulas; B reordena colocando a mais específica no topo. |
| 2 | prazo de vigência | Empate | Top-3 idêntico nos dois modos. |
| 3 | reajuste de preços | B melhora levemente | B troca um resultado genérico por outro mais direto ao tema (reajustamento de preços). |
| 4 | rescisão contratual | Empate, com reordenação | B só troca a ordem das mesmas três cláusulas de rescisão — sem o ruído de penalidades que a versão anterior deste experimento tinha mostrado. |
| 5 | garantia contratual | B muda o conjunto | B traz `PREÂMBULO` na 2ª posição (score alto) e uma cláusula de garantia de execução — plausível, mas vale conferir manualmente se o preâmbulo citado realmente fala de garantia. |
| 6 | multa moratória | **B corrigido** | Sem mais `'art. 137 d'`. A e B agora concordam nos 4 primeiros resultados; B traz uma 5ª cláusula (PREÂMBULO) com score 0,830, plausível. |
| 7 | obrigações da contratada | Empate | Praticamente o mesmo conjunto, reordenação mínima. |
| 8 | foro eleito | B melhora | Mesmo conjunto de cláusulas de foro, ordenado por especificidade. |
| 9 | sanções administrativas | B melhora | Top-4 idêntico; scores altos e consistentes (0,86–0,98) no top-5 inteiro — melhor que a rodada anterior, que tinha uma 5ª posição com score 0,014. |
| 10 | "art. 78" (controle, ambígua) | Ambos ruins, como esperado | **Sem mais `'art. 94 d'`.** Os scores de B continuam todos abaixo de 0,18 — a consulta é genuinamente ambígua no domínio (Lei 8.666/1993 vs. 14.133/2021), não um problema de dado. |

**Conclusão prática:** com o bug do chunker corrigido, híbrido+rerank empata ou melhora sobre denso-puro em 9 das 10 consultas (a 5ª posição da consulta de garantia é a única que merece checagem manual antes de confiar cegamente). Os dois `clause_path` fantasma que apareciam antes sumiram completamente da comparação, confirmando a correção em dado real, não só em teste sintético.

### O achado mais importante não é "quem ganhou", é o *score*

A pergunta que a Fase 3 precisa responder para a Fase 4 não é só "qual modo dá o melhor top-5", é **"como eu sei, programaticamente, quando não confiar em nenhum dos dois?"** — e a resposta está nos números acima:

- Nas consultas bem resolvidas (1, 2, 3, 6, 7, 8, 9), o top-3 do reranker fica quase sempre **acima de 0,8**.
- Na consulta ambígua de controle (10), todos os scores ficam **abaixo de 0,18** — bem abaixo de qualquer limiar razoável.

Esse é o dado real que vai calibrar `MIN_RERANK_SCORE` na Fase 4 (o limiar de abstenção): o valor inicial no `.env.example` (0,35) cai exatamente na fronteira entre esses dois grupos. A Fase 7 vai validar isso contra um golden set de verdade, não contra 10 amostras — mas a ordem de grandeza bate.

### O bug real encontrado nesta comparação (corrigido e confirmado)

Antes da correção, as consultas 6 e 10 traziam `clause_path` quebrados: `'art. 137 d'` e `'art. 94 d'`. Não eram cláusulas do contrato — eram citações de lei ("...conforme o disposto no art. 137 da Lei nº 14.133...") que quebraram de linha exatamente no número do artigo, por causa do reflow do PDF. O detector de cabeçalhos do chunker (Fase 2) não distinguia isso de um `Art. N` genuíno, porque olhava só a linha atual, nunca o que vinha antes.

**Corrigido em `src/rag/ingest/chunker.py`** (`find_headings`): o padrão `"artigo"` só é aceito como cabeçalho quando a linha anterior está em branco, termina em pontuação de fim de frase, ou é a primeira linha do documento — o que uma citação quebrada no meio da frase nunca satisfaz. Três testes de regressão cobrem isso em `tests/test_chunker.py`. **Confirmado em dado real:** reindexando o corpus inteiro (326 chunks, contra 334 antes) e rodando esta mesma comparação de novo, os dois `clause_path` fantasma desapareceram por completo.

---

## 1. "quais as penalidades por atraso na entrega"

**A — denso-puro (só embeddings):**

- `CLÁUSULA SÉTIMA DAS` — Contrato+119+Thiago+ass.pdf
  > CLÁUSULA SÉTIMA DAS SANÇÕES ADMINISTRATIVAS PARA O CASO DE INADIMPLEMENTO CONTRATUAL 7.1. As licitantes [...]
- `CLÁUSULA SÉTIMA - DAS` — 8594802._CONTRATO_N_01.2404-2026_-_INEX_011-2026_-_MOICANO.pdf
  > CLÁUSULA SÉTIMA - DAS PENALIDADES CONTRATUAIS 7.1. As penalidades a seguir tratam sobre as apresentações [...]
- `CLÁUSULA DÉCIMA TERCEIRA` — 42._CONTRATO_ADMINISTRATIVO_N.42_2026_-_Proc._518_2025_-_OBRA_PAVIMENTACAO_assinadoassass.pdf
  > CLÁUSULA DÉCIMA TERCEIRA – DAS PENALIDADES 15.1.​ Pelo atraso e inexecução total ou parcial deste [...]
- `CLÁUSULA DÉCIMA OITAVA` — CONTRATO26.2026BOMJESUSDOGURGUEIA.pdf
  > 18.11.2.4 Quanto à frequência, o atraso será classificado como: a) eventual: quando ocorrer apenas uma [...]
- `CLÁUSULA DÉCIMA SEGUNDA` — Ra73e6mdBi1OPsgasLk4kXMpDLd62nB2iWqxdDr5.pdf
  > CLÁUSULA DÉCIMA SEGUNDA – INFRAÇÕES E SANÇÕES ADMINISTRATIVAS Contrato 145555671 SEI [...]

**B — híbrido (RRF) + reranking:**

- `CLÁUSULA DÉCIMA TERCEIRA` — 342d3edfbee4f61e2b9c4c03e1732b5f2819d125.pdf (score=0.967)
  > CLÁUSULA DÉCIMA TERCEIRA - DAS PENALIDADES CABÍVEIS 13.1. O descumprimento total ou parcial das [...]
- `CLÁUSULA SÉTIMA DAS` — Contrato+119+Thiago+ass.pdf (score=0.932)
  > CLÁUSULA SÉTIMA DAS SANÇÕES ADMINISTRATIVAS PARA O CASO DE INADIMPLEMENTO CONTRATUAL 7.1. As licitantes [...]
- `CLÁUSULA SÉTIMA - DAS` — 8594802._CONTRATO_N_01.2404-2026_-_INEX_011-2026_-_MOICANO.pdf (score=0.853)
  > CLÁUSULA SÉTIMA - DAS PENALIDADES CONTRATUAIS 7.1. As penalidades a seguir tratam sobre as apresentações [...]
- `CLÁUSULA DÉCIMA OITAVA` — CONTRATO26.2026BOMJESUSDOGURGUEIA.pdf (score=0.705)
  > 18.11.2.4 Quanto à frequência, o atraso será classificado como: a) eventual: quando ocorrer apenas uma [...]
- `CLÁUSULA DÉCIMA TERCEIRA` — 42._CONTRATO_ADMINISTRATIVO_N.42_2026_-_Proc._518_2025_-_OBRA_PAVIMENTACAO_assinadoassass.pdf (score=0.670)
  > CLÁUSULA DÉCIMA TERCEIRA – DAS PENALIDADES 15.1.​ Pelo atraso e inexecução total ou parcial deste [...]

## 2. "qual o prazo de vigência do contrato"

**A — denso-puro (só embeddings):**

- `CLÁUSULA TERCEIRA` — 42._CONTRATO_ADMINISTRATIVO_N.42_2026_-_Proc._518_2025_-_OBRA_PAVIMENTACAO_assinadoassass.pdf
  > CLÁUSULA TERCEIRA – VIGÊNCIA E PRORROGAÇÃO. 5.1.​ O prazo de vigência do presente contrato é de 12 [...]
- `CLÁUSULA SEGUNDA` — Ra73e6mdBi1OPsgasLk4kXMpDLd62nB2iWqxdDr5.pdf
  > CLÁUSULA SEGUNDA – VIGÊNCIA E PRORROGAÇÃO 2.1. O prazo de vigência da contratação é de 30 (trinta) [...]
- `CLÁUSULA SEXTA - DA` — 342d3edfbee4f61e2b9c4c03e1732b5f2819d125.pdf
  > CLÁUSULA SEXTA - DA VIGÊNCIA 6.1. O prazo de vigência deste Contrato é até 31 de dezembro de 2026, [...]
- `CLÁUSULA SEGUNDA` — Contrato.pdf
  > CLÁUSULA SEGUNDA – DA VIGÊNCIA E PRORROGAÇÃO 2.1. O prazo de vigência da contratação é de 12 (doze) [...]
- `CLÁUSULA SEGUNDA` — Contrato nº 141-2026- Dispensa de Licitação nº 027-2026- Serviço de dedetização- MAX IMUNIZAÇÃO E SERVIÇOS LTDA.pdf
  > CLÁUSULA SEGUNDA – DA VIGÊNCIA 2.1. O prazo de vigência do contrato será até 31 de dezembro de 2026, [...]

**B — híbrido (RRF) + reranking:**

- `CLÁUSULA TERCEIRA` — 42._CONTRATO_ADMINISTRATIVO_N.42_2026_-_Proc._518_2025_-_OBRA_PAVIMENTACAO_assinadoassass.pdf (score=0.997)
  > CLÁUSULA TERCEIRA – VIGÊNCIA E PRORROGAÇÃO. 5.1.​ O prazo de vigência do presente contrato é de 12 [...]
- `CLÁUSULA SEGUNDA` — Contrato nº 141-2026- Dispensa de Licitação nº 027-2026- Serviço de dedetização- MAX IMUNIZAÇÃO E SERVIÇOS LTDA.pdf (score=0.996)
  > CLÁUSULA SEGUNDA – DA VIGÊNCIA 2.1. O prazo de vigência do contrato será até 31 de dezembro de 2026, [...]
- `CLÁUSULA SEXTA - DA` — 342d3edfbee4f61e2b9c4c03e1732b5f2819d125.pdf (score=0.994)
  > CLÁUSULA SEXTA - DA VIGÊNCIA 6.1. O prazo de vigência deste Contrato é até 31 de dezembro de 2026, [...]
- `CLÁUSULA SEGUNDA` — Ra73e6mdBi1OPsgasLk4kXMpDLd62nB2iWqxdDr5.pdf (score=0.993)
  > CLÁUSULA SEGUNDA – VIGÊNCIA E PRORROGAÇÃO 2.1. O prazo de vigência da contratação é de 30 (trinta) [...]
- `CLÁUSULA SEGUNDA` — _var_www_html_ararangua.atende.net_temp_WCO_PNCP_Contrato_CONTRATO_6_2026_IPM.pdf (score=0.970)
  > CLÁUSULA SEGUNDA – PRAZOS 2.1. A implantação deverá ser iniciada após a emissão da Ordem de Serviço ou [...]

## 3. "como funciona o reajuste de preços"

**A — denso-puro (só embeddings):**

- `CLÁUSULA OITAVA - REAJUSTE` — TEXTO+DE+CONTRATO.pdf
  > CLÁUSULA OITAVA - REAJUSTE (art. 92, V) Os preços inicialmente contratados são fixos e irreajustáveis no [...]
- `CLÁUSULA SÉTIMA - REAJUSTE` — 1769616745717-529927364-_-202509243__CONTRATO_EXT_E_PUBLICAO_E_CERTIDES.pdf
  > CLÁUSULA SÉTIMA - REAJUSTE 7.1 . O objeto do presente instrumento não será reajustado, salvo condições [...]
- `CLÁUSULA SÉTIMA - REAJUSTE` — Contrato.pdf
  > CLÁUSULA SÉTIMA - REAJUSTE 7.1. Os preços inicialmente contratados são fixos e irreajustáveis no prazo [...]
- `CLÁUSULA SÉTIMA - REAJUSTE` — 42._CONTRATO_ADMINISTRATIVO_N.42_2026_-_Proc._518_2025_-_OBRA_PAVIMENTACAO_assinadoassass.pdf
  > CLÁUSULA SÉTIMA - REAJUSTE 9.1.​ Os preços inicialmente contratados são fixos e irreajustáveis no prazo [...]
- `CLÁUSULA OITAVA - DOS` — 342d3edfbee4f61e2b9c4c03e1732b5f2819d125.pdf
  > CLÁUSULA OITAVA - DOS CRITÉRIOS, DATA BASE E PERIODICIDADE DO REAJUSTAMENTO DE PREÇOS E CRITÉRIOS DE [...]

**B — híbrido (RRF) + reranking:**

- `CLÁUSULA OITAVA - REAJUSTE` — TEXTO+DE+CONTRATO.pdf (score=0.987)
  > CLÁUSULA OITAVA - REAJUSTE (art. 92, V) Os preços inicialmente contratados são fixos e irreajustáveis no [...]
- `CLÁUSULA DÉCIMA - REAJUSTAMENTO` — CONTRATO26.2026BOMJESUSDOGURGUEIA.pdf (score=0.961)
  > CLÁUSULA DÉCIMA - REAJUSTAMENTO DE PREÇOS 10.1. Os preços contratuais propostos não serão reajustados. [...]
- `CLÁUSULA SÉTIMA - REAJUSTE` — 1769616745717-529927364-_-202509243__CONTRATO_EXT_E_PUBLICAO_E_CERTIDES.pdf (score=0.945)
  > CLÁUSULA SÉTIMA - REAJUSTE 7.1 . O objeto do presente instrumento não será reajustado, salvo condições [...]
- `CLÁUSULA SÉTIMA - REAJUSTE` — Contrato.pdf (score=0.935)
  > CLÁUSULA SÉTIMA - REAJUSTE 7.1. Os preços inicialmente contratados são fixos e irreajustáveis no prazo [...]
- `CLÁUSULA QUARTA` — Contrato+n+0302026+PREF+TRECHO+IMBUIA+2+100+.pdf (score=0.790)
  > CLÁUSULA QUARTA (DO REAJUSTE, REEQUILÍBRIO E REPACTUAÇÃO) 1 - Os preços praticados quanto ao valor da [...]

## 4. "quais as hipóteses de rescisão contratual"

**A — denso-puro (só embeddings):**

- `CLÁUSULA NONA - DA` — C_009499.PDF
  > CLÁUSULA NONA - DA RESCISÃO E/OU EXTINÇÃO CONTRATUAL 9.1. As hipóteses que constituem motivo para [...]
- `CLÁUSULA DÉCIMA NONA` — CONTRATO26.2026BOMJESUSDOGURGUEIA.pdf
  > CLÁUSULA DÉCIMA NONA - DA RESCISÃO 19.1. O presente Termo de Contrato poderá ser rescindido nas [...]
- `CLAUSULA DECIMA SEGUNDA` — Contrato+n+0302026+PREF+TRECHO+IMBUIA+2+100+.pdf
  > CLAUSULA DECIMA SEGUNDA (DA RESCISÃO CONTRATUAL) Página 9 de 15 1 - A rescisão do presente contrato [...]
- `CLÁUSULA DÉCIMA QUINTA` — 42._CONTRATO_ADMINISTRATIVO_N.42_2026_-_Proc._518_2025_-_OBRA_PAVIMENTACAO_assinadoassass.pdf
  > CLÁUSULA DÉCIMA QUINTA – DA RESCISÃO E DA EXTINÇÃO DO CONTRATO. 17.1.​ O contrato pode ser extinto antes [...]
- `CLÁUSULA NONA` — Contrato nº 141-2026- Dispensa de Licitação nº 027-2026- Serviço de dedetização- MAX IMUNIZAÇÃO E SERVIÇOS LTDA.pdf
  > CLÁUSULA NONA – DA INEXECUÇÃO E DA RESCISÃO DO CONTRATO 9.1. O contrato será extinto, assegurada o [...]

**B — híbrido (RRF) + reranking:**

- `CLÁUSULA DÉCIMA NONA` — CONTRATO26.2026BOMJESUSDOGURGUEIA.pdf (score=0.992)
  > CLÁUSULA DÉCIMA NONA - DA RESCISÃO 19.1. O presente Termo de Contrato poderá ser rescindido nas [...]
- `CLAUSULA DECIMA SEGUNDA` — Contrato+n+0302026+PREF+TRECHO+IMBUIA+2+100+.pdf (score=0.991)
  > CLAUSULA DECIMA SEGUNDA (DA RESCISÃO CONTRATUAL) Página 9 de 15 1 - A rescisão do presente contrato [...]
- `CLÁUSULA NONA - DA` — C_009499.PDF (score=0.989)
  > CLÁUSULA NONA - DA RESCISÃO E/OU EXTINÇÃO CONTRATUAL 9.1. As hipóteses que constituem motivo para [...]
- `CLÁUSULA DÉCIMA QUINTA` — 42._CONTRATO_ADMINISTRATIVO_N.42_2026_-_Proc._518_2025_-_OBRA_PAVIMENTACAO_assinadoassass.pdf (score=0.981)
  > CLÁUSULA DÉCIMA QUINTA – DA RESCISÃO E DA EXTINÇÃO DO CONTRATO. 17.1.​ O contrato pode ser extinto antes [...]
- `CLÁUSULA DÉCIMA TERCEIRA` — 342d3edfbee4f61e2b9c4c03e1732b5f2819d125.pdf (score=0.797)
  > CLÁUSULA DÉCIMA TERCEIRA - DAS PENALIDADES CABÍVEIS 13.1. O descumprimento total ou parcial das [...]

## 5. "qual o valor da garantia contratual exigida"

**A — denso-puro (só embeddings):**

- `CLÁUSULA TERCEIRA` — _var_www_html_ararangua.atende.net_temp_WCO_PNCP_Contrato_CONTRATO_6_2026_IPM.pdf
  > CLÁUSULA TERCEIRA – VALOR DO CONTRATO E RECURSOS ORÇAMENTÁRIOS 3.1. Como contrapartida à execução do [...]
- `CLÁUSULA DÉCIMA` — SEI14813041TermoAditivoProrrogaoedecrscimo.pdf
  > CLÁUSULA DÉCIMA – GARANTIA DE EXECUÇÃO 10.1. A contratação conta com garantia de execução, nos moldes do [...]
- `CLÁUSULA QUINTA` — SEI14813041TermoAditivoProrrogaoedecrscimo.pdf
  > CLÁUSULA QUINTA – PREÇO 5.1. O valor global do contrato é de R$ 4.077.076,00 (quatro milhões, setenta e [...]
- `CLÁUSULA TERCEIRA - DA` — C_009499.PDF
  > CLÁUSULA TERCEIRA - DA GARANTIA CONTRATUAL 3.1. A licitante vencedora prestará garantia ao Contrato em [...]
- `CLÁUSULA DÉCIMA SEGUNDA` — TEXTO+DE+CONTRATO.pdf
  > A garantia assegurará, qualquer que seja a modalidade escolhida, o pagamento de: prejuízos advindos do [...]

**B — híbrido (RRF) + reranking:**

- `CLÁUSULA TERCEIRA - DA` — C_009499.PDF (score=0.925)
  > CLÁUSULA TERCEIRA - DA GARANTIA CONTRATUAL 3.1. A licitante vencedora prestará garantia ao Contrato em [...]
- `PREÂMBULO` — CONTRATO+0722026+ass+cel+Ronildo-2545466046187100704.pdf (score=0.775)
  > 6 de 11 Brasil, e deverá constar expressa renúncia do fiador aos benefícios do artigo 827 do Código [...]
- `CLÁUSULA DÉCIMA` — SEI14813041TermoAditivoProrrogaoedecrscimo.pdf (score=0.715)
  > CLÁUSULA DÉCIMA – GARANTIA DE EXECUÇÃO 10.1. A contratação conta com garantia de execução, nos moldes do [...]
- `CLÁUSULA OITAVA` — Contrato nº 141-2026- Dispensa de Licitação nº 027-2026- Serviço de dedetização- MAX IMUNIZAÇÃO E SERVIÇOS LTDA.pdf (score=0.435)
  > DISPENSA DE LICITAÇÃO Nº 027/2026 Fls. 6/8 II. Caso o cumprimento da obrigação principal, uma vez [...]
- `CLÁUSULA TERCEIRA` — _var_www_html_ararangua.atende.net_temp_WCO_PNCP_Contrato_CONTRATO_6_2026_IPM.pdf (score=0.429)
  > CLÁUSULA TERCEIRA – VALOR DO CONTRATO E RECURSOS ORÇAMENTÁRIOS 3.1. Como contrapartida à execução do [...]

## 6. "multa moratória por descumprimento"

**A — denso-puro (só embeddings):**

- `CLÁUSULA DÉCIMA TERCEIRA` — 342d3edfbee4f61e2b9c4c03e1732b5f2819d125.pdf
  > CLÁUSULA DÉCIMA TERCEIRA - DAS PENALIDADES CABÍVEIS 13.1. O descumprimento total ou parcial das [...]
- `CLÁUSULA OITAVA` — Contrato nº 141-2026- Dispensa de Licitação nº 027-2026- Serviço de dedetização- MAX IMUNIZAÇÃO E SERVIÇOS LTDA.pdf
  > DISPENSA DE LICITAÇÃO Nº 027/2026 Fls. 6/8 II. Caso o cumprimento da obrigação principal, uma vez [...]
- `CLÁUSULA DÉCIMA TERCEIRA` — TEXTO+DE+CONTRATO.pdf
  > CLÁUSULA DÉCIMA TERCEIRA – INFRAÇÕES E SANÇÕES ADMINISTRATIVAS (art. 92, XIV) Comete infração [...]
- `CLÁUSULA DÉCIMA TERCEIRA` — 42._CONTRATO_ADMINISTRATIVO_N.42_2026_-_Proc._518_2025_-_OBRA_PAVIMENTACAO_assinadoassass.pdf
  > CLÁUSULA DÉCIMA TERCEIRA – DAS PENALIDADES 15.1.​ Pelo atraso e inexecução total ou parcial deste [...]
- `CLÁUSULA DÉCIMA SEGUNDA` — Contrato.pdf
  > CLÁUSULA DÉCIMA SEGUNDA- INFRAÇÕES E SANÇÕES ADMINISTRATIVAS 12.1. Comete infração administrativa, nos [...]

**B — híbrido (RRF) + reranking:**

- `CLÁUSULA DÉCIMA TERCEIRA` — 342d3edfbee4f61e2b9c4c03e1732b5f2819d125.pdf (score=0.962)
  > CLÁUSULA DÉCIMA TERCEIRA - DAS PENALIDADES CABÍVEIS 13.1. O descumprimento total ou parcial das [...]
- `CLÁUSULA OITAVA` — Contrato nº 141-2026- Dispensa de Licitação nº 027-2026- Serviço de dedetização- MAX IMUNIZAÇÃO E SERVIÇOS LTDA.pdf (score=0.937)
  > DISPENSA DE LICITAÇÃO Nº 027/2026 Fls. 6/8 II. Caso o cumprimento da obrigação principal, uma vez [...]
- `CLÁUSULA DÉCIMA TERCEIRA` — TEXTO+DE+CONTRATO.pdf (score=0.917)
  > CLÁUSULA DÉCIMA TERCEIRA – INFRAÇÕES E SANÇÕES ADMINISTRATIVAS (art. 92, XIV) Comete infração [...]
- `CLÁUSULA DÉCIMA SEGUNDA` — Contrato.pdf (score=0.885)
  > CLÁUSULA DÉCIMA SEGUNDA- INFRAÇÕES E SANÇÕES ADMINISTRATIVAS 12.1. Comete infração administrativa, nos [...]
- `PREÂMBULO` — CONTRATO+0722026+ass+cel+Ronildo-2545466046187100704.pdf (score=0.830)
  > 7 de 11 11.1.8. Apresentar declaração ou documentação falsa exigida para o certame ou prestar declaração [...]

## 7. "quais as obrigações da contratada"

**A — denso-puro (só embeddings):**

- `CLÁUSULA NONA - OBRIGAÇÕES` — Contrato.pdf
  > CLÁUSULA NONA - OBRIGAÇÕES DA CONTRATADA 9.1. A Contratada deve cumprir todas as obrigações constantes [...]
- `CLÁUSULA DÉCIMA QUINTA` — CONTRATO26.2026BOMJESUSDOGURGUEIA.pdf
  > CLÁUSULA DÉCIMA QUINTA- DA RESPONSABILIDADE DA CONTRATADA 15.1. A contratada obriga-se a adotar todas as [...]
- `CLÁUSULA QUINTA - DAS` — C_009499.PDF
  > CLÁUSULA QUINTA - DAS OBRIGAÇÕES DA CONTRATADA 5.1. Assume a Contratada inteira e expressa [...]
- `CLÁUSULA SEXTA - DAS` — 8594802._CONTRATO_N_01.2404-2026_-_INEX_011-2026_-_MOICANO.pdf
  > CLÁUSULA SEXTA - DAS OBRIGAÇÕES Compete à Contratada: 6.1. Não transferir os direitos e obrigações: A [...]
- `CLÁUSULA OITAVA` — Contrato+n+0302026+PREF+TRECHO+IMBUIA+2+100+.pdf
  > CLÁUSULA OITAVA – (DAS RESPONSABILIDADES E OBRIGAÇÕES DAS PARTES) 1 - OBRIGAÇÕES DA CONTRATADA 1.1 - A [...]

**B — híbrido (RRF) + reranking:**

- `CLÁUSULA NONA - OBRIGAÇÕES` — Contrato.pdf (score=0.998)
  > CLÁUSULA NONA - OBRIGAÇÕES DA CONTRATADA 9.1. A Contratada deve cumprir todas as obrigações constantes [...]
- `CLÁUSULA QUINTA - DAS` — C_009499.PDF (score=0.998)
  > CLÁUSULA QUINTA - DAS OBRIGAÇÕES DA CONTRATADA 5.1. Assume a Contratada inteira e expressa [...]
- `CLÁUSULA DÉCIMA QUINTA` — CONTRATO26.2026BOMJESUSDOGURGUEIA.pdf (score=0.998)
  > CLÁUSULA DÉCIMA QUINTA- DA RESPONSABILIDADE DA CONTRATADA 15.1. A contratada obriga-se a adotar todas as [...]
- `CLÁUSULA SÉTIMA - DAS` — CONTRATO26.2026BOMJESUSDOGURGUEIA.pdf (score=0.992)
  > CLÁUSULA SÉTIMA - DAS OBRIGAÇÕES DA CONTRATANTE 7.1. A CONTRATANTE se obriga a: a) Aprovar as medições [...]
- `Cláusula Primeira` — CONTRATO26.2026BOMJESUSDOGURGUEIA.pdf (score=0.989)
  > Cláusula Primeira, bem como as mesmas condições de habilitação; d) Administrar com zelo e probidade a [...]

## 8. "qual o foro eleito para dirimir litígios"

**A — denso-puro (só embeddings):**

- `CLÁUSULA DÉCIMA SEXTA` — C_009499.PDF
  > CLÁUSULA DÉCIMA SEXTA – DO FORO 16.1. As partes elegem, de comum acordo, o foro da Comarca de Erechim, [...]
- `CLÁUSULA DÉCIMA OITAVA` — Ra73e6mdBi1OPsgasLk4kXMpDLd62nB2iWqxdDr5.pdf
  > CLÁUSULA DÉCIMA OITAVA– FORO (ART. 92, §1º) 18.1. Fica eleito o Foro da Justiça Federal em Minas Gerais, [...]
- `CLÁUSULA VIGÉSIMA` — TEXTO+DE+CONTRATO.pdf
  > CLÁUSULA VIGÉSIMA – FORO (art. 92, §1º) Fica eleito o Foro de São Bento do Sul-SC para dirimir os [...]
- `CLÁUSULA DÉCIMA OITAVA` — Contrato.pdf
  > CLÁUSULA DÉCIMA OITAVA– FORO 18.1. Fica eleito o Foro da comarca de João Pessoa-PB para dirimir os [...]
- `CLÁUSULA DÉCIMA SEGUNDA` — Contrato+119+Thiago+ass.pdf
  > CLÁUSULA DÉCIMA SEGUNDA DO FORO Fica eleito o foro da Comarca de Peabiru PR, para dirimir dúvidas ou [...]

**B — híbrido (RRF) + reranking:**

- `CLÁUSULA DÉCIMA OITAVA` — Contrato.pdf (score=0.991)
  > CLÁUSULA DÉCIMA OITAVA– FORO 18.1. Fica eleito o Foro da comarca de João Pessoa-PB para dirimir os [...]
- `CLÁUSULA VIGÉSIMA` — TEXTO+DE+CONTRATO.pdf (score=0.991)
  > CLÁUSULA VIGÉSIMA – FORO (art. 92, §1º) Fica eleito o Foro de São Bento do Sul-SC para dirimir os [...]
- `CLÁUSULA DÉCIMA OITAVA` — Ra73e6mdBi1OPsgasLk4kXMpDLd62nB2iWqxdDr5.pdf (score=0.990)
  > CLÁUSULA DÉCIMA OITAVA– FORO (ART. 92, §1º) 18.1. Fica eleito o Foro da Justiça Federal em Minas Gerais, [...]
- `CLÁUSULA DÉCIMA SEXTA` — C_009499.PDF (score=0.989)
  > CLÁUSULA DÉCIMA SEXTA – DO FORO 16.1. As partes elegem, de comum acordo, o foro da Comarca de Erechim, [...]
- `CLÁUSULA DÉCIMA SEGUNDA` — Contrato+119+Thiago+ass.pdf (score=0.803)
  > CLÁUSULA DÉCIMA SEGUNDA DO FORO Fica eleito o foro da Comarca de Peabiru PR, para dirimir dúvidas ou [...]

## 9. "quais as sanções administrativas previstas"

**A — denso-puro (só embeddings):**

- `CLÁUSULA DÉCIMA OITAVA` — CONTRATO26.2026BOMJESUSDOGURGUEIA.pdf
  > CLÁUSULA DÉCIMA OITAVA - DAS PENALIDADES 18.1. As penalidades administrativas aplicáveis à Contratada, [...]
- `CLÁUSULA OITAVA` — Contrato nº 141-2026- Dispensa de Licitação nº 027-2026- Serviço de dedetização- MAX IMUNIZAÇÃO E SERVIÇOS LTDA.pdf
  > CLÁUSULA OITAVA – DAS PENALIDADES 8.1. Constituem ilícitos administrativos as condutas previstas no art. [...]
- `CLÁUSULA NONA` — _var_www_html_ararangua.atende.net_temp_WCO_PNCP_Contrato_CONTRATO_6_2026_IPM.pdf
  > 11 12. CLÁUSULA DÉCIMA SEGUNDA – SANÇÕES ADMINISTRATIVAS 12.1. Com fulcro na Lei Federal n. 14.133/2021 [...]
- `CLÁUSULA DÉCIMA TERCEIRA` — 342d3edfbee4f61e2b9c4c03e1732b5f2819d125.pdf
  > CLÁUSULA DÉCIMA TERCEIRA - DAS PENALIDADES CABÍVEIS 13.1. O descumprimento total ou parcial das [...]
- `CLÁUSULA DÉCIMA SEGUNDA > §9º` — Ra73e6mdBi1OPsgasLk4kXMpDLd62nB2iWqxdDr5.pdf
  > §9º, da Lei nº 14.133, de 2021) 12.4. Todas as sanções previstas neste Contrato poderão ser aplicadas [...]

**B — híbrido (RRF) + reranking:**

- `CLÁUSULA NONA` — _var_www_html_ararangua.atende.net_temp_WCO_PNCP_Contrato_CONTRATO_6_2026_IPM.pdf (score=0.977)
  > 11 12. CLÁUSULA DÉCIMA SEGUNDA – SANÇÕES ADMINISTRATIVAS 12.1. Com fulcro na Lei Federal n. 14.133/2021 [...]
- `CLÁUSULA DÉCIMA OITAVA` — CONTRATO26.2026BOMJESUSDOGURGUEIA.pdf (score=0.964)
  > CLÁUSULA DÉCIMA OITAVA - DAS PENALIDADES 18.1. As penalidades administrativas aplicáveis à Contratada, [...]
- `CLÁUSULA DÉCIMA TERCEIRA` — 342d3edfbee4f61e2b9c4c03e1732b5f2819d125.pdf (score=0.961)
  > CLÁUSULA DÉCIMA TERCEIRA - DAS PENALIDADES CABÍVEIS 13.1. O descumprimento total ou parcial das [...]
- `CLÁUSULA OITAVA` — Contrato nº 141-2026- Dispensa de Licitação nº 027-2026- Serviço de dedetização- MAX IMUNIZAÇÃO E SERVIÇOS LTDA.pdf (score=0.871)
  > CLÁUSULA OITAVA – DAS PENALIDADES 8.1. Constituem ilícitos administrativos as condutas previstas no art. [...]
- `CLÁUSULA DÉCIMA TERCEIRA` — Contrato+n+0302026+PREF+TRECHO+IMBUIA+2+100+.pdf (score=0.859)
  > Página 11 de 15 realizada de forma incompleta ou em desconformidade com as condições avençadas, sobre o [...]

## 10. "art. 78"

**A — denso-puro (só embeddings):**

- `CLÁUSULA DÉCIMA SÉTIMA` — C_009499.PDF
  > CLÁUSULA DÉCIMA SÉTIMA – DAS DISPOSIÇÕES FINAIS 17.1 E, por estarem justos e acordados, as partes acima [...]
- `CLÁUSULA DÉCIMA - OBRIGAÇÕES` — TEXTO+DE+CONTRATO.pdf
  > CLÁUSULA DÉCIMA - OBRIGAÇÕES DO CONTRATADO (art. 92, XIV, XVI e XVII) executar o objeto do presente [...]
- `CLÁUSULA SEGUNDA` — SEI14813041TermoAditivoProrrogaoedecrscimo.pdf
  > CLÁUSULA SEGUNDA – DO FUNDAMENTO LEGAL O presente termo aditivo é celebrado com base nas disposições dos [...]
- `CLÁUSULA OITAVA - DA` — 1769616745717-529927364-_-202509243__CONTRATO_EXT_E_PUBLICAO_E_CERTIDES.pdf
  > 51 09 -08-11 MUNICÍPIO DE AÇAILÂNDIA - MA SECRETARIA MUNICIPAL DE INFRAESTRUTURA E URBANISMO demais [...]
- `CLÁUSULA DÉCIMA SEGUNDA` — 342d3edfbee4f61e2b9c4c03e1732b5f2819d125.pdf
  > CLÁUSULA DÉCIMA SEGUNDA - DAS OBRIGAÇÕES DA CONTRATADA 12.1. Executar os serviços conforme [...]

**B — híbrido (RRF) + reranking:**

- `PREÂMBULO` — CONTRATO+0722026+ass+cel+Ronildo-2545466046187100704.pdf (score=0.178)
  > 2 de 11 ITEM ESPECIFICAÇÃO UNIDADE QTD. VALOR UNITÁRIO VALOR TOTAL densidade D-45, dimensões 78 x 188 x [...]
- `CLÁUSULA DÉCIMA - OBRIGAÇÕES` — TEXTO+DE+CONTRATO.pdf (score=0.028)
  > CLÁUSULA DÉCIMA - OBRIGAÇÕES DO CONTRATADO (art. 92, XIV, XVI e XVII) executar o objeto do presente [...]
- `PREÂMBULO` — Contrato nº 141-2026- Dispensa de Licitação nº 027-2026- Serviço de dedetização- MAX IMUNIZAÇÃO E SERVIÇOS LTDA.pdf (score=0.012)
  > DISPENSA DE LICITAÇÃO Nº 027/2026 Fls. 1/8 CONTRATO Nº 141/2026 DISPENSA DE LICITAÇÃO Nº 027/2026 Pelo [...]
- `CLÁUSULA DÉCIMA SÉTIMA` — C_009499.PDF (score=0.003)
  > CLÁUSULA DÉCIMA SÉTIMA – DAS DISPOSIÇÕES FINAIS 17.1 E, por estarem justos e acordados, as partes acima [...]
- `CLÁUSULA SEGUNDA` — SEI14813041TermoAditivoProrrogaoedecrscimo.pdf (score=0.003)
  > CLÁUSULA SEGUNDA – DO FUNDAMENTO LEGAL O presente termo aditivo é celebrado com base nas disposições dos [...]
