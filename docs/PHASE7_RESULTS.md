# Fase 7 - Resultados da Avaliacao (Golden Set)
## Resumo Agregado - Comparacao das 3 Configuracoes
| Configuracao | Abstencao Correta | Acuracia de Citacao | Abstencao Indevida | Erros |
|---|---|---|---|---|
| **A** Baseline ingenuo (denso puro, sem reranking) | 100.0% | 100.0% | 10.0% | 0 |
| **B** Chunking estrutural + Hibrido (sem reranking) | 100.0% | 100.0% | 7.5% | 0 |
| **C** Completo: Crivo (estrutural + hibrido + reranking + verificador) | 100.0% | 100.0% | 7.5% | 0 |

## Interpretacao

- **Abstencao Correta**: das 10 perguntas sem resposta, quantas o modelo recusou corretamente? (ideal: 100%)
- **Acuracia de Citacao**: das citacoes feitas, quantas bateram com texto verificado? (ideal: 100%)
- **Abstencao Indevida**: das 40 perguntas com resposta, quantas o modelo recusou indevidamente? (ideal: 0%)
- **Erros**: excecoes ou timeouts durante a avaliacao

## Ganhos de Cada Componente

**A -> B** (Chunking estrutural + Hibrido): +0.0pp de acuracia de citacao

**B -> C** (Reranking + Verificador): +0.0pp de acuracia de citacao

**A -> C** (Todos os componentes): +0.0pp de acuracia de citacao

