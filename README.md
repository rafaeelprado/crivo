<div align="center">

<img src="assets/logo.svg" alt="Crivo" width="250">

**Due diligence contratual com citações verificadas contra o texto de origem.**

Responde perguntas sobre um acervo de contratos citando documento, cláusula, página e trecho literal — e **se recusa a responder** quando a resposta não está nos documentos.

![Python](https://img.shields.io/badge/Python-3.11+-14243D?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-14243D?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-14243D?style=flat-square)
![BGE--M3](https://img.shields.io/badge/BGE--M3-local-9C7B4D?style=flat-square)
![Claude](https://img.shields.io/badge/Claude_Haiku-9C7B4D?style=flat-square)
![Licença](https://img.shields.io/badge/licen%C3%A7a-MIT-5A6472?style=flat-square)

</div>

<!-- ─────────────────────────────────────────────────────────────────────────
     CAPTURA DE TELA — inserir aqui, com o corpus real indexado:
       1. assets/captura-resposta.png  → resposta com citação clicada e o
          texto-fonte aberto no painel da direita
       2. assets/captura-abstencao.png → pergunta fora do corpus, abstenção
     Enquanto não existirem, este bloco fica comentado (o GitHub renderizaria
     um ícone de imagem quebrada).

<img src="assets/captura-resposta.png" alt="Resposta com citação verificada e o texto-fonte aberto">
────────────────────────────────────────────────────────────────────────── -->

---

## A tese

Qualquer pessoa pluga um PDF num vector store em 40 linhas. O que quase ninguém entrega é **prova** de que o sistema não inventa.

Análise de contratos é um dos poucos lugares onde uma IA que alucina é pior que nenhuma IA. Um analista de M&A que recebe *"a multa rescisória é de 10% do valor do contrato"* precisa saber **em qual cláusula, de qual documento, em qual página** isso está escrito — ou a informação não vale nada, porque ele vai ter que conferir tudo de novo na mão.

A maior parte dos projetos de RAG resolve a parte fácil (achar textos parecidos) e ignora a difícil (provar que a resposta veio deles). Aqui, a segunda parte **é** o produto: toda citação que sai do sistema foi casada, caractere a caractere, contra o chunk de onde ela diz ter vindo. A que não casa não chega ao usuário.

## Resultados

Medido contra um golden set de 50 perguntas escritas à mão (30 factuais, 10 de localização de cláusula, 10 propositalmente sem resposta no corpus), rodado três vezes — uma por configuração — para isolar a contribuição de cada componente do pipeline.

| Configuração | Abstenção correta | Acurácia de citação | Abstenção indevida | Erros |
|---|---|---|---|---|
| **A** — baseline ingênuo (chunk fixo de 512 tokens, busca densa pura) | 100% | 100% | 10,0% | 0 |
| **B** — chunking estrutural + busca híbrida (sem reranking) | 100% | 100% | 7,5% | 0 |
| **C** — B + reranking + verificação de citação (Crivo completo) | 100% | 100% | 7,5% | 0 |

**Abstenção correta** — das 10 perguntas sem resposta no corpus, quantas o modelo recusou corretamente (ideal: 100%). **Acurácia de citação** — das citações emitidas, quantas bateram, caractere a caractere, com o texto de origem (ideal: 100%). **Abstenção indevida** — das 40 perguntas com resposta no corpus, quantas o modelo recusou por engano (ideal: 0%).

Três achados — incluindo os que não confirmam a hipótese inicial:

- **Zero alucinação nas 150 chamadas** (50 perguntas × 3 configurações): toda citação emitida bateu com o texto de origem, em todas as configurações, inclusive na baseline mais simples.
- **A busca híbrida reduz abstenção indevida em 2,5pp** sobre a busca densa pura (10,0% → 7,5%) — o ganho de combinar recuperação lexical (BM25) com semântica.
- **O reranking não mudou nenhuma métrica nesta base** (B → C, 20 documentos, 326 chunks): a fusão por RRF já colocava o chunk correto no topo na maioria das consultas, sem espaço para o cross-encoder melhorar o resultado. Não invalida a técnica — indica que, neste corpus, o gargalo não estava ali. Raciocínio completo em [`docs/DIARIO_TECNICO.md`](docs/DIARIO_TECNICO.md).

O que mais está medido, contra o corpus real de 20 contratos administrativos (326 chunks):

- **10/10 abstenções** em perguntas fora do domínio na Fase 4, todas *antes* de chamar a Claude API, no corte por score de relevância — custo de geração: zero. ([relatório](docs/abstencao_fase4.md))
- **9 de 10 consultas** em que busca híbrida + reranking empata ou melhora sobre busca densa pura na Fase 3, com a análise consulta a consulta de onde e por quê. ([comparação](docs/comparacao_fase3.md))
- **67 testes** verdes, incluindo a orquestração completa de geração com um `model_caller` falso — a suíte roda em segundos e não gasta um token de API.

## Como funciona

<img src="assets/arquitetura.svg" alt="Arquitetura do Crivo: ingestão offline e consulta em tempo real, com as quatro defesas contra alucinação marcadas nos pontos onde agem" width="100%">

O pipeline tem dois tempos. **Offline**, uma vez por corpus: os PDFs são lidos preservando o número da página, cortados em chunks **por cláusula** (não a cada 512 tokens — uma cláusula partida ao meio é uma regra partida ao meio), embutidos com BGE-M3 local e indexados em ChromaDB e BM25 ao mesmo tempo. **Em tempo real**, a cada pergunta: busca nos dois índices, fusão por Reciprocal Rank Fusion, reranking por cross-encoder, e só então — se o melhor resultado passar do limiar de confiança — a pergunta chega ao modelo, com as fontes numeradas.

### As quatro defesas contra alucinação

| # | Defesa | Onde atua | O que ela impede |
|---|---|---|---|
| **1** | **Abstenção por limiar** — se o melhor chunk fica abaixo do score mínimo do reranker, o modelo não é chamado | Recuperação | O modelo "tentar ajudar" com contexto ruim |
| **2** | **Contexto numerado e prompt fechado** — as fontes entram como `[1]…[5]` com documento, cláusula e página; a instrução proíbe conhecimento externo | Prompt | Resposta apoiada no que o modelo memorizou no treino |
| **3** | **Verificação literal da citação** — cada trecho citado é casado por fuzzy matching (≥ 0,90) contra o chunk de origem; se não casa, foi inventado | Saída | Citação plausível mas fabricada |
| **4** | **Rastro de origem** — toda citação na interface abre o texto-fonte completo, com o trecho destacado | Interface | Que o usuário tenha que confiar sem poder conferir |

Se a citação falhar na verificação, o sistema reprocessa **uma vez**, apontando ao modelo exatamente qual trecho não bateu. Se falhar de novo, ele abstém — porque uma resposta sem nenhuma citação confirmada é pior que nenhuma resposta.

## O que este projeto não faz

Um README de portfólio que só lista virtudes não é informação. As limitações abaixo são **por design**, e cada uma tem um motivo:

- **Não responde perguntas cross-document** ("compare as multas dos três contratos"). Isso exige decomposição de consulta, que é um projeto em si. Declarado como trabalho futuro, não escondido atrás de uma resposta parcial.
- **Não faz OCR.** PDFs escaneados sem camada de texto são rejeitados na triagem (`scripts/check_corpus.py`), com a métrica de caracteres por página à vista. Um PDF que entra sem texto polui o índice silenciosamente — e você só descobre três fases depois, tentando entender por que uma pergunta óbvia não acha nada.
- **O streaming é de progresso, não de token.** O endpoint `POST /query/stream` emite as etapas do pipeline ("buscando fontes" → "gerando resposta" → "verificando citações"), não a resposta palavra a palavra. É uma consequência direta de forçar o formato de saída via *tool use*: a API entrega o bloco validado inteiro, não incrementalmente. Fingir streaming de token aqui seria mentir sobre o que a arquitetura permite.
- **O estado do job de ingestão vive em memória.** Suficiente para uma API de instância única; não sobrevive a restart nem é compartilhado entre múltiplos workers. Um job store real (Redis, uma tabela) é o próximo passo para produção — mas seria desproporcional aqui.
- **Os limiares foram validados, não otimizados.** Os valores no `.env.example` vieram da distribuição observada em 10 consultas (Fase 3) e se sustentaram nas 50 perguntas do golden set da Fase 7 — mas nunca passaram por uma busca sistemática (grid search) sobre o espaço de hiperparâmetros. É a diferença entre "não falhou nos testes" e "é o ponto ótimo".

## Rodar localmente

> Guia de referência rápida, com tabela de erros comuns: **[`docs/COMO_RODAR.md`](docs/COMO_RODAR.md)**.

### Pré-requisitos

- Python 3.11 ou superior
- Uma chave da [Claude API](https://console.anthropic.com/settings/keys) **vinculada a um workspace** (uma chave de organização é recusada com `not scoped to a workspace`)
- Docker Desktop (opcional, para o caminho containerizado)

### 1. Instalação

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
notepad .env          # cole sua ANTHROPIC_API_KEY
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

### 2. Montar o índice a partir do seu corpus

Coloque os PDFs em `data/raw/` e rode, em ordem:

```bash
python scripts/check_corpus.py           # triagem dos PDFs, gera manifest.csv
python -m rag.ingest.pipeline --verbose  # chunking estrutural -> data/processed/chunks.jsonl
python -m rag.ingest.indexer             # embeddings + BM25 -> data/chroma/
```

Na primeira execução, o BGE-M3 e o reranker são baixados (~2,5 GB somados). Depois disso, tudo roda local e offline — só a geração da resposta usa a rede.

```bash
python -m rag.retrieval.cli "multa por atraso na entrega"    # só a busca
python -m rag.generation.cli "multa por atraso na entrega"   # busca + resposta verificada
```

### 3. Subir a API e a interface

```bash
uvicorn rag.api.main:app --reload   # API em :8000, documentação em /docs
streamlit run ui/app.py             # interface em :8501
```

### 4. Rodar os testes

```bash
pytest -q
```

### Caminho Docker

```bash
docker compose up --build
```

> **Windows:** o `Makefile` é um atalho para quem tem `make`. Sem ele, use os comandos completos acima — são exatamente o que cada alvo do Makefile executa.

## Decisões técnicas

Este projeto documenta o **porquê** de cada decisão, não só o como:

- **[`docs/PLANO.md`](docs/PLANO.md)** — as nove fases, com critério de aceite binário por fase, a tabela de trade-offs de cada camada (e o que foi descartado), e os riscos com suas mitigações.
- **[`docs/DIARIO_TECNICO.md`](docs/DIARIO_TECNICO.md)** — o diário de desenvolvimento: os conceitos que sustentam cada escolha, três bugs reais encontrados com dado real (um deles só aparecia em outra máquina, por causa da aleatorização de `hash()` no Python) e o raciocínio por trás de cada trade-off.
- **[`docs/comparacao_fase3.md`](docs/comparacao_fase3.md)** — busca densa vs. híbrida + reranking, 10 consultas, análise caso a caso.
- **[`docs/abstencao_fase4.md`](docs/abstencao_fase4.md)** — as 10 perguntas fora do corpus e o que o resultado prova (e o que ainda não prova).
- **[`docs/IDENTIDADE.md`](docs/IDENTIDADE.md)** — o sistema visual: paleta, tipografia e as regras de uso da cor semântica.

## Estrutura

```
sistema-rag/
├── assets/             logo, favicon e o diagrama de arquitetura
├── data/
│   ├── raw/            PDFs originais + manifest.csv   (fora do git)
│   ├── processed/      chunks.jsonl                    (fora do git)
│   └── chroma/         índice vetorial persistido      (fora do git)
├── src/rag/
│   ├── settings.py     configuração central — toda constante mora aqui
│   ├── ingest/         Fase 2 — parsing e chunking por cláusula
│   ├── retrieval/      Fase 3 — busca híbrida e reranking
│   ├── generation/     Fase 4 — geração e verificação de citações
│   ├── api/            Fase 5 — FastAPI
│   └── eval/           Fase 7 — golden set e métricas
├── ui/app.py           Fase 6 — interface Streamlit
├── scripts/            utilitários de linha de comando
├── tests/              67 testes, nenhum gasta chamada de API
└── docs/               plano, diário técnico e relatórios de cada fase
```

## Roadmap

| Fase | Entrega | Estado |
|---|---|---|
| 0 | Fundação do repositório, API `/health`, Docker | ✅ |
| 1 | Corpus: 20 contratos públicos reais + manifest | ✅ |
| 2 | Parsing e chunking estrutural por cláusula | ✅ |
| 3 | Índice híbrido e reranking | ✅ |
| 4 | Geração com citação verificada | ✅ |
| 5 | API completa | ✅ |
| 6 | Interface com auditoria de fontes | ✅ |
| 7 | Suíte de avaliação — golden set de 50 perguntas | ✅ |
| 8 | Empacotamento e apresentação | ⬜ |

## Licença

MIT.
