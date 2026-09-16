# Plano de construção — Crivo

RAG para due diligence contratual. Nove fases, cada uma com critério de aceite binário.

- **Esforço estimado:** 45–55 h, 5 a 7 semanas em ritmo noturno
- **Custo de API:** abaixo de US$ 10 no total (embeddings e reranking rodam local)

---

## 1. A tese

Qualquer pessoa pluga um PDF num vector store em 40 linhas. O que quase ninguém entrega é **prova** de que o sistema não alucina. Este projeto existe para produzir essa prova: uma tabela de métricas antes/depois e um mecanismo de citação verificada programaticamente.

O domínio jurídico foi escolhido porque é onde a alucinação é inaceitável e a linguagem é densa, hierárquica e cheia de termos técnicos.

**Pergunta de demonstração:** *"Quais são as cláusulas rescisórias e as penalidades por atraso previstas nos contratos anexados?"* — toda decisão técnica deve ser justificável por quanto melhora essa resposta.

---

## 2. Corpus

Contratos públicos brasileiros — publicáveis no repositório sem risco de vazamento e verificáveis por quem avaliar.

- **PNCP** (Portal Nacional de Contratações Públicas) — contratos administrativos, aditivos, termos de rescisão
- **Concessões** de agências reguladoras (ANEEL, ANTT, ANATEL) — documentos longos com penalidades e indicadores
- **Termos de uso e contratos de adesão** de grandes empresas — linguagem contratual diferente, testa generalização
- **Contratos sociais e alterações** de juntas comerciais — material clássico de due diligence

Meta: 20–30 documentos, 400–1.500 páginas. Todos com camada de texto (sem OCR na v1).

---

## 3. Arquitetura

```
INGESTÃO (offline)
  PDFs → PyMuPDF → chunking por cláusula → BGE-M3 → ChromaDB + BM25

CONSULTA (tempo real)
  pergunta → busca híbrida (RRF, top-20) → reranker (top-5)
           → Claude Haiku (JSON estruturado) → verificador de citações
           → resposta com fontes  |  ou abstenção
```

---

## 4. Decisões e trade-offs

| Camada | Escolha | Descartado | Por quê |
|---|---|---|---|
| Framework | LlamaIndex só nos readers, core próprio | LangChain ponta a ponta | Abstração pesada esconde o que o projeto precisa demonstrar |
| Parsing | PyMuPDF (+ Docling nos difíceis) | pypdf | Entrega coordenadas e página — necessário para destacar o trecho na UI |
| Chunking | Estrutural por cláusula, com teto de tokens | Splitter de 512 tokens | Cláusula é a unidade semântica e a unidade de citação |
| Embeddings | BGE-M3 local | OpenAI text-embedding-3 | Forte em PT, custo zero, reindexação ilimitada, denso + esparso no mesmo modelo |
| Vector store | ChromaDB persistente | Pinecone | Roda no Compose sem conta nem chave — quem clona consegue rodar |
| Busca | Híbrida denso + BM25, fundidos por RRF | Só vetorial | Termos jurídicos exatos e números de artigo: o léxico ganha |
| Reranking | BGE-reranker-v2-m3 (cross-encoder local) | Nenhum | Maior ganho de precisão por hora investida |
| Geração | Claude Haiku 4.5 | Sonnet como padrão | Aqui a qualidade depende mais do retrieval que do modelo |
| Saída | JSON estruturado via tool use | Texto livre + regex | Citação precisa ser campo tipado para ser verificável |
| API / UI | FastAPI + Streamlit | Next.js | O diferencial não está no front |

---

## 5. As quatro defesas contra alucinação

1. **Abstenção por limiar (recuperação).** Se o score do reranker no melhor chunk fica abaixo do corte calibrado, o LLM nem é chamado.
2. **Contexto numerado e prompt fechado.** Chunks entram como `[1]…[5]` com documento, cláusula e página; a instrução é responder só com o que está entre as fontes.
3. **Verificação literal da citação (saída).** Cada `quote` é casada por fuzzy match (≥ 0,90) contra o chunk. Não casou = inventada → rejeita, reprocessa uma vez, depois abstém.
4. **Rastro de origem (interface).** Toda citação é clicável e abre o trecho do PDF destacado.

### Contrato de saída do modelo

```json
{
  "answer": "...",
  "citations": [
    {
      "source_id": 3,
      "document": "contrato_ANEEL_2024.pdf",
      "clause": "CLÁUSULA DÉCIMA SEGUNDA — DA RESCISÃO",
      "page": 27,
      "quote": "trecho literal, verificado contra o chunk"
    }
  ],
  "confidence": "alta | media | baixa",
  "insufficient_context": false
}
```

---

## 6. Roadmap

### Fase 0 — Fundação (2–3 h) ✅

Repo, `pyproject.toml`, estrutura de pacotes, `.env.example`, `Makefile`, `Dockerfile`, `settings.py` com Pydantic Settings.

**Aceite:** `/health` responde 200. *(Feito — `pytest` verde.)*

### Fase 1 — Corpus (3–4 h)

Baixar 20–30 contratos. Rodar `python scripts/check_corpus.py` para triar e gerar o `manifest.csv`. Preencher `doc_type` e `source_url` à mão.

**Aceite:** ≥ 20 PDFs válidos e `manifest.csv` completo.

> Escreva já 10 das 50 perguntas do golden set enquanto lê o corpus. É a hora mais barata de fazer isso.

### Fase 2 — Ingestão e chunking estrutural (7–9 h)

Extração com PyMuPDF preservando página e coordenadas. Detector de hierarquia jurídica brasileira: `CLÁUSULA PRIMEIRA`, `Art. 5º`, `§ 2º`, `I –`, `a)`. Cláusulas longas subdivididas mantendo o cabeçalho como prefixo.

Metadados obrigatórios por chunk:

```python
{
  "chunk_id", "document", "doc_type",
  "clause_path",    # "CLÁUSULA XII > § 2º > inciso IV"
  "clause_title", "page_start", "page_end",
  "char_start", "token_count", "text"
}
```

**Aceite:** amostra de 30 chunks com ≥ 90% de `clause_path` correto, nenhum acima de 1.200 tokens, nenhuma cláusula partida sem herdar o cabeçalho.

### Fase 3 — Índice híbrido (5–6 h)

BGE-M3 em lote → ChromaDB. BM25 paralelo. Fusão RRF. Reranking cross-encoder. Filtros de metadados.

**Aceite:** `make search Q="multa por atraso na entrega"` devolve top-5 com cláusula e página, mais uma comparação escrita entre denso-puro e híbrido+rerank em 10 consultas.

**✅ Atingido.** `scripts/compare_retrieval.py` rodou as 10 consultas contra o corpus real (326 chunks, 20 contratos, após a correção do chunker) e gravou `docs/comparacao_fase3.md`. Resumo (análise completa lá):

- Em 9 das 10 consultas, híbrido+rerank (B) empata ou melhora a ordenação de denso-puro (A); a única que merece checagem manual (garantia contratual) traz um `PREÂMBULO` bem pontuado que vale conferir antes de confiar cegamente.
- O *score* do reranker (não só a posição) é o dado mais valioso que a Fase 3 produz para a Fase 4: nas consultas bem resolvidas, o top-3 fica quase sempre ≥ 0,8; na consulta ambígua de controle ("art. 78"), todos os scores ficam abaixo de 0,18 — evidência real para calibrar `MIN_RERANK_SCORE`.
- A comparação expôs um bug real no chunker (Fase 2): citações de lei quebradas de linha pelo PDF ("...art. 137 da Lei 14.133...") eram detectadas como cabeçalho de artigo genuíno, virando `clause_path` fantasma (`'art. 137 d'`, `'art. 94 d'`). Corrigido em `find_headings()` com uma checagem de contexto (a linha anterior precisa terminar a frase, ou estar em branco, para um "Art. N" contar como cabeçalho) — regressão coberta por teste. **Confirmado contra dado real:** reindexado o corpus inteiro (334 → 326 chunks) e rerodada a comparação — os dois `clause_path` fantasma desapareceram por completo. Ver diário técnico, Fase 3.

### Fase 4 — Geração com citação verificada (7–9 h)

Prompt com contexto numerado, tool use forçando o schema, verificador fuzzy, política de abstenção, um reprocessamento. Logging estruturado de cada consulta.

**Aceite:** 10 perguntas fora do corpus → 10 abstenções, zero respostas inventadas.

**✅ Atingido — 10/10 abstenções**, confirmado contra a Claude API real (`scripts/test_abstention.py`, resultado em `docs/abstencao_fase4.md`). Todos os scores de reranking ficaram entre 0,000 e 0,004 — bem abaixo do `MIN_RERANK_SCORE=0,35` — e a abstenção veio inteiramente da **defesa 1** (o corte por score, antes de qualquer chamada à Claude API): nenhuma das 10 perguntas chegou a gerar uma resposta ou consumir tokens de geração. Isso prova o caso mais óbvio (pergunta sem nenhuma relação com o corpus) com custo zero, mas não exercita as defesas 2–4 (prompt fechado, verificação de citação, reprocessamento) — essa validação mais fina, com perguntas "quase certas" (relacionadas ao domínio mas sem resposta *naquele* corpus específico), fica para o golden set da Fase 7.

**Construído:** `src/rag/generation/` — `prompt.py` (contexto numerado + prompt fechado), `schema.py` (contrato de saída via tool use), `client.py` (chamada à Claude API, injetável), `verify.py` (verificação fuzzy de citação com `rapidfuzz`), `query_log.py` (log estruturado em `data/logs/queries.jsonl`), `answer.py` (orquestra as quatro defesas: limiar de rerank antes de chamar o modelo, prompt fechado, verificação literal, um reprocessamento, abstenção se a verificação falhar de novo). CLI: `python -m rag.generation.cli "pergunta"`. 48 testes no total, toda a orquestração coberta com um `model_caller` falso — nenhum teste gasta chamada de API real.

**Bug real encontrado e corrigido nesta fase (fora do escopo da geração em si):** o `fake_embedder` usado nos testes das Fases 3 e 4 usava `hash()` embutido do Python — que é aleatorizado por processo desde o Python 3.3 (proteção contra hash flooding), não determinístico como o comentário original afirmava. Um teste que sempre passava no ambiente de desenvolvimento falhou na primeira vez que rodou na máquina do usuário. Corrigido trocando por `zlib.crc32`; validado rodando a suíte sob 5 valores diferentes de `PYTHONHASHSEED`. Ver diário técnico, Fase 4.

### Fase 5 — API (4–5 h)

`POST /ingest`, `POST /query` (SSE), `GET /documents`, `/health`, `/metrics`. Pydantic nas fronteiras. Testes com corpus de fixture.

**Aceite:** `/docs` navegável e `pytest` verde, incluindo teste de abstenção.

**Construído:** `src/rag/api/` ganhou seis módulos novos, e `main.py` foi reescrito para só rotear (cada endpoint delega para um módulo específico):

- `schemas.py` — todos os modelos Pydantic de fronteira (`QueryRequest`, `QueryResponse`, `DocumentsResponse`, `MetricsResponse`, `IngestStatusResponse` etc.), num só lugar.
- `documents.py` — `GET /documents`: une o manifest da Fase 1 (`doc_type`, `source_url`, preenchido à mão) com a contagem real de chunks da Fase 2, por documento. União (não interseção) das duas fontes: um documento só no manifest (ainda não chunkado) ou só nos chunks (manifest desatualizado) aparece do mesmo jeito — nenhum dos dois casos desaparece silenciosamente.
- `metrics.py` — `GET /metrics`: só lê e agrega `data/logs/queries.jsonl`, o log estruturado que a própria Fase 4 já escreve a cada consulta (`query_log.log_query`). Nenhuma métrica nova instrumentada — a Fase 5 ganha um endpoint quase de graça, e a Fase 7 vai reler o mesmo arquivo para métricas mais ricas.
- `query_service.py` — `POST /query`: chama `answer_question()` (Fase 4) e converte o `AnswerResult` pro `QueryResponse` da fronteira.
- `query_stream.py` — `POST /query/stream`: a mesma pergunta como Server-Sent Events. Importante deixar explícito: são eventos de **progresso de pipeline** ("buscando fontes" → "gerando resposta" → "verificando citações"), não streaming **token a token** — a saída da Fase 4 é forçada via `tool_choice`, e a Claude API só entrega um bloco `tool_use` completo e validado, não incremental. Um endpoint que fingisse streaming de token aqui estaria mentindo sobre o que a arquitetura permite; ainda assim, progresso por etapa tem valor real numa UI (ver diário técnico). Mecanismo: `answer_question` roda numa thread (é síncrona e bloqueante), o hook `on_stage` empurra eventos numa fila via `loop.call_soon_threadsafe`, e o gerador async consome a fila.
- `ingest_job.py` — `POST /ingest` + `GET /ingest/status`: dispara a reindexação (chunking da Fase 2 + build do índice da Fase 3) em background via `BackgroundTasks` do FastAPI, com o estado (`idle`/`running`/`done`/`error`) num singleton em memória protegido por lock. Limitação documentada e aceita: o estado se perde se o processo reiniciar e não é compartilhado entre múltiplos workers uvicorn — correto para uma API de instância única; um job store real (Redis/banco) seria o próximo passo para produção multi-worker.

`documents.list_documents()` e `metrics.compute_metrics()` seguem o mesmo padrão de injeção de dependência já usado para `Embedder`/`Scorer`/`ModelCaller` (Fase 4), mas aplicado a **caminhos de arquivo**: ambos aceitam parâmetros opcionais (`chunks_file`, `manifest_file`, `log_file`) que default para `settings.*` — os testes nunca tocam `data/` real.

62 testes no total (14 novos em `tests/test_api.py`), toda a suíte passando; `ruff check` limpo nos módulos novos.

**✅ Atingido.** Confirmado na máquina do usuário: `pytest -q` verde (62 testes) e `/docs` navegável com `uvicorn rag.api.main:app --reload`, endpoints exercitados contra o corpus e a Claude API reais.

### Fase 6 — Interface (5–6 h)

Streamlit em dois painéis: conversa e fontes. Citação clicável abre o trecho do PDF destacado. Seletor de documentos. Latência e nº de chunks sempre visíveis.

**Aceite:** clicar numa citação exibe o texto-fonte com documento e página. Abstenção visualmente distinta de resposta fundamentada.

**Construído:** `ui/app.py`, reescrito por completo, em dois painéis lado a lado — conversa (`st.chat_message`/`st.chat_input`) à esquerda, fontes à direita:

- **Citação clicável, literalmente.** Cada citação na resposta vira um botão (`[1] ✅`/`[2] ⚠️`); clicar seleciona aquele `source_id` e abre, no painel de fontes, o **chunk inteiro** (não só o `quote` curto) com o trecho citado destacado dentro do texto completo — documento e página sempre visíveis no cabeçalho do expander. Isso exigiu estender a API: `QueryResponse` (Fase 5) ganhou um campo `sources` novo (`SourceChunk`: `source_id`, `document`, `clause`, `page_start`/`page_end`, `text`, `rerank_score`) — antes só os `citations` (com `quote` curto) cruzavam a fronteira HTTP, e não havia como a UI mostrar o "texto-fonte" completo sem isso.
- **Abstenção visualmente distinta.** Avatar (`🚫` vs `⚖️`) e container diferentes (`st.warning` com o motivo da abstenção, não só uma frase a mais no meio do texto) — não dá pra confundir uma abstenção com uma resposta fundamentada só de olhar a tela.
- **Seletor de documentos real, não cosmético.** A `search()` da Fase 3 ganhou um parâmetro `document_filter`: quando a UI restringe a documentos específicos, a busca híbrida busca mais candidatos antes de filtrar e só então corta em `top_k` — sem isso, filtrar depois de já ter cortado para `top_k=20` quase sempre devolveria lista vazia. Threading completo: `search()` → `answer_question()` → `QueryRequest.documents` (API) → `ui/app.py` (multiselect na sidebar).
- **Progresso real via SSE, não um spinner genérico.** A UI consome `POST /query/stream` (Fase 5) do lado do cliente com `httpx.stream`, atualizando um `st.status` a cada evento `status` ("buscando fontes" → "gerando resposta" → "verificando citações"). Latência (medida no cliente, do primeiro byte enviado ao evento `result`) e número de chunks usados ficam sempre visíveis — na sidebar (última consulta) e como legenda de cada resposta no histórico.

6 testes novos (3 em `tests/test_retrieval.py` para `document_filter`, 1 em `tests/test_generation.py` ponta a ponta, 2 em `tests/test_api.py` para `sources`/`documents`), 67 no total, todos passando; `ruff check` limpo em `src/`, `tests/` e `ui/`. A UI em si não tem teste automatizado formal — `streamlit.testing.v1.AppTest` foi usado manualmente durante o desenvolvimento para confirmar que a tela sobe sem exceção com corpus vazio e que o caminho de erro (API sem chave configurada) aparece como aviso, não como crash.

**Pendente:** confirmação do critério de aceite na máquina do usuário, contra o corpus e a Claude API reais — subir a API e `streamlit run ui/app.py`, fazer uma pergunta com resposta fundamentada, clicar numa citação e conferir que o texto-fonte (documento + página + trecho destacado) aparece corretamente, e confirmar que uma pergunta fora do domínio aparece visualmente como abstenção.

### Fase 7 — Avaliação (8–10 h) — *a fase que vale o projeto*

Golden set de 50 perguntas escritas à mão: 30 factuais, 10 de localização de cláusula, 10 sem resposta possível.

| Configuração | Faithfulness | Context recall | Acurácia de citação | Abstenção correta |
|---|---|---|---|---|
| A — baseline ingênuo (chunk fixo, denso puro) | — | — | — | — |
| B — chunking estrutural + híbrido | — | — | — | — |
| C — B + reranker + verificador | — | — | — | — |

Métricas do `ragas` mais duas próprias, que valem mais: **acurácia de citação** e **taxa de abstenção correta**.

**Aceite:** tabela preenchida com números reais, reprodutível por `make eval`. É ela que abre o README.

### Fase 8 — Empacotamento (4–5 h)

Compose com api + ui + volume do Chroma. README como argumento, não manual. Vídeo de 3 min abrindo com a pergunta de demonstração e fechando com uma armadilha sendo recusada.

**Aceite:** alguém que nunca viu o projeto roda `docker compose up` e pergunta em menos de 10 minutos.

**Adiantado (feito junto com a Fase 6, enquanto a Claude API estava bloqueada por créditos):** a camada de apresentação que não depende de rodar o pipeline.

- `docs/IDENTIDADE.md` — o sistema visual (paleta sóbria, tipografia, regras de uso da cor semântica), fonte única de verdade consumida pela UI, pelo diagrama e pelo README. Mesmo princípio do `settings.py`: nenhum consumidor inventa um valor próprio.
- `assets/` — logo, símbolo, favicon e avatares. Renomeado de "Cláusula" para **Crivo** antes da publicação (o nome descreve o mecanismo — uma peneira de quatro defesas sucessivas — em vez do objeto lido; ver `docs/DIARIO_TECNICO.md`). O símbolo é um "C" com a marca de verificação encaixada na abertura — o mecanismo do produto, não um ornamento. O nome no logo está convertido em contorno vetorial, para não depender da fonte estar instalada em quem abre o arquivo.
- `assets/arquitetura.svg` — o pipeline completo com as quatro defesas ancoradas nos pontos onde agem, gerado por script (`scripts/gerar_diagrama.py`) para manter o alinhamento do grid.
- `README.md` reescrito em formato de argumento: tese → resultados (tabela da Fase 7, ainda vazia e declarada como vazia) → como funciona → as quatro defesas → **o que o projeto não faz** → como rodar → decisões técnicas.

**Pendente da Fase 8:** capturas de tela com o corpus real, vídeo de 3 minutos, revisão do Compose, e o preenchimento da tabela de resultados (que depende da Fase 7).

---

## 7. Riscos

| Risco | Sinal de alerta | Mitigação |
|---|---|---|
| Chunking quebra em contratos mal formatados | Muitos chunks sem `clause_path` | Fallback semântico; medir e reportar a taxa de fallback |
| BGE-M3 lento em CPU | Indexar 1.000 páginas passa de 1 h | Lote, cache por hash do chunk, Colab grátis para a indexação inicial |
| Perguntas cross-document | Respostas parciais e confusas | Fora do escopo da v1 — declarar como limitação no README |
| Escopo crescendo | Reescrevendo a UI na semana 5 | Congelar funcionalidades ao fim da Fase 5 |
| Travar na avaliação | "Faço o golden set depois" | Escrever 10 perguntas já na Fase 1 |

---

## 8. Apresentação

1. **README abre com a tabela de resultados**, não com instalação.
2. **Vídeo de 3 minutos:** pergunta real → clique numa citação abrindo o PDF → pergunta-armadilha sendo recusada. O terceiro momento é o que fica.
3. **Post técnico curto** sobre uma decisão só — sugestão: "por que chunking de 512 tokens destrói contratos", com antes/depois do corpus real.

### Três perguntas que você vai ouvir

- *"Como você sabe que não alucina?"* → a tabela da Fase 7 e as 10 armadilhas.
- *"Por que não usou só busca vetorial?"* → a comparação da Fase 3, com um termo jurídico que o BM25 acerta e o denso erra.
- *"O que faria diferente com mais tempo?"* → decomposição de consultas para cross-document e fine-tuning do embedding no domínio jurídico.
