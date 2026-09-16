# Diário Técnico — Crivo (RAG para Due Diligence Contratual)

> **O que é este documento:** seu material de estudo pessoal sobre o projeto. Não é
> documentação do código (isso é o `docs/PLANO.md` e os comentários no código) — é a
> explicação de **por que** cada peça existe e **o que você precisa saber** para
> defender essas decisões numa entrevista técnica. Vou atualizando este arquivo a
> cada fase concluída. Leia-o de tempos em tempos, não só no fim do projeto — o
> objetivo é que o conhecimento fique com você, não só o código.

**Status:** Fases 0 a 5 concluídas. Fase 3 fechada (comparação escrita de 10 consultas, confirmada contra dado real). Fase 4 (geração com citação verificada) fechada — 10/10 abstenções confirmadas contra a Claude API real. Fase 5 (API completa) fechada — `/docs` navegável e suíte verde (62 testes), confirmado na máquina do usuário. Fase 6 (interface Streamlit) construída e testada (67 testes), pendente de confirmação do critério de aceite (clicar numa citação) na máquina do usuário. Camada de apresentação (identidade visual, diagrama de arquitetura, README como argumento) adiantada da Fase 8. Última atualização: apresentação.

---

## 1. O problema que o projeto resolve (e por que isso importa)

Um LLM sozinho tem dois defeitos para uso corporativo:

1. **Conhecimento estático** — o modelo não sabe o conteúdo dos *seus* contratos, porque eles nunca estiveram nos dados de treino.
2. **Alucinação** — quando não sabe, o modelo tende a *inventar* uma resposta plausível em vez de dizer "não sei". Em direito e finanças, isso é inaceitável: uma multa inventada num parecer pode custar uma decisão de negócio.

**RAG (Retrieval-Augmented Generation)** resolve os dois problemas com uma ideia simples: em vez de perguntar direto ao LLM, primeiro *recupera* os trechos relevantes dos seus documentos e só depois pede ao LLM para responder **usando apenas esses trechos**. O LLM deixa de ser uma fonte de conhecimento e passa a ser um "leitor e sintetizador" de um contexto que você controla.

Isso não elimina a alucinação sozinho — um LLM ainda pode inventar uma citação mesmo tendo o texto certo na frente. É por isso que este projeto vai além do RAG básico: cada resposta passa por um **verificador que casa a citação contra o texto de origem, literalmente**, antes de chegar ao usuário (isso é a Fase 4, ainda não construída — mas toda a Fase 2 e 3 já foram desenhadas em função dela).

**Se alguém te perguntar "o que é RAG" numa entrevista**, a resposta em uma frase: *"É uma arquitetura que separa recuperação de informação (busca) da geração de texto (LLM), usando a busca para fundamentar a geração em documentos reais em vez de depender só do que o modelo memorizou no treino."*

---

## 2. Stack e por que cada peça foi escolhida

| Camada | Escolha | Alternativa comum | Por que esta e não aquela |
|---|---|---|---|
| Linguagem | Python 3.12 | — | Padrão de fato em ML/IA; todo o ecossistema (transformers, chromadb) é Python-first |
| Config | Pydantic Settings | variáveis soltas / `os.environ` | Validação de tipo automática + uma fonte única da verdade. Ver seção 4. |
| Parsing de PDF | PyMuPDF (`pymupdf`) | `pypdf`, `pdfplumber` | Entrega coordenadas e número de página nativamente — essencial pra citar a página certa |
| Embeddings | BGE-M3 (`BAAI/bge-m3`), local | OpenAI `text-embedding-3` | Multilíngue forte em PT-BR, custo zero por reindexação, roda em CPU |
| Banco vetorial | ChromaDB | Pinecone, Weaviate | Persiste em disco sem servidor externo — quem clona o repo roda sem criar conta em lugar nenhum |
| Busca léxica | BM25 (`rank_bm25`) | Elasticsearch | Mesma razão: zero infraestrutura externa, e BM25 puro já resolve o caso de uso (termo exato) |
| Reranking | Cross-encoder `BAAI/bge-reranker-v2-m3` | Cohere Rerank (API paga) | Local, sem custo por chamada, mesma família do embedder (M3) |
| Geração | Claude Haiku 4.5 | GPT-4o-mini | Barato, rápido, e a qualidade da resposta aqui depende muito mais do retrieval que do LLM |
| API | FastAPI | Flask | Validação automática via Pydantic + documentação OpenAPI de graça |

**A pergunta que vem depois de qualquer tabela dessas numa entrevista é "e se você tivesse que trocar X por Y, o que quebraria?"** — pense nisso enquanto lê o resto do documento: cada componente foi isolado atrás de uma interface (você vai ver isso repetido nas Fases 2 e 3) exatamente para que essa troca seja barata.

---

## 3. Conceitos fundamentais — o que sustenta cada decisão

Esta seção é o "porquê" que fica por trás do código. É o que separa "eu segui um tutorial" de "eu sei por que meu sistema funciona assim".

### 3.1 Embeddings e similaridade vetorial

Um embedding transforma um texto num vetor de números (no BGE-M3, 1024 dimensões) de tal forma que **textos com significado parecido ficam geometricamente próximos** nesse espaço. "Multa por atraso" e "penalidade moratória" viram vetores próximos mesmo sem compartilhar uma palavra — é assim que a busca semântica encontra parecidos, não apenas idênticos.

A proximidade é medida por **similaridade de cosseno**: o ângulo entre dois vetores, não a distância. Dois vetores apontando na mesma direção (ângulo 0°) têm similaridade 1; perpendiculares (90°) têm 0. Por isso o projeto normaliza os embeddings antes de salvar (`normalize_embeddings=True` no `indexer.py`) — com vetores unitários, a distância euclidiana ao quadrado do ChromaDB (`d² = 2 - 2·cos_sim`) fica matematicamente equivalente a ranquear por cosseno, sem ter que calcular cosseno manualmente.

**Pergunta de entrevista provável:** *"Por que normalizar os embeddings?"* — porque sem normalização, um vetor "maior" (mais magnitude) pode parecer mais similar por tamanho, não por direção/significado. Normalizar remove a magnitude da equação e deixa só o ângulo, que é o que carrega o significado.

### 3.2 Busca densa vs. busca léxica — por que as duas

- **Densa (embeddings + ChromaDB):** entende sinônimo e paráfrase. Ótima pra "quais as penalidades" encontrar um trecho que diz "multa moratória" sem usar a palavra "penalidade".
- **Léxica (BM25):** casa termo exato. Essencial para "Art. 78, IV" ou "CLÁUSULA DÉCIMA" — um embedding pode achar isso "parecido" com qualquer outro artigo de lei, porque semanticamente *são* parecidos. BM25 não erra isso porque não tenta entender significado, só conta frequência de termo.

Nenhuma das duas sozinha resolve bem um contrato jurídico, porque a linguagem jurídica mistura os dois regimes na mesma página: prosa normal (onde sinônimo importa) e referências numéricas exatas (onde só o termo exato serve). **Por isso a fusão híbrida — não é redundância, é cobertura de dois modos de falha diferentes.**

### 3.3 Reciprocal Rank Fusion (RRF) — como funde duas listas incompatíveis

O problema: a busca densa devolve *distância euclidiana* (0.02, 1.6...) e o BM25 devolve um *score de frequência ponderada* (pode ser 3.4, 11.8, qualquer escala dependendo do corpus). Comparar esses números diretamente não faz sentido — não são a mesma unidade.

A solução do RRF é ignorar o valor do score e usar só a **posição no ranking**:

```
rrf_score(chunk) = Σ  1 / (k + posição_do_chunk_na_lista)
```

somado por cada lista onde o chunk aparece. `k=60` é uma constante da literatura original (Cormack et al., 2009) que suaviza a diferença entre a 1ª e a 2ª posição sem dar peso desproporcional ao topo. Um chunk que aparece bem posicionado nas duas listas (denso e léxico) acumula os dois termos e sobe; um chunk que só uma lista encontrou soma só uma vez.

**Pergunta de entrevista provável:** *"Por que não normalizar os scores e somar direto, em vez de usar RRF?"* — porque normalização (min-max, z-score) pressupõe que a distribuição dos scores é comparável entre os dois sistemas, o que geralmente é falso na prática — BM25 pode ter uma distribuição de cauda longa muito diferente da distância vetorial. RRF é *rank-based*, não *score-based*: funciona mesmo quando as escalas são incomensuráveis, ao custo de perder a magnitude da diferença entre 1º e 2º lugar.

### 3.4 Reranking com cross-encoder — por que uma segunda passada

Existem dois jeitos de comparar pergunta e passagem com um modelo:

- **Bi-encoder** (o que o embedding faz): embeda a pergunta e o texto *separadamente*, depois compara os vetores. Rápido — dá pra pré-computar o embedding de milhões de chunks uma vez e comparar depois. Mas perde precisão porque nunca "olha os dois juntos".
- **Cross-encoder** (o reranker): recebe o par (pergunta, texto) **concatenado** e processa tudo junto numa única passada pelo modelo. Muito mais preciso, porque o modelo pode fazer atenção cruzada entre pergunta e texto. Mas é caro — não dá pra rodar contra todo o corpus, só contra um conjunto pequeno já pré-filtrado.

Por isso o pipeline é em cascata: busca híbrida (rápida, sobre milhares de chunks) → top-20 → reranking (caro, só sobre 20) → top-5. É o padrão *retrieve-then-rerank*, usado em praticamente todo sistema de busca sério (inclusive motores de busca web).

**Pergunta de entrevista provável:** *"Por que não usar só o cross-encoder desde o início?"* — custo computacional: um cross-encoder faz uma inferência completa de transformer por par (pergunta, chunk). Rodar isso contra 9990 contratos por consulta seria proibitivamente lento. A busca híbrida existe para reduzir o espaço de busca a um tamanho onde o cross-encoder é viável.

### 3.5 Chunking — por que é o problema mais subestimado de RAG

A maioria dos tutoriais de RAG corta o texto em pedaços de tamanho fixo (ex: 512 tokens, com sobreposição). Isso funciona para prosa genérica, mas **quebra a unidade lógica de um documento estruturado**: uma cláusula de multa pode ter a condição no início e o valor no fim, e um corte arbitrário separa as duas em chunks diferentes — o sistema recupera metade da regra.

Este projeto faz **chunking estrutural**: a unidade de corte é a cláusula (ou o artigo), detectada por um parser de hierarquia (regex sobre os padrões `CLÁUSULA`, `Art.`, `§`, incisos romanos, alíneas). Isso garante que o chunk = a unidade semântica = a unidade que um advogado citaria.

**O trade-off que você precisa saber defender:** chunks estruturais têm tamanho variável — uma cláusula pode ter 50 tokens ou 2000. O projeto lida com isso com um teto (`max_chunk_tokens=1200`): cláusulas que cabem, viram um chunk só; cláusulas maiores descem um nível na hierarquia (parágrafos, incisos) até caber. Isso significa que **o corte nunca é arbitrário** — sempre acontece numa fronteira que já existe no documento (fim de um parágrafo, fim de um inciso), nunca no meio de uma frase.

**Pergunta de entrevista provável:** *"Como você lida com documentos que não têm essa estrutura?"* — fallback: se o detector de hierarquia não encontra nenhum cabeçalho, o chunker cai para um splitter por parágrafo com o mesmo teto de tokens. A taxa de fallback é medida e reportada (`ChunkingReport.fallback_chunks`) — no corpus real do projeto, ficou em 0%, mas o código não assume isso, ele *mede*.

### 3.6 Grounding e mitigação de alucinação (implementado na Fase 4)

"Grounding" é o termo técnico para forçar a resposta do LLM a se apoiar só no contexto fornecido. As quatro técnicas que o projeto usa, em ordem de execução no pipeline:

1. **Prompt fechado** — instrução explícita para responder só com o que está nas fontes numeradas.
2. **Saída estruturada (JSON via tool use)** — a citação vira um campo tipado, não texto livre. Isso é o que torna a verificação da próxima etapa possível.
3. **Verificação programática da citação** — o texto que o LLM diz ter citado é comparado, por *fuzzy matching* (similaridade de string, não semântica), contra o texto real do chunk. Se não bate, a citação foi inventada.
4. **Abstenção por limiar** — se a busca não encontrar nada suficientemente relevante (score do reranker abaixo de um corte), o LLM nem é chamado. Isso evita a "alucinação por obrigação" — um LLM instruído a sempre responder vai inventar algo quando não tem contexto bom.

**Isso é a diferença entre RAG ingênuo e RAG com engenharia de confiabilidade.** A maioria dos tutoriais para no passo 1. O diferencial deste projeto está nos passos 3 e 4.

### 3.7 Avaliação de RAG — por que "parece bom" não é uma métrica

Sem medir, você não sabe se uma mudança (trocar o embedder, ajustar o chunking) melhorou ou piorou o sistema — só *parece* melhor no exemplo que você testou manualmente. As métricas padrão da área (framework RAGAS):

- **Faithfulness** — a resposta é sustentada pelo contexto recuperado, ou o modelo disse algo que não está lá?
- **Context precision / recall** — dos chunks recuperados, quantos eram relevantes (precision)? Dos chunks relevantes que existem no corpus, quantos foram recuperados (recall)?
- **Answer relevancy** — a resposta de fato responde à pergunta feita?

Este projeto soma duas métricas próprias, mais direto ao ponto do caso de uso: **acurácia de citação** (a cláusula citada é *de fato* a cláusula correta, não só parecida) e **taxa de abstenção correta** (perguntas sem resposta no corpus realmente resultam em abstenção, não em invenção).

---

## 4. Diário de desenvolvimento — fase a fase

### Fase 0 — Fundação

**O que foi construído:** estrutura do repositório, `settings.py` (Pydantic Settings), API FastAPI mínima com `/health`.

**Decisão que vale explicar:** toda constante ajustável do sistema (top-k, limiares, nomes de modelo) mora numa única classe `Settings`, carregada do `.env`. Nenhum outro módulo lê variável de ambiente diretamente ou hardcoda um número. Isso não é só organização — é o que torna a Fase 7 possível: para comparar três configurações do pipeline, basta trocar o `.env` e rodar de novo, sem tocar em código.

**Termo técnico para isso:** *configuração externalizada* — um dos princípios do [12-Factor App](https://12factor.net/config). Se alguém perguntar "como você gerencia configuração", essa é a resposta.

### Fase 1 — Corpus

**O que foi construído:** `scripts/check_corpus.py`, que tria PDFs por densidade de caracteres por página (PDFs escaneados sem OCR têm quase zero texto extraível) e gera um `manifest.csv`.

**Decisão que vale explicar:** rejeitar cedo. Um PDF sem camada de texto, se entrasse no pipeline, geraria um chunk vazio ou com lixo — poluindo o índice silenciosamente, sem erro nenhum, e você só descobriria isso na Fase 7 tentando entender por que uma pergunta óbvia não encontra resposta. **Fail fast e com uma métrica visível** (`chars_per_page`) é mais barato que debugar um sintoma três fases depois.

Corpus final: **20 documentos, 274 páginas**, contratos administrativos brasileiros reais (PNCP), com `doc_type` classificado manualmente.

### Fase 2 — Chunking estrutural

**O que foi construído:** `parser.py` (PDF → texto com mapa de página) e `chunker.py` (texto → chunks por cláusula).

**A invariante mais importante do projeto inteiro está aqui:** o campo `text` de todo chunk é sempre uma **fatia contígua e literal** do documento original — nunca uma reconstrução ou concatenação. Isso é o que torna a verificação de citação da Fase 4 confiável: se o texto citado bate com o chunk, ele necessariamente existe, palavra por palavra, no PDF. Há um teste (`test_texto_do_chunk_e_fatia_literal_do_documento`) que existe só para proteger essa garantia.

**Um bug real que encontramos, e por que ele é bom material de entrevista:**
Quando uma cláusula é grande demais e tem um `§` no meio, o chunker divide em duas partes: o texto antes do primeiro parágrafo (`head_text`) e os parágrafos seguintes. Os parágrafos passavam por um divisor que respeitava o teto de tokens; o `head_text` **não** — ia direto como um chunk único. Resultado: se essa abertura fosse longa (aconteceu em `Contrato.pdf`, real, do corpus), o chunk saía com 1518 tokens contra um teto de 1200.

O motivo de eu contar isso: **é exatamente o tipo de bug que só aparece com dado real**, nunca no exemplo sintético que você testa primeiro. Os 17 testes que eu tinha antes de rodar contra o corpus real passavam todos — porque nenhum deles tinha uma cláusula com essa forma específica (abertura longa + parágrafo curto). Depois de corrigir, escrevi um teste de regressão reproduzindo exatamente essa forma, não só "meça se o número deu certo agora" — **um teste de regressão bom reproduz o formato do bug, não só o sintoma**.

**Se alguém perguntar "me conta um bug difícil que você resolveu":** este é um bom exemplo — mostra que você testa com dado real (não só sintético), que sabe ler um relatório agregado (`chunks acima do teto: 1`) e rastrear até a causa específica, e que fecha o ciclo com um teste de regressão.

Resultado no corpus real: **334 chunks, 100% com hierarquia detectada, 0% em fallback, 0 chunks acima do teto.**

### Fase 3 — Índice híbrido

**O que foi construído:** `indexer.py` (constrói os dois índices), `hybrid.py` (busca + fusão RRF), `rerank.py` (cross-encoder), `cli.py`.

**A decisão de design mais importante aqui:** o modelo de embedding e o modelo de reranking nunca são chamados diretamente dentro da lógica de indexação/busca — são **parâmetros injetáveis** (`Embedder`, `Scorer`), com uma função `get_default_*()` separada que só importa a biblioteca pesada (`sentence-transformers`) e baixa o modelo quando *de fato* chamada.

Por que isso importa na prática, além de "boas práticas": permitiu testar a integração real com ChromaDB e BM25 (nenhum dos dois é mockado — são as bibliotecas de verdade rodando) usando um embedder falso e determinístico, sem precisar de PyTorch nem baixar 2GB de modelo. 33 testes passam em menos de 2 segundos. Isso é **injeção de dependência** aplicada a um problema de ML — o mesmo princípio que você usaria para trocar um banco de dados por um mock em teste de backend, aplicado a um modelo de embedding.

**Se alguém perguntar "como você testa código que depende de modelos de ML pesados":** essa é a resposta — isole o modelo atrás de uma interface simples (função que recebe texto, devolve vetor/score), injete a implementação real em produção e uma falsa em teste. Você não está testando se o BGE-M3 é bom (isso é trabalho da Fase 7, com métricas), está testando se o *seu código* usa o resultado dele corretamente.

**Duas armadilhas técnicas que vale saber explicar:**

- **Por que a distância do Chroma, e não a similaridade, decide a ordem:** o ChromaDB por padrão devolve distância (menor = mais similar), não similaridade. O código nunca inverte isso manualmente — só ordena ascendente, que é o que a própria API já devolve.
- **Por que o BM25 descarta score zero:** quando uma pergunta não compartilha nenhum termo com um chunk, o BM25 dá score 0 — isso não é "pouco relevante", é *ausência de sinal*. Incluir isso na fusão RRF adicionaria ruído, não informação — por isso `lexical_search` filtra `score > 0` antes de devolver.

**Um segundo bug real, encontrado na comparação final de 10 consultas — e por que ele é ainda melhor material de entrevista que o do `head_text`:**

Rodando `scripts/compare_retrieval.py` contra o corpus real, dois `clause_path` apareceram quebrados: `'art. 137 d'` (consulta "multa moratória") e `'art. 94 d'` (consulta "art. 78"). Não eram cláusulas nem artigos do contrato — eram **citações de lei que quebraram de linha no meio da frase**, por causa do jeito como o PyMuPDF extrai o texto de um PDF com reflow (ex.: "...conforme o disposto no\nart. 137 da Lei nº 14.133, de 2021..." — a quebra de linha caiu bem depois de "no", deixando "art. 137" sozinho no início da linha seguinte).

O detector de cabeçalhos (`find_headings`) processa o texto **linha a linha**, e o padrão `"artigo"` (`r"^\s*(Art(?:igo)?\.?\s*\d+..."`) não tinha como saber, olhando só aquela linha, se "art. 137" era o início de um artigo de verdade ou o resto de uma frase que só por acaso começa uma nova linha ali. Pior: o padrão é case-insensitive e aceita opcionalmente uma letra logo depois do número (para capturar "Art. 137-A"), então "art. 137 **da** Lei..." virava marcador `"art. 137 d"` — o "d" de "da" sendo capturado como se fosse o sufixo de um artigo.

**A correção não foi mudar o regex — foi dar contexto ao detector.** Um cabeçalho de artigo de verdade sempre abre um parágrafo novo: vem depois de uma linha em branco, ou depois do ponto final da frase anterior. Uma citação quebrada, ao contrário, continua uma frase em andamento — a linha anterior termina sem pontuação de fechamento (ex.: "...disposto no"). `find_headings` passou a rastrear a linha anterior e só aceitar `"artigo"` como cabeçalho quando ela está vazia, termina em pontuação de fim de frase, ou é a primeira linha do documento.

**Por que isso é bom material de entrevista, mais até que o bug do `head_text`:** o do `head_text` era um erro de lógica dentro do próprio código (um caminho que não passava pelo splitter). Este é diferente — é a classe de bug mais comum e mais sutil em pipelines de NLP sobre PDF real: **o parser está funcionando exatamente como programado, mas o pressuposto por trás do padrão (regex ancorado numa linha isolada) não é suficiente para distinguir dois casos que são idênticos na superfície e diferentes no contexto.** Regex sobre texto natural quase sempre precisa de uma noção de contexto (linha anterior, posição no documento) além do "casa ou não casa" — e é exatamente isso que separa um chunker que funciona no exemplo sintético de um que aguenta 20 contratos reais de fontes diferentes.

**Teste de regressão** (`tests/test_chunker.py`): reproduz o formato exato do bug (uma citação de lei quebrando de linha) e confirma que ela não vira cabeçalho, mais dois testes confirmando que um `Art. N` genuíno — depois de linha em branco, ou como primeira linha do documento — continua sendo detectado normalmente.

**Evidência real, rodando contra o corpus de 20 contratos (`python -m rag.retrieval.cli`):**

| Query | Rerank score (top-5) | Leitura |
|---|---|---|
| "quais as penalidades por atraso na entrega" | 0,83 – 0,97 | Excelente. Todos os 5 resultados são de fato cláusulas de sanção/penalidade. |
| "art. 78" | 0,015 – 0,18 | Fraco e uniforme — nenhum resultado se destaca dos outros. |

**Por que a segunda deu ruim, e por que isso é um achado, não um bug:** "art. 78" sozinho é uma referência de lei genérica, citada em contratos administrativos por motivos completamente diferentes (sanção, rescisão, obrigação) — e o corpus mistura contratos regidos pela Lei 8.666/1993 e pela Lei 14.133/2021, então **o mesmo número de artigo aponta para dispositivos diferentes** dependendo de qual lei o contrato cita. Uma query de duas palavras, sem verbo nem contexto, não dá ao cross-encoder informação suficiente pra diferenciar — daí os scores baixos e homogêneos, em vez de um vencedor claro.

**Isso é dado real para calibrar `MIN_RERANK_SCORE`, que na Fase 4 decide a abstenção:** query bem resolvida ficou acima de 0,83; query ambígua ficou abaixo de 0,18. O valor inicial no `.env.example` (0,35) senta exatamente no meio desses dois extremos — plausível como primeira aproximação, mas a Fase 7 vai calibrar isso de verdade contra o golden set, não contra duas amostras.

**Pergunta de entrevista que este resultado responde direto:** *"Busca léxica não deveria ser perfeita pra citação exata tipo número de artigo?"* — não quando o termo é ambíguo *no domínio*, não na busca. BM25 encontra todo chunk que contém "art. 78" (é literal, funciona bem); o gargalo é que "art. 78" aparece em contextos semânticos diferentes no corpus, e nem o cross-encoder consegue desambiguar sem mais contexto na pergunta. A lição prática: buscas por citação numérica funcionam melhor quando a pergunta inclui o assunto ("art. 78 sobre rescisão"), não o número isolado.

**Fechamento do critério de aceite — comparação completa em 10 consultas (`docs/comparacao_fase3.md`):**

Rodando `scripts/compare_retrieval.py` contra o corpus inteiro (334 chunks, antes da correção do chunker), o padrão em 10 consultas foi: híbrido+rerank empata ou melhora sobre denso-puro em 8 das 10; um empate qualificado (score baixo nas últimas posições, mas o reranker já sinaliza isso); e um caso real de piora (consulta "hipóteses de rescisão contratual" — a fusão RRF, que soma posição em vez de avaliar conteúdo, deixou entrar na 5ª posição uma cláusula de *penalidades* de outro contrato, só por sobreposição de vocabulário jurídico comum a rescisão e sanção). Isso não é um bug — é o comportamento esperado de um sistema desenhado para recall alto na entrada (a busca híbrida) e precisão crescente a cada estágio (reranking, e na Fase 4, o limiar de abstenção): o próprio reranker deu a esse resultado ruim um score bem mais baixo (0,797) que aos quatro corretos (0,981–0,992), só não baixo o bastante para cair fora de um `top_k=5` fixo.

**Confirmação em dado real, depois da correção do bug do chunker:** reindexando o corpus inteiro com o `find_headings()` corrigido, o número total de chunks caiu de 334 para 326 — os 8 chunks a menos são exatamente os cabeçalhos fantasma que citações de lei quebradas geravam. Rerodando a mesma comparação de 10 consultas, `'art. 137 d'` e `'art. 94 d'` desapareceram por completo dos resultados (consultas 6 e 10), a consulta de rescisão deixou de trazer a cláusula de penalidades errada, e o padrão geral melhorou para 9 de 10 empatando ou favorecendo híbrido+rerank. A consulta ambígua de controle ("art. 78") continua com todos os scores abaixo de 0,18 mesmo depois da correção — confirmando que aquele resultado ruim é ambiguidade real do domínio (duas leis diferentes citando o mesmo número de artigo), não um efeito colateral do bug de chunking.

**Por que vale registrar essa confirmação, e não só "corrigi e segui":** um teste de regressão sintético prova que a lógica está certa; rodar contra o corpus real de novo prova que o bug realmente desapareceu do produto final, não só da unidade de código isolada. São duas garantias diferentes, e uma entrevista técnica costuma testar se você sabe a diferença entre elas.

**A conclusão que mais importa para a Fase 4 não é "qual modo ganhou", é a distribuição do score do reranker:** nas consultas bem resolvidas, o top-3 fica quase sempre acima de 0,7; no ruído de cauda e na consulta ambígua de controle, cai abaixo de 0,35 — muitas vezes abaixo de 0,05. O valor inicial de `MIN_RERANK_SCORE` no `.env.example` (0,35) cai bem na fronteira entre esses dois grupos, o que valida a escolha como ponto de partida razoável (a calibração final e rigorosa é trabalho da Fase 7, contra um golden set, não contra 10 amostras).

**Se alguém perguntar "como você decide o limiar de abstenção":** não adivinhando um número — rodando o pipeline real contra dados reais e observando onde os scores de resultados corretos e incorretos se separam. Esse é exatamente o processo documentado aqui: os números de `MIN_RERANK_SCORE` não vieram de uma referência externa, vieram da distribuição observada nesta comparação.

### Fase 4 — Geração com citação verificada

**O que foi construído:** `src/rag/generation/` — cinco módulos, cada um com uma responsabilidade só:

| Módulo | Responsabilidade |
|---|---|
| `prompt.py` | Monta o contexto numerado (`[1]`, `[2]`...) e o prompt fechado. Só texto — não decide nada. |
| `schema.py` | O contrato de saída (`emitir_resposta`), como JSON Schema de uma tool use. |
| `client.py` | A única função que fala com a Claude API — injetável, mesmo padrão do embedder/scorer. |
| `verify.py` | Verificação fuzzy da citação: o `quote` que o modelo alega ter tirado da fonte bate, literalmente (com tolerância), com o texto do chunk? |
| `answer.py` | Orquestra tudo: recupera (Fase 3) → decide se aborta antes de chamar o modelo → gera → verifica → reprocessa uma vez se falhar → decide a resposta final. |

**A decisão de design que mais vale explicar: tool use em vez de "peça JSON no prompt".** Um prompt pedindo "responda em JSON" ainda pode voltar com texto antes/depois do JSON, aspas erradas, ou um campo faltando — parsing frágil. `tool_choice` forçando uma ferramenta faz a própria Claude API validar a FORMA da resposta contra um JSON Schema antes dela chegar ao meu código. Isso não elimina alucinação de conteúdo (o modelo ainda pode citar a fonte errada) — só elimina alucinação de formato. São dois problemas diferentes, resolvidos em dois lugares diferentes: `schema.py` (formato) e `verify.py` (conteúdo).

**As quatro defesas, na ordem em que `answer_question()` as aplica:**

1. **Abstenção por limiar, antes de qualquer chamada à API.** Se o melhor `rerank_score` da Fase 3 já vem abaixo de `MIN_RERANK_SCORE`, a Claude API nem é chamada — custo zero, e sem chance de o modelo "tentar ajudar" com contexto ruim. Isso também é o motivo de a consulta ambígua de controle da Fase 3 ("art. 78", scores < 0,18) já ser resolvida aqui, na Fase 4: ela nunca chega a virar uma chamada de API.
2. **Contexto numerado + prompt fechado.** O modelo só vê os chunks que passaram pelo reranking, numerados, e a instrução deixa claro que não há conhecimento fora dali.
3. **Verificação literal da citação.** Depois que a resposta volta, cada `quote` é comparado por fuzzy matching (`rapidfuzz.fuzz.partial_ratio`) contra o texto do chunk que ele alega citar. Abaixo de `CITATION_MATCH_THRESHOLD` (0,90), a citação é rejeitada.
4. **Um reprocessamento, depois abstenção.** Se alguma citação falha, o sistema reformula a pergunta ao modelo apontando exatamente qual citação não bateu e pedindo para copiar literalmente (ou remover a afirmação). Se falhar de novo, o sistema prefere abster a devolver uma resposta com zero citações confirmadas — a mesma filosofia da abstenção por limiar: melhor "não sei" do que uma alegação sem lastro.

**Um trade-off de design que vale saber defender:** o reprocessamento (defesa 4) reenvia o prompt inteiro de novo, com uma instrução de correção anexada — não usa o protocolo formal de `tool_result` multi-turno da Claude API (que economizaria reenviar o contexto). Escolhi a versão mais simples de propósito: cada chamada ao `model_caller` é uma função pura e sem estado (mesma assinatura sempre), o que faz o `model_caller` falso dos testes ser trivial de escrever — sem simular IDs de `tool_use` nem histórico de conversa. Custo: mais tokens gastos no reprocessamento do que o estritamente necessário. Numa versão de produção com volume alto, valeria trocar por `tool_result` de verdade; num portfólio, a simplicidade testável ganha.

**Por que a verificação nunca lança exceção, só devolve `verified=False`:** uma citação errada não é um bug do meu código, é um dado sobre a resposta do modelo — e o chamador (`answer.py`) precisa decidir o que fazer com isso (reprocessar, descartar, abster), não ser interrompido por uma exceção no meio do fluxo.

**Testes:** 15 testes novos (`tests/test_generation.py`), toda a orquestração coberta com um `model_caller` falso e determinístico — nenhum gasta uma chamada de API real. Casos cobertos: abstenção por score baixo sem chamar o modelo nem uma vez; resposta aceita de primeira; reprocessamento bem-sucedido depois de uma citação ruim; abstenção depois de reprocessar e ainda falhar; e confiança na abstenção que o próprio modelo declara (`insufficient_context=true`), sem tentar reprocessar isso.

**Um bug de teste que só apareceu na SUA máquina, não no ambiente onde eu desenvolvo — e que ensina uma lição de Python que vai além deste projeto:**

Rodando `pytest -q` na sua máquina, um teste que sempre passou aqui falhou: `test_build_dense_index_e_dense_search_encontram_texto_similar` (Fase 3), esperando que o chunk `c1` viesse em primeiro lugar numa busca e recebendo `c2`. A lógica testada (`dense_search`) estava certa — o problema estava no `fake_embedder` que os testes da Fase 3 e 4 usam para simular embeddings sem baixar um modelo de ML de verdade.

O `fake_embedder` transforma cada palavra num "balde" de um vetor pequeno usando `hash(palavra) % 32`. O comentário no código dizia, com todas as letras, "determinístico" — e isso é **falso** para o `hash()` embutido do Python: desde o Python 3.3, o hash de uma `str` é **aleatorizado a cada novo processo do interpretador** (controlado por uma variável de ambiente chamada `PYTHONHASHSEED`), de propósito — é uma proteção de segurança contra um ataque chamado *hash flooding* (alguém manda milhares de chaves desenhadas para colidir no mesmo balde de uma hash table, degradando um servidor O(1) para O(n²)). O efeito colateral: `hash("multa")` dá um número na sua máquina e um número *diferente* no ambiente onde eu testo — o balde de cada palavra muda, o vetor final muda, e um teste que compara "qual chunk vem em 1º lugar" pode dar uma resposta diferente dependendo unicamente de qual processo rodou primeiro, sem nenhuma mudança de código.

**Por que este bug é um dos melhores para contar numa entrevista, mais até que os dois de chunking:** os bugs do `head_text` e do `find_headings` eram erros de lógica no código de PRODUÇÃO. Este é mais sutil — o código de teste em si nunca teve um erro de lógica, e "passou no meu ambiente" não prova nada aqui, porque o resultado depende de um fator invisível (a seed de hash daquele processo específico) que muda entre execuções sem qualquer edição. É a categoria de bug mais cara que existe: passa na sua máquina de desenvolvimento, passa no CI, e falha (ou pior — silenciosamente dá um resultado diferente sem falhar) só na máquina de outra pessoa, sem ninguém ter tocado em uma linha.

**A correção:** trocar `hash()` por `zlib.crc32()`, que não é aleatorizado — mesma string, mesmo número, sempre, em qualquer processo, em qualquer máquina. Antes de considerar corrigido, rodei a suíte inteira sob cinco valores diferentes de `PYTHONHASHSEED` — não bastava "passou uma vez", porque essa vez também podia ter sido sorte.

**Pergunta de entrevista que este bug responde direto:** *"Por que um teste passaria numa máquina e falharia em outra sem nenhuma mudança de código?"* — não-determinismo em algo que o código assume ser determinístico. Os suspeitos clássicos: `hash()` de string (este caso), ordem de iteração de `set`/`dict` em versões antigas do Python, `datetime.now()` sem congelar o relógio, ou concorrência sem lock. A lição prática, além deste projeto: nunca use o `hash()` embutido do Python para nada que precisa ser reproduzível entre execuções — ele existe para tabelas hash internas do interpretador, não para gerar identificadores estáveis. `hashlib` ou `zlib.crc32` são as ferramentas certas para isso.

**Pergunta de entrevista provável:** *"Como você testa um sistema que depende de uma API de LLM paga, sem gastar dinheiro a cada `pytest`?"* — o mesmo princípio de injeção de dependência das Fases 2 e 3, aplicado de novo: a chamada à API vira uma função (`ModelCaller`) isolada atrás de uma interface simples (`(system, mensagens) -> dict`), injetável. O teste não valida se a Claude API é boa — valida se o MEU código reage certo a diferentes respostas possíveis dela (boa, com citação inventada, admitindo que não sabe).

**Fechamento do critério de aceite — 10/10 abstenções, confirmado contra a Claude API real:** `scripts/test_abstention.py` rodou 10 perguntas propositalmente fora do domínio do corpus (capital da Mongólia, cotação de bitcoin, receita de bolo, fórmula de Bhaskara...) e todas resultaram em abstenção — nenhuma resposta inventada. Resultado completo em `docs/abstencao_fase4.md`.

**O achado mais interessante deste resultado não é o "10/10" em si, é ONDE a abstenção aconteceu.** Todos os scores de reranking ficaram entre 0,000 e 0,004 — muito abaixo do `MIN_RERANK_SCORE=0,35` — o que significa que **nenhuma das 10 perguntas chegou a chamar a Claude API**. A abstenção veio inteiramente da defesa 1 (o corte por score, na camada de recuperação), não das defesas 2–4 (prompt fechado, verificação de citação, reprocessamento), que dependem de o modelo ser chamado.

**Por que isso importa, e não é só "ainda melhor, custou zero":** este teste prova que o sistema não inventa resposta para o caso mais óbvio (pergunta sem nenhuma relação com o domínio) — mas não prova ainda que as defesas 2–4 funcionam, porque elas nunca tiveram a chance de entrar em ação. O teste que vai exercitar isso de verdade precisa de perguntas na FRONTEIRA: relacionadas ao domínio de contratos públicos, plausíveis o bastante para a busca devolver um score moderado, mas sem resposta *naquele* corpus específico (ex.: "qual a multa por atraso num contrato de fornecimento de merenda escolar", se nenhum contrato do corpus for sobre merenda escolar). É exatamente esse tipo de pergunta "quase certa" que compõe as 10 perguntas sem resposta do golden set de 50 da Fase 7 — de propósito, mais difíceis que estas 10.

**Pergunta de entrevista que este resultado prepara bem:** *"Seu sistema de abstenção realmente testa o LLM, ou só a busca?"* — depende de que teste você está olhando. Este teste específico (Fase 4) provou a camada de recuperação; a camada de geração (verificação de citação, reprocessamento) tem sua própria cobertura nos 15 testes com `model_caller` falso (que simulam citação inventada de propósito), mas ainda não foi provada contra casos reais e ambíguos — isso é trabalho da Fase 7. Saber apontar essa diferença — "o que este teste prova" vs. "o que ele não prova" — é exatamente o tipo de honestidade técnica que diferencia alguém que testou de verdade de alguém que só rodou um script e viu "10/10" passar.

### Fase 5 — API completa

Até aqui, cada fase tinha seu próprio ponto de entrada (`compare_retrieval.py`, `generation/cli.py`, `test_abstention.py`). A Fase 5 amarra tudo atrás de uma API HTTP única, em `src/rag/api/`, com um princípio simples: **`main.py` só roteia.** Cada endpoint é uma chamada de uma linha para um módulo específico — a lógica de verdade mora em `documents.py`, `metrics.py`, `query_service.py`, `query_stream.py`, `ingest_job.py`. Isso não é só estética: significa que cada peça é testável isoladamente, sem subir um servidor FastAPI, exatamente como o resto do projeto trata `Embedder`/`Scorer`/`ModelCaller`.

| Módulo | Endpoint(s) | O que faz |
|---|---|---|
| `schemas.py` | (todos) | Modelos Pydantic de fronteira — validação de entrada e forma de saída, num só lugar |
| `documents.py` | `GET /documents` | Une manifest (Fase 1) + contagem de chunks (Fase 2) por documento |
| `metrics.py` | `GET /metrics` | Agrega `data/logs/queries.jsonl` (Fase 4) — nenhuma métrica nova instrumentada |
| `query_service.py` | `POST /query` | Chama `answer_question()` (Fase 4) e converte pro schema de resposta |
| `query_stream.py` | `POST /query/stream` | A mesma pergunta, como eventos de progresso via SSE |
| `ingest_job.py` | `POST /ingest`, `GET /ingest/status` | Dispara reindexação em background, com estado em memória |

**Duas ideias que já apareceram antes no projeto, aplicadas de novo em contextos novos:**

1. **DI estendida para caminhos de arquivo.** `list_documents()` e `compute_metrics()` aceitam `chunks_file`/`manifest_file`/`log_file` opcionais, com default em `settings.*`. Mesmo raciocínio de `Embedder`/`Scorer`/`ModelCaller` (Fases 2–4): um teste passa um `tmp_path` e nunca toca `data/` real, sem precisar de mocks de sistema de arquivo.
2. **Um log estruturado escrito uma vez, lido duas.** `/metrics` não calcula nada em tempo real — só lê e agrega `queries.jsonl`, que a Fase 4 já escreve a cada consulta. Um endpoint inteiro "de graça" por causa de uma decisão de logging tomada duas fases antes. A Fase 7 vai reler o mesmo arquivo para métricas bem mais ricas (faithfulness, acurácia de citação por categoria de pergunta) — de novo, sem re-instrumentar nada.

**"SSE" honesto: progresso de pipeline, não streaming de token — e por quê.** `POST /query/stream` existe porque uma pergunta que passa por busca + reranking + chamada de API (às vezes com reprocessamento) pode levar alguns segundos, e uma UI sem nenhum feedback nesse intervalo parece travada. Mas a saída da Fase 4 é forçada via `tool_choice` (tool use) — a Claude API só entrega o bloco `tool_use` **completo e validado**, não incrementalmente; não existe um "meio de resposta" coerente para transmitir token a token. Um endpoint que fingisse streaming de token aqui estaria mentindo sobre o que a arquitetura permite. A solução honesta: eventos de **etapa** ("buscando fontes" → "gerando resposta" → "verificando citações" → eventualmente "reprocessando citação rejeitada" → "concluído"), que já é o suficiente para tirar uma UI de um spinner genérico.

**O mecanismo por trás disso é o ponto técnico mais interessante desta fase:** `answer_question()` é síncrona e bloqueante (embeddings, reranking e a chamada de rede à Claude API não são `async`). Só que o endpoint precisa ser um gerador assíncrono (`AsyncIterator[str]`) para o FastAPI transmitir SSE incrementalmente. A ponte: `answer_question` roda numa `threading.Thread` separada; o hook `on_stage` (chamado de dentro dessa thread) empurra eventos numa `queue.Queue` via `loop.call_soon_threadsafe(events.put_nowait, ...)` — **não** dá pra chamar `asyncio.Queue.put` direto de outra thread, não é thread-safe, por isso o `call_soon_threadsafe` fazendo a travessia de volta pro event loop. O gerador async, por sua vez, fica em loop chamando `await loop.run_in_executor(None, events.get)` — que bloqueia numa thread do executor, não no event loop — até receber um sentinela `None` que marca o fim.

**Pergunta de entrevista que este mecanismo prepara bem:** *"Como você expõe uma função síncrona e bloqueante como um streaming assíncrono?"* — não trocando a função de lugar (reescrever `answer_question` em `async` complicaria toda a Fase 4 para embeddings/reranking que são inerentemente síncronos), mas isolando-a numa thread e usando uma fila como ponte de mão única, com `call_soon_threadsafe` no lado de escrita (thread) e `run_in_executor` no lado de leitura (event loop). É o mesmo padrão usado por qualquer integração de biblioteca síncrona dentro de um servidor ASGI.

**O trade-off do job de ingestão: estado em memória, de propósito.** `POST /ingest` devolve 202 imediatamente e dispara `run_ingest_job()` via `BackgroundTasks` do FastAPI; o progresso fica num `_IngestState` — um dataclass simples, module-level, protegido por um `threading.Lock` — que `GET /ingest/status` lê. Limitação real e aceita: esse estado some se o processo reiniciar, e não é compartilhado se alguém subir a API com múltiplos workers uvicorn (`--workers 4`, por exemplo — cada worker teria seu próprio singleton). Para uma API de instância única (o caso deste portfólio), é suficiente e muito mais simples que a alternativa correta em produção, que seria um job store externo (Redis, uma tabela no banco) compartilhado entre processos. Vale nomear essa troca explicitamente numa entrevista: "eu sei que isso não escala para múltiplos workers, e sei qual seria o próximo passo" é uma resposta mais forte que fingir que a limitação não existe.

**Por que a união (não interseção) de manifest e chunks em `/documents`:** um documento pode estar no manifest e ainda não ter sido chunkado (`chunk_count=0` — sinal de que algo falhou no parsing), ou ter chunks sem entrada no manifest (esquecimento ao preencher a Fase 1). Fazer interseção esconderia os dois casos, que são justamente os que alguém rodando due diligence sobre o próprio pipeline (não sobre os contratos) precisa enxergar.

**Testes:** 14 testes novos (`tests/test_api.py`), usando `TestClient` do FastAPI para os endpoints e chamada direta de função para `list_documents()`/`compute_metrics()`/`run_ingest_job()` como funções puras. Nenhum toca a Claude API, o BGE-M3 ou o reranker de verdade — `/query` e `/query/stream` têm `answer_question` trocado por um `model_caller` falso via monkeypatch; `/ingest` tem `run_chunking`/`load_chunks`/`build_index` trocados da mesma forma. 62 testes no total (suíte inteira), todos passando; `ruff check` limpo.

**Um teste que nasceu quebrado, e por que vale a pena registrar o próprio erro:** o primeiro rascunho de `tests/test_api.py` tinha um teste (`test_compute_metrics_sem_log`) que era código morto — chamava `compute_metrics(log_file=None)` e não fazia nenhuma asserção sobre o resultado, sobrando de uma tentativa de testar "o que acontece quando cai no caminho padrão de `settings.query_log_file`" sem realmente isolar esse caminho. Corrigido usando `monkeypatch.setattr(type(settings), "query_log_file", property(...))` para sobrescrever a property calculada com um caminho de `tmp_path` controlado (`settings.query_log_file` é uma `@property` derivada de `ROOT_DIR`, sem setter — não dá pra fazer `monkeypatch.setattr(settings, "query_log_file", ...)` direto numa instância, só na classe). Pequeno, mas é o tipo de coisa que passa despercebida numa revisão rápida: um teste sem `assert` sempre "passa".

**Fechamento do critério de aceite:** confirmado na sua máquina — suíte inteira verde (`pytest -q`, 62 testes) e `/docs` (Swagger UI) navegável com `uvicorn rag.api.main:app --reload`, com os endpoints exercitados contra o corpus de 20 contratos e a Claude API reais.

### Fase 6 — Interface

**O que foi construído:** `ui/app.py` reescrito por completo — dois painéis lado a lado (`st.columns`): conversa à esquerda, fontes à direita. Junto, três mudanças no backend que a UI precisou puxar:

1. `QueryResponse` ganhou um campo `sources` novo (`schemas.SourceChunk`), populado em `query_service.to_query_response()` a partir de `AnswerResult.sources` — o chunk inteiro por trás de cada `source_id`, não só o `quote` curto que já existia em `citations`.
2. `search()` (Fase 3, `hybrid.py`) ganhou um parâmetro `document_filter`, usado pelo seletor de documentos da UI.
3. `answer_question()`, `run_query()` e `stream_query()` passaram a aceitar e repassar esse filtro até `search()`.

**A decisão de design que mais vale explicar: por que "texto-fonte" precisou de um campo novo na API, em vez de reaproveitar `citations`.** `citations[].quote` é o trecho curto que o MODELO alegou ter citado — é o dado que a Fase 4 verifica. Mas "clicar numa citação exibe o texto-fonte com documento e página" (o critério de aceite) pede o chunk inteiro ao redor daquele trecho, pro usuário conferir o contexto, não só a frase isolada. São dois dados diferentes, com dois donos diferentes: `citations` vem do que o LLM disse; `sources` vem do que a Fase 3 recuperou. Misturar os dois num campo só teria sido mais simples de escrever e mais confuso de usar.

**O seletor de documentos é um filtro de busca de verdade, não um filtro cosmético — e por que isso quase deu errado.** A implementação ingênua seria: rodar `search()` normalmente (top_k=20 do corpus inteiro) e, na UI, só *esconder* da tela os resultados de documentos não selecionados. Isso pareceria funcionar em qualquer demonstração rápida, mas é enganoso: se os 20 candidatos mais bem rankeados do corpus INTEIRO não incluírem nenhum chunk do documento escolhido, o usuário veria uma tela vazia mesmo que aquele documento tivesse a resposta perfeita — o filtro nunca teve a chance de agir, porque o corte de `top_k` já tinha acontecido antes. A implementação certa filtra ANTES do corte: quando `document_filter` está ativo, `search()` busca `top_k × 10` candidatos (denso e léxico) antes de aplicar o filtro, e só corta em `top_k` depois. Escrevi um teste que prova exatamente essa diferença: `test_search_com_document_filter_exclui_o_documento_certo_mesmo_sendo_o_mais_relevante` — filtra para um documento que NÃO contém o resultado mais óbvio do corpus inteiro, e confirma que ele de fato não aparece (o filtro é real), enquanto outro teste confirma que o documento CERTO ainda aparece mesmo sem estar entre os top-*k* do ranking geral (o filtro não é vazio).

**Pergunta de entrevista que esse detalhe prepara bem:** *"Como você teria implementado um filtro por documento sem pensar demais?"* — a resposta ingênua (`resultados = [r for r in candidatos_ja_cortados if r.document in filtro]`) é um bug esperando pra acontecer: funciona nos primeiros testes manuais (o documento testado calha de estar no top-20) e falha silenciosamente em produção assim que alguém filtra para um documento menos óbvio. A lição generaliza além deste projeto: filtro pós-paginação/pós-corte é uma armadilha clássica sempre que o corte acontece ANTES do filtro na pipeline.

**Progresso real via SSE consumido do lado do cliente — fechando o círculo da Fase 5.** A Fase 5 construiu `POST /query/stream` e documentou, com honestidade, que "SSE" ali significa eventos de etapa, não token a token. A Fase 6 é onde isso vira valor visível: `ui/app.py` usa `httpx.stream(...)` para consumir esse SSE e atualiza um `st.status` a cada evento (`"buscando fontes"` → `"gerando resposta"` → `"verificando citações"`), em vez de mostrar um spinner genérico enquanto a pergunta (que pode levar vários segundos, com reranking + chamada de API + eventual reprocessamento) está em voo.

**Um detalhe de parsing que merece registro, porque é o tipo de bug que só aparece com dado real:** o parser SSE do lado do cliente divide cada linha em `evento:conteúdo` usando `line.split(":", 1)` — o `1` (maxsplit) é essencial. Sem ele, uma linha como `data: {"answer": "prazo: 12 meses", ...}` quebraria no PRIMEIRO `:` depois de "data", mas o conteúdo JSON também tem `:` dentro (entre chave e valor) — sem limitar a um corte só, `str.split(":")` sem argumento fatiaria o JSON ao meio. Testado isoladamente antes de confiar nele: uma consulta com `"data": {"answer": "ok: sim"}` (dois-pontos dentro do próprio conteúdo) precisa continuar parseando certo.

**Abstenção visualmente distinta — a parte mais simples de implementar, e a mais fácil de esquecer.** Avatar diferente (`🚫` vs `⚖️`) e `st.warning(...)` (fundo colorido, ícone de alerta do próprio Streamlit) em vez de markdown normal. Parece pouco, mas é literalmente metade do critério de aceite desta fase — e é o tipo de requisito de UX que separa "o sistema tecnicamente aborta certo" (já provado pela Fase 4) de "o usuário PERCEBE que o sistema abstraiu, sem ler a frase com atenção".

**Testes:** a lógica de backend nova (filtro por documento, campo `sources`) tem cobertura formal — 6 testes novos, 67 no total (ver `PLANO.md`). A UI Streamlit em si não tem teste automatizado formal no CI, mas foi validada com `streamlit.testing.v1.AppTest` (a própria ferramenta oficial de teste do Streamlit, que roda o script sem navegador e captura exceções) em dois cenários: corpus vazio (sem crash) e API sem chave configurada (erro tratado, aparece como aviso na tela, não como stack trace). A confirmação final — clicar numa citação de verdade contra o corpus e a Claude API reais — é o critério de aceite formal desta fase, e fica para a validação na máquina do usuário.

**Pergunta de entrevista que esta fase prepara bem:** *"Por que você não usou React/Next.js pra interface, já que o resto do sistema é 'sério'?"* — Streamlit troca controle de UI fino por velocidade de desenvolvimento, e para este portfólio o que precisa ficar visível é o RACIOCÍNIO do RAG (citação, verificação, abstenção), não a qualidade do design da interface. É uma escolha consciente de onde investir tempo de engenharia — documentada desde a Fase 0 (ver seção 4, tabela de stack) — não uma limitação percebida tarde demais.

### Apresentação — identidade visual, diagrama e README

Feito junto com a Fase 6, aproveitando uma janela em que a Claude API estava bloqueada (primeiro por uma chave sem workspace, depois por falta de créditos). Escolha deliberada de sequenciamento: a Fase 7 depende inteiramente da API, e o tema da interface precisa estar pronto **antes** das capturas de tela que vão para o README — senão esse trabalho é feito duas vezes.

**Dois erros reais de integração com a Claude API, que valem mais que qualquer tutorial:**

1. `This API key is not scoped to a workspace` — a chave tinha sido criada no nível da organização, e não dentro de um workspace específico. A API exige um dos dois: uma chave vinculada a workspace, ou o header `anthropic-workspace-id` em toda requisição. Resolvido gerando a chave dentro de um workspace, que é a opção que não espalha configuração pelo código.
2. `Your credit balance is too low` — conta sem créditos. Erro trivial, mas revelador: só aparece na primeira chamada real, e um sistema que trata isso como "não sei responder" estaria mentindo para o usuário.

**Esse segundo erro expôs uma incoerência no meu próprio código.** A UI capturava qualquer exceção e renderizava como *abstenção* — mesma cor, mesmo rótulo. Só que abstenção e falha de infraestrutura são conceitualmente opostas: uma é o sistema **decidindo** não responder porque as fontes não sustentam a resposta (o comportamento que este projeto inteiro existe para provar); a outra é ele **não ter conseguido nem tentar**. Confundir as duas na tela apaga justamente a distinção que é o produto. Corrigido: falha tem cor, rótulo e ícone próprios, e uma função (`mensagem_de_falha`) traduz a exceção crua num motivo acionável — inclusive esses dois erros de chave e crédito, que agora viram instrução em vez de stack trace.

**A identidade visual foi tratada como decisão de engenharia, não de gosto.** `docs/IDENTIDADE.md` é a fonte única de cor e tipografia — `.streamlit/config.toml`, o CSS da UI, o diagrama e o README consomem de lá, nenhum inventa valor próprio. É o mesmo princípio do `settings.py` (nenhum módulo hardcoda parâmetro), aplicado ao design. Três regras que valem explicar:

- **Cor é informação.** A tela é marinho, grafite e papel; as únicas cores vivas são as três semânticas (verde = citação verificada, âmbar = citação rejeitada, tijolo = abstenção). Se um elemento não comunica um desses estados, não usa nenhuma dessas cores — é isso que mantém a abstenção inconfundível, que era o critério de aceite da Fase 6.
- **Contraste verificado, não estimado.** Todas as combinações de texto passam WCAG AA, as principais passam AAA. Acessibilidade é um dos poucos critérios de design objetivamente verificáveis; ignorá-lo num projeto que se vende por rigor seria incoerente.
- **O logo é o mecanismo.** `[✓]` — a fonte numerada entre colchetes (exatamente como entra no prompt, `[1]`, `[2]`…) com a marca de verificação. Uma balança da justiça seria o clichê óbvio e não diria nada sobre o que o sistema faz.

**O diagrama de arquitetura é gerado por script** (`scripts/gerar_diagrama.py`), não desenhado à mão nem numa ferramenta visual. Com cerca de quarenta elementos, manter caixas no grid e setas sem cruzar texto à mão é onde o erro entra — e um diagrama desalinhado lê como rascunho por mais correto que esteja o conteúdo. Ter o desenho em código também significa que mudar a paleta ou renomear uma etapa é editar uma linha e rodar de novo. O SVG carrega fundo próprio de propósito: o README do GitHub renderiza em tema claro ou escuro conforme a preferência de quem lê, e um SVG transparente com texto escuro fica ilegível no escuro.

**O README foi reescrito como argumento, não como manual.** A ordem é tese → resultados → como funciona → as quatro defesas → **o que o projeto não faz** → como rodar. A seção de limitações é a que mais vale: um README que só lista virtudes não carrega informação, porque nenhum sistema só tem virtudes. Listar o que foi deixado de fora *e por quê* (streaming de progresso e não de token, estado de ingestão em memória, sem cross-document, sem OCR) demonstra que as escolhas foram feitas, não sofridas — e é exatamente o que um entrevistador técnico procura para distinguir quem entende o próprio sistema de quem montou um tutorial.

A tabela de resultados está no topo, **vazia e declarada como vazia**, até a Fase 7 rodar. Publicar um README que insinua métricas que não existem seria o mesmo erro que o projeto inteiro existe para combater: afirmação sem lastro verificável.

### Rebranding — de Cláusula a Crivo

Antes de publicar no GitHub, revisitei o nome. "Cláusula" nomeia o *objeto*
que o sistema lê — descreve o domínio (contratos), não o produto. Um teste
simples: qualquer outro RAG jurídico poderia se chamar assim, porque o nome
não carrega nenhuma informação sobre o que este sistema faz de diferente.

O que este sistema faz de diferente é filtrar: quatro defesas sucessivas
(limiar de score, prompt fechado, verificação literal de citação, rastro de
origem auditável) que só deixam passar uma resposta se ela atender a um
critério verificável. Isso tem nome em português — **crivo**, uma peneira,
algo que retém o que não passa no teste. O nome novo descreve o mecanismo,
não o material de entrada. É a mesma lógica que já sustentava a métrica
"Faithfulness" no README: nomear pelo que é medido, não pelo que é genérico.

O símbolo mudou junto, e vale registrar o motivo técnico, não só o estético.
A primeira tentativa de redesenho foi literal: pegar o "C" de Crivo e
desenhar farpas retas nas duas pontas abertas da curva, imitando a serifa
dos colchetes do símbolo anterior. O resultado, renderizado e inspecionado
em PNG antes de qualquer coisa ser aceita como final, lia como um gancho —
não como um remate tipográfico. Serifa funciona numa haste reta porque é,
literalmente, um pequeno traço perpendicular; numa curva, o mesmo truque
geométrico produz uma protuberância assimétrica que o olho lê como ruído,
não como acabamento. A correção foi assumir a curva pelo que ela é:
terminações arredondadas, com o mesmo peso e o mesmo remate
(`stroke-linecap: round`) do check que já existia dentro dela — duas formas,
uma linguagem só, em vez de uma curva fingindo ser outra coisa. É um
lembrete útil de que "refinar" às vezes significa simplificar a ideia até
ela parar de brigar com a geometria que a carrega, não adicionar detalhe.

A marca de verificação continua com o mesmo trabalho de antes — encaixada na
abertura da letra, e não ao lado dela, para manter o símbolo literal: o que
está dentro é o que passou pelo crivo. `docs/IDENTIDADE.md` tem a versão
completa do racional, incluindo por que o favicon precisou de um desenho
próprio (alargar a abertura e engrossar o traço, porque em 16 px silhueta é
tudo que sobra).

---

## 5. Perguntas de entrevista — banco de respostas

Isto complementa a seção "perguntas que você vai ouvir" do `PLANO.md`, mas indo mais fundo no *como* técnico.

**"Descreva a arquitetura do seu sistema RAG."**
> Pipeline em duas fases. Offline: PDF → parser preserva página → chunker corta por cláusula, respeitando um teto de tokens → embeddings BGE-M3 → ChromaDB (denso) + BM25 (léxico). Online: pergunta → busca híbrida nos dois índices, fundida por Reciprocal Rank Fusion → top-20 candidatos → reranking por cross-encoder → top-5 → prompt numerado para o Claude → resposta em JSON estruturado → verificação literal de cada citação contra o chunk de origem → resposta com fontes, ou abstenção.

**"Por que RRF e não simplesmente concatenar os dois resultados?"**
> Concatenar duplicaria ou perderia ordenação relativa. RRF funde as duas listas por posição, não por valor bruto — o que resolve o problema de as escalas de distância vetorial e score BM25 não serem comparáveis.

**"Como você garante que o sistema não alucina?"**
> Quatro camadas, cada uma no seu ponto do pipeline: abstenção por limiar de relevância na recuperação, contexto numerado e fechado no prompt, verificação fuzzy da citação na saída, e rastro de origem auditável na interface. Medido na Fase 7 com um golden set de 50 perguntas, incluindo 10 propositalmente sem resposta no corpus.

**"Qual foi a decisão mais difícil do projeto?"**
> Chunking estrutural em vez de fixed-size. Chunk fixo é mais simples de implementar, mas corta a unidade semântica de uma cláusula jurídica. Implementar um detector de hierarquia (regex para CLÁUSULA/Art./§/inciso/alínea) com fallback pra documentos sem essa estrutura levou mais tempo, mas é o que garante que cada chunk seja uma unidade citável de verdade.

**"O que você faria diferente com mais tempo/recursos?"**
> Fine-tuning do embedding no domínio jurídico brasileiro (o BGE-M3 é multilíngue genérico, não especializado); decomposição de consultas para perguntas cross-document ("compare os três contratos"), que está fora do escopo da v1 por design.

---

## 6. Glossário rápido

| Termo | Definição em uma linha |
|---|---|
| **Embedding** | Vetor numérico que representa o significado de um texto num espaço onde proximidade = similaridade semântica |
| **Similaridade de cosseno** | Medida de quão "alinhados" dois vetores estão, ignorando magnitude |
| **Bi-encoder** | Modelo que embeda dois textos separadamente (rápido, usado no embedding) |
| **Cross-encoder** | Modelo que processa dois textos juntos numa passada (lento, preciso, usado no reranking) |
| **BM25** | Algoritmo de ranking léxico baseado em frequência de termo (ponderada pela raridade do termo no corpus) |
| **RRF** | Reciprocal Rank Fusion — funde múltiplos rankings por posição, não por score |
| **Grounding** | Forçar a resposta de um LLM a se apoiar no contexto fornecido, não em conhecimento paramétrico |
| **Alucinação** | O modelo gera uma afirmação plausível mas falsa/não sustentada pelas fontes |
| **Faithfulness** | Métrica de RAG: a resposta é sustentada pelo contexto recuperado? |
| **Golden set** | Conjunto de perguntas com resposta esperada conhecida, usado para avaliação |
| **Chunking** | Divisão de um documento em pedaços menores para indexação e recuperação |
| **Fuzzy matching** | Comparação de strings tolerante a pequenas diferenças (não é comparação exata) |
| **Tool use** | Recurso da Claude API que força a resposta a seguir um JSON Schema específico, em vez de texto livre |
| **Abstenção** | O sistema recusar responder, de propósito, quando não tem contexto suficiente — a alternativa correta a alucinar |
| **SSE (Server-Sent Events)** | Protocolo HTTP simples para o servidor empurrar eventos incrementais ao cliente numa única conexão (mão única, ao contrário de WebSocket) |
| **Background task** | Trabalho disparado por uma requisição HTTP que continua depois da resposta ser devolvida — o cliente não espera o trabalho terminar |
| **Filtro pós-corte (armadilha)** | Aplicar um filtro depois que um `top_k`/paginação já cortou os candidatos — pode zerar resultados que existiam mas ficaram fora do corte |
| **WCAG AA/AAA** | Níveis de contraste mínimo entre texto e fundo — 4,5:1 e 7:1 — que tornam acessibilidade um critério objetivo, não opinião |

---

## 7. O que vem a seguir

Fases 7 e 8 ainda vão gerar conteúdo para este documento — avaliação (a fase mais rica em aprendizado de todas, com o golden set de 50 perguntas) e empacotamento. As Fases 4 e 5 estão fechadas e confirmadas na máquina do usuário. A Fase 6 (interface Streamlit, dois painéis, citação clicável, seletor de documentos real) está construída e com a suíte inteira verde (67 testes); falta a confirmação do critério de aceite — clicar numa citação de verdade — na máquina do usuário. Próximo passo depois disso: Fase 7, a avaliação com o golden set de 50 perguntas. Vou continuar atualizando esta seção conforme cada fase for concluída.
