# Como rodar o Crivo

Guia de referência — não é o README (que é o argumento de portfólio), é o
"cola" prático para quando você só quer subir o sistema de novo sem
reconstruir o raciocínio do zero.

---

## Resumo rápido (já configurado antes)

Dois terminais PowerShell, ambos dentro de `C:\Sistema RAG`, com o venv
ativado:

```powershell
# Terminal 1 — API
.venv\Scripts\Activate.ps1
uvicorn rag.api.main:app --reload

# Terminal 2 — interface
.venv\Scripts\Activate.ps1
streamlit run ui/app.py
```

- API e documentação interativa: http://localhost:8000/docs
- Interface: abre sozinha no navegador, normalmente http://localhost:8501

Para parar: `Ctrl+C` em cada terminal. Fechar o terminal também derruba o
processo.

**O erro mais comum de todos:** rodar `streamlit run ui/app.py` num terminal
onde o venv não foi ativado. O sintoma é `streamlit` "não reconhecido como
comando" ou a versão errada do Python sendo usada. Solução: sempre
`.venv\Scripts\Activate.ps1` primeiro, em **cada** terminal novo — ativar
num terminal não vale para o outro.

---

## 1. Primeira vez — configuração do zero

Só precisa fazer isto uma vez por máquina (ou depois de apagar o `.venv`).

### 1.1 Pré-requisitos

- Python 3.11 ou superior instalado e no PATH
- Uma chave da [Claude API](https://console.anthropic.com/settings/keys)
  **vinculada a um workspace específico** — uma chave criada no nível da
  organização é recusada com `not scoped to a workspace`. Ao criar a chave,
  no Console, selecione um workspace específico, não "no workspace" nem o
  nível de organização.
- (Opcional) Docker Desktop, só se for usar o caminho containerizado (seção 4).

### 1.2 Criar o ambiente e instalar

```powershell
cd "C:\Sistema RAG"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

A instalação em modo editável (`-e`) é o que faz `uvicorn rag.api.main:app`
e `streamlit run ui/app.py` funcionarem sem precisar setar `PYTHONPATH` —
diferente do sandbox de desenvolvimento, onde isso era necessário.

### 1.3 Configurar o `.env`

```powershell
copy .env.example .env
notepad .env
```

Preencha pelo menos `ANTHROPIC_API_KEY`. Os demais valores em
`.env.example` já vêm com um padrão razoável (comentado, com o motivo de
cada um).

**Depois de editar o `.env`, sempre reinicie a API** (`Ctrl+C` e rodar de
novo) — as configurações (`settings.py`) são lidas uma vez por processo e
ficam em cache. Editar o arquivo com a API já rodando não tem efeito até
reiniciar.

### 1.4 Montar o índice a partir do corpus

Só precisa repetir esta etapa se o corpus (os PDFs em `data/raw/`) mudar.
O índice fica salvo em `data/chroma/` e sobrevive a reinícios normais.

```powershell
python scripts/check_corpus.py           # triagem dos PDFs, gera manifest.csv
python -m rag.ingest.pipeline --verbose  # chunking estrutural -> data/processed/chunks.jsonl
python -m rag.ingest.indexer             # embeddings + BM25 -> data/chroma/
```

Na primeira execução do `indexer`, o BGE-M3 e o reranker são baixados do
Hugging Face (~2,5 GB somados) — só acontece uma vez, ficam em cache local.
Depois disso, ingestão e indexação rodam offline; só a geração da resposta
(a chamada à Claude API) usa rede.

---

## 2. Rodar no dia a dia

Depois que a seção 1 já foi feita uma vez:

```powershell
# Terminal 1
cd "C:\Sistema RAG"
.venv\Scripts\Activate.ps1
uvicorn rag.api.main:app --reload

# Terminal 2
cd "C:\Sistema RAG"
.venv\Scripts\Activate.ps1
streamlit run ui/app.py
```

`--reload` na API faz o uvicorn reiniciar sozinho quando um arquivo `.py`
muda — útil enquanto você mexe no código, mas não recarrega o `.env` (ver
seção 1.3).

Sem interface, só para testar a busca e a geração pela linha de comando:

```powershell
python -m rag.retrieval.cli "multa por atraso na entrega"    # só a busca
python -m rag.generation.cli "multa por atraso na entrega"   # busca + resposta verificada
```

---

## 3. Rodar os testes

```powershell
pytest -q
```

Nenhum teste gasta chamada de API real (nem Claude, nem os modelos locais)
— toda a suíte roda em segundos, mesmo sem `.env` configurado.

Para checar formatação/lint antes de commitar:

```powershell
ruff check --fix src tests ui scripts
ruff format src tests ui scripts
```

---

## 4. Caminho Docker (alternativo)

Sobe API + UI + volume persistente do Chroma num único comando, sem venv:

```powershell
docker compose up --build
```

- API: http://localhost:8000
- Interface: http://localhost:8501

Para derrubar:

```powershell
docker compose down
```

O `.env` é lido pelos dois containers (`api` e `ui`) via `env_file` no
`docker-compose.yml` — precisa existir antes de subir.

---

## 5. Tabela de comandos (equivalente ao `Makefile`)

O `Makefile` é um atalho para quem tem `make` instalado (não é padrão no
Windows). Os comandos completos abaixo são exatamente o que cada alvo roda:

| Atalho (`make ...`) | Comando completo |
|---|---|
| `install` | `pip install -e ".[dev]"` |
| `api` | `uvicorn rag.api.main:app --reload --host 0.0.0.0 --port 8000` |
| `ui` | `streamlit run ui/app.py` |
| `test` | `pytest -q` |
| `lint` | `ruff check --fix src tests ui scripts` + `ruff format src tests ui scripts` |
| `corpus` | `python scripts/check_corpus.py` |
| `ingest` | `python -m rag.ingest.pipeline` (ainda falta o `indexer` — ver 1.4) |
| `search Q="..."` | `python -m rag.retrieval.cli "..."` |
| `up` / `down` | `docker compose up --build` / `docker compose down` |
| `clean` | Apaga `data/chroma/*` e `data/processed/*` (mantém os PDFs de `data/raw/`) — obriga a rodar a seção 1.4 de novo |

---

## 6. Erros comuns

| Sintoma | Causa | Solução |
|---|---|---|
| `streamlit` "não é reconhecido como um comando" | Venv não ativado nesse terminal | `.venv\Scripts\Activate.ps1` |
| `ModuleNotFoundError: No module named 'rag'` | Pacote não instalado em modo editável, ou venv errado | `pip install -e ".[dev]"` dentro do venv ativado |
| `This API key is not scoped to a workspace...` | Chave da Claude API criada no nível da organização, não dentro de um workspace | Gerar uma chave nova, selecionando um workspace específico no Console; atualizar `.env`; reiniciar a API |
| `Your credit balance is too low...` | Sem créditos na conta Anthropic | Adicionar créditos em [console.anthropic.com](https://console.anthropic.com) → Plans & Billing. Não precisa reiniciar nada — o próximo pedido já funciona |
| A UI mostra "Falha de conexão" | A API não está no ar, ou está numa porta diferente da esperada | Confirmar que o terminal da API está rodando sem erro e escutando em `:8000` |
| Editei o `.env` e nada mudou | `Settings` é lido uma vez por processo e fica em cache | Reiniciar a API (`Ctrl+C` e rodar de novo) |
| Busca não acha nada num documento que existe | Índice desatualizado (PDF novo em `data/raw/` sem reindexar) | Rodar a seção 1.4 de novo |
| Porta 8000 ou 8501 "already in use" | Um processo anterior ainda está rodando (terminal fechado sem `Ctrl+C`, ou um segundo terminal tentando subir o mesmo serviço) | Fechar o processo antigo, ou trocar a porta (`--port` no uvicorn, `--server.port` no streamlit) |

---

## 7. Onde as coisas ficam

```
C:\Sistema RAG\
├── .env                  suas chaves e configuração — nunca vai pro git
├── .venv\                ambiente virtual Python — nunca vai pro git
├── data\
│   ├── raw\               PDFs originais + manifest.csv
│   ├── processed\         chunks.jsonl (gerado pela ingestão)
│   ├── chroma\            índice vetorial persistido (gerado pela indexação)
│   └── logs\queries.jsonl log estruturado de cada consulta (usado pelo /metrics)
```

Apagar `data/chroma/` e `data/processed/` (o que `make clean` faz) não
apaga os PDFs — só obriga a rodar a ingestão e a indexação de novo.
