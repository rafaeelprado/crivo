# Publicar no GitHub — checklist e comandos

Guia de referência para a primeira publicação do **Crivo**. Como não tenho
acesso a um terminal com `git` na sua máquina, os comandos abaixo são para
você rodar no PowerShell, dentro de `C:\Sistema RAG`.

---

## 0. Decisões já tomadas

- **Repositório já criado:** [github.com/rafaeelprado/crivo](https://github.com/rafaeelprado/crivo)
  — nome `crivo`, público, sem README/.gitignore/license automáticos (os
  seus locais já existem). Falta só o primeiro push.
- **Visibilidade:** público
- **Corpus (PDFs em `data/raw/`):** vai para o repositório — são contratos
  públicos do PNCP, escolhidos desde a Fase 1 exatamente por serem
  publicáveis sem risco de vazamento. `.gitignore` já foi ajustado para
  parar de ignorar `data/raw/` (só continuam de fora `data/processed/`,
  `data/chroma/` e `data/logs/`, que são derivados e se regeneram com os
  comandos de `docs/COMO_RODAR.md`).

Se algum desses três pontos mudar de ideia depois, é só reeditar
`.gitignore` e recomeçar do passo 2.

---

## 1. Checklist antes de publicar (rodar no PowerShell)

### 1.1 Confirmar que o `.env` nunca foi rastreado

```powershell
git status --ignored
```

Se `.env` aparecer na lista de "Untracked" ou, pior, já commitado em algum
commit anterior, **pare aqui** — chave de API não pode ir pro histórico do
git (remover depois de commitado não basta; o commit antigo continua
acessível). Ele deve aparecer em "Ignored files".

### 1.2 Medir o tamanho do corpus

```powershell
$total = (Get-ChildItem -Recurse "data\raw" -File | Measure-Object -Property Length -Sum).Sum
"{0:N1} MB" -f ($total / 1MB)
Get-ChildItem -Recurse "data\raw" -File | Sort-Object Length -Descending | Select-Object -First 3 Name, @{N="MB";E={"{0:N1}" -f ($_.Length/1MB)}}
```

O GitHub recusa qualquer arquivo individual acima de 100 MB e alerta a
partir de 50 MB. Contratos administrativos em PDF de texto quase nunca
chegam nem perto disso, mas vale rodar antes de descobrir depois de um
push travado. Se algum arquivo passar de 100 MB, ele precisa ir por
[Git LFS](https://git-lfs.com/) — me avisa que ajusto o `.gitignore`/README.

### 1.3 Confirmar que os testes passam

```powershell
pytest -q
```

Publicar com a suíte quebrada é o tipo de coisa que um revisor técnico
confere primeiro.

---

## 2. Inicializar o repositório local

Se a Fase 0 já rodou `git init` nesta pasta, pule para o passo 2.2.

### 2.1 Primeira vez (sem `.git` ainda)

```powershell
git init
git branch -M main
```

### 2.2 Verificar o que já existe

```powershell
git status
git log --oneline -5
git remote -v
```

Cole o resultado aqui se quiser que eu confira antes de continuar —
principalmente se `git remote -v` já mostrar alguma URL (significa que já
existe um repositório remoto configurado, e os passos 3–4 abaixo mudam).

---

## 3. Conectar ao repositório já criado

O repositório já existe em
[github.com/rafaeelprado/crivo](https://github.com/rafaeelprado/crivo),
vazio. Só falta apontar o repositório local pra ele:

```powershell
git remote add origin https://github.com/rafaeelprado/crivo.git
```

Se `git remote -v` (passo 2.2) já mostrar uma URL diferente em `origin`,
não rode o comando acima — me avisa antes, porque significa que o remoto já
aponta pra outro lugar.

---

## 4. Commit e push

```powershell
git add .
git status                 # confira a lista antes de commitar — é a sua
                            # última chance de pegar algo que não devia ir
git commit -m "Crivo: RAG para due diligence contratual com citações verificadas"
git push -u origin main
```

Se o Windows pedir login nesse momento, use o navegador que abrir (fluxo
padrão do Git Credential Manager) — não digite usuário/senha direto no
terminal.

---

## 5. Depois do push — checklist final

- [ ] Abrir [github.com/rafaeelprado/crivo](https://github.com/rafaeelprado/crivo) e conferir que o
      README renderiza com o logo e o diagrama de arquitetura aparecendo
      corretamente (imagens de `assets/` só renderizam se o repo for
      público ou se você estiver logado)
- [ ] Conferir que **nenhum arquivo `.env` aparece** na lista de arquivos
      do repositório
- [ ] Em **Settings → General**, preencher a descrição curta e adicionar
      `README.md` como website, se quiser
- [ ] Adicionar **topics** (a caixinha de tags no topo da página do repo):
      `rag`, `llm`, `legal-tech`, `fastapi`, `streamlit`, `python` — é o
      que faz o projeto aparecer em busca dentro do GitHub
- [ ] Se quiser, fixar o repositório no seu perfil (**Pinned repositories**,
      na página do seu perfil) — é o primeiro lugar que um recrutador olha

---

## 6. Se algo vazar sem querer

Se um `git push` acabar subindo o `.env` ou qualquer segredo:

1. **Revogue a chave imediatamente** em
   [console.anthropic.com](https://console.anthropic.com/settings/keys) —
   antes de qualquer limpeza de histórico. Uma chave que já foi para um
   push público deve ser tratada como comprometida, mesmo que o repositório
   seja apagado logo em seguida.
2. Só depois disso, limpar o histórico (`git filter-repo` ou, mais simples
   para um repositório recém-criado, apagar e recriar do zero).

Me chama que eu ajudo com os comandos dessa parte se precisar — não é o
caminho normal, é só o plano B.
