# Identidade visual — Crivo

Fonte única da verdade para cor, tipografia e uso da marca. A UI (`ui/app.py`,
`.streamlit/config.toml`), o diagrama de arquitetura (`assets/arquitetura.svg`)
e o README derivam **daqui** — nenhum deles inventa um valor próprio. É o mesmo
princípio do `settings.py` (nenhum módulo hardcoda um parâmetro), aplicado ao
design.

---

## 1. O princípio

O Crivo analisa contratos administrativos para due diligence. Quem olha a
tela é alguém acostumado a ler documento jurídico — advogado, analista de M&A,
auditor. A interface tem que parecer **um instrumento de trabalho sério**, não
uma demo de chatbot.

Três consequências práticas:

1. **Sóbrio, não colorido.** Cor é informação, não decoração. Praticamente tudo
   é marinho, grafite e papel; as únicas cores vivas da tela carregam um
   significado funcional (citação verificada, citação rejeitada, abstenção).
2. **Papel, não tela.** O fundo é um off-white quente, não branco puro nem
   dark mode. Documento jurídico se lê em papel; a metáfora ajuda a leitura
   longa e diferencia de todo dashboard genérico.
3. **Serifa no que argumenta, sans no que opera, mono no que mede.** A
   hierarquia tipográfica separa três registros diferentes de informação.

---

## 2. Paleta

### Núcleo

| Nome | Hex | Uso |
|---|---|---|
| Marinho | `#14243D` | Marca, títulos, elementos primários |
| Marinho médio | `#1F3A5F` | Links, destaques secundários, traços do diagrama |
| Grafite | `#22262B` | Texto corrido |
| Cinza-pedra | `#5A6472` | Texto secundário, legendas, metadados |
| Papel | `#FBFAF7` | Fundo principal |
| Papel sombreado | `#F2F0EA` | Sidebar, cards, blocos de fonte |
| Linha | `#DDD9D0` | Bordas, divisores |

### Acento de marca

| Nome | Hex | Uso |
|---|---|---|
| Bronze | `#9C7B4D` | Só no logo e em detalhes finos. Nunca em área grande. |

### Semânticos — cada um com um trabalho só

| Nome | Hex | Significa |
|---|---|---|
| Verde-verificado | `#2C6E49` | A citação passou na verificação literal contra o chunk |
| Âmbar-rejeitada | `#8F5209` | A citação falhou na verificação (ficou abaixo do limiar) |
| Tijolo-abstenção | `#8C2F1E` | O sistema se recusou a responder |

**Regra:** se um elemento não comunica um desses três estados, ele não usa
nenhuma dessas três cores. É o que mantém a abstenção visualmente inconfundível
— o critério de aceite da Fase 6.

### Contraste

Todas as combinações de texto sobre `Papel` passam WCAG AA (≥ 4,5:1); as
principais passam AAA:

| Combinação | Razão | Nível |
|---|---|---|
| Grafite sobre Papel | ~14:1 | AAA |
| Marinho sobre Papel | ~13:1 | AAA |
| Tijolo sobre Papel | ~8:1 | AAA |
| Cinza-pedra sobre Papel | ~6:1 | AA |
| Âmbar sobre Papel | ~6:1 | AA |
| Verde sobre Papel | ~5,5:1 | AA |

Isso não é preciosismo: acessibilidade é um dos poucos critérios de design que
são objetivamente verificáveis, e um portfólio que respeita contraste diz algo
sobre o rigor de quem o fez.

---

## 3. Tipografia

| Registro | Fonte | Fallback | Onde |
|---|---|---|---|
| Títulos | **Lora** (600) | Georgia, serif | `h1`–`h3`, nome da marca, cabeçalhos de seção |
| Corpo / interface | **Source Sans 3** | system-ui, sans-serif | Texto, botões, rótulos, respostas |
| Dados | **Source Code Pro** | ui-monospace, monospace | Scores, latência, contagens, `chunk_id` |

**Por que serifa nos títulos:** é o sinal tipográfico mais barato de "documento
sério" que existe. Custo zero, leitura imediata.

**Por que Lora, especificamente:** contraste moderado entre traço grosso e fino
(lê bem em tela, ao contrário de uma Didone), terminações levemente
caligráficas (dá o ar jurídico sem virar fantasia) e um peso 600 que sustenta
título curto sem pesar. Source Sans 3 e Source Code Pro já são as fontes
padrão do Streamlit — ou seja, o único webfont extra que a interface carrega é
a Lora. Menos rede, menos flash de fonte não estilizada, uma dependência só.

**Por que mono nos números:** dígitos monoespaçados alinham em coluna. Quando o
usuário compara `score 0,981` com `score 0,797` em fontes diferentes, o
alinhamento faz a diferença ser visível sem ler.

---

## 4. A marca

**Nome:** Crivo. **Símbolo:** um "C" que se abre para uma marca de
verificação encaixada na sua própria abertura.

### Por que Crivo, e não Cláusula

"Cláusula" nomeia o *objeto* que o sistema lê — descreve o domínio, não o
produto. Qualquer RAG jurídico poderia se chamar assim. "Crivo" nomeia o
*mecanismo*: um crivo é uma peneira, algo que só deixa passar o que atende a
um critério — e é exatamente o que o pipeline faz em quatro pontos
sucessivos (limiar de score, prompt fechado, verificação literal de citação,
rastro de origem auditável na interface). O nome descreve a tese do projeto
("provar que não inventa"), não só o material de entrada.

Efeito colateral que pesou a favor: "Crivo" começa com a mesma letra que já
sustentava o símbolo anterior (o colchete de citação), então o redesenho do
símbolo em torno da inicial **C** não é coincidência forçada — é a mesma
ideia visual reconciliada com o nome novo, em vez de duas marcas competindo.

### O símbolo

O símbolo evoluiu de dois colchetes serifados (`[✓]`, a citação numerada
entre colchetes) para um único "C" monogramático com a marca de verificação
encaixada na abertura. Três motivos:

1. **A inicial de Crivo** — o símbolo agora lê como a própria marca, não
   como um elemento genérico de pontuação tipográfica.
2. **Um monograma sustenta peso em qualquer tamanho.** Duas formas separadas
   (colchete esquerdo + colchete direito + check) competem por espaço num
   favicon de 16 px; uma forma fechada não.
3. **A metáfora ainda é literal, não decorativa.** O check não está "do lado"
   do C — ele está *dentro* da abertura, como o que passou pelo crivo e foi
   confirmado. É o mesmo compromisso de antes (o símbolo é o mecanismo, não
   um ícone de estoque): uma balança da justiça continua sendo o clichê
   óbvio que não diria nada sobre o que o sistema faz.

Terminações arredondadas, não serifadas: uma primeira tentativa desenhou
farpas retas nas duas pontas abertas do C, imitando a serifa do colchete
original. O resultado lia como um gancho, não como um remate tipográfico —
serifa é fácil de desenhar numa haste reta e difícil de fingir numa curva
sem o aparato de uma fonte de verdade. A saída foi assumir o traço
arredondado com o mesmo remate do check: dois elementos, uma única
linguagem de terminação, em vez de uma curva tentando imitar uma reta.

### Uso

- Área de proteção: no mínimo o raio do C em volta do símbolo.
- Monocromia: marinho sobre papel, ou papel sobre marinho. O bronze aparece só
  na marca de verificação, e só na versão principal.
- Nunca esticar, rotacionar, aplicar sombra ou gradiente.

### Arquivos

| Arquivo | O que é |
|---|---|
| `assets/logo.svg` | Horizontal: símbolo + nome. Uso principal (README, apresentações) |
| `assets/simbolo.svg` | Só o símbolo. Espaços quadrados, cabeçalho compacto |
| `assets/favicon.svg` | Símbolo simplificado sobre marinho, para 16–32 px |
| `assets/favicon.png` | O mesmo, rasterizado em 256 px (o `page_icon` do Streamlit é mais confiável com PNG) |

**O nome no `logo.svg` está convertido em contorno vetorial**, não em `<text>`.
Um `<text>` dependeria da Lora estar instalada em quem abre o arquivo — no
GitHub, no LinkedIn ou num slide, cairia no fallback e a marca sairia diferente
em cada lugar. Contorno renderiza igual em qualquer máquina. O custo é que o
arquivo fica maior (~7 KB) e o texto não é mais editável: para mudar o nome,
gera-se o contorno de novo a partir da fonte.

**A versão do favicon é desenhada à parte, de propósito.** Reduzir o
`simbolo.svg` para 16 px funde o traço do C e o check numa mancha. A versão de
favicon espessa o traço, alarga a abertura do C e inverte as cores (C claro
sobre fundo marinho) — mesma ideia, desenho diferente, porque em 16 px o que
importa é a silhueta, não o desenho fiel.

---

## 5. Onde cada valor é consumido

| Arquivo | O que consome |
|---|---|
| `.streamlit/config.toml` | Núcleo da paleta + base clara |
| `ui/app.py` | Semânticos, fontes (via CSS), logo no cabeçalho |
| `assets/arquitetura.svg` | Núcleo + semânticos nas quatro defesas |
| `README.md` | Logo, diagrama |

Mudou um hex aqui? Ele precisa mudar nos quatro. É o custo de não ter um
sistema de build de CSS — aceito conscientemente, porque a alternativa
(pipeline de tokens de design) seria desproporcional a um projeto deste tamanho.
