"""Gera assets/arquitetura.svg — o pipeline do Crivo com as quatro defesas
ancoradas nos pontos onde de fato agem.

O desenho é gerado por script, e não escrito à mão, por um motivo prático:
alinhamento. Caixa fora do grid e seta que cruza texto são o que separa um
diagrama que parece deliberado de um que parece rascunho — e manter isso à
mão, com ~40 elementos, é onde o erro entra.
"""

MARINHO = "#14243D"
PEDRA = "#5A6472"
PAPEL = "#FBFAF7"
PAPEL_S = "#F2F0EA"
LINHA = "#DDD9D0"
VERDE = "#2C6E49"
TIJOLO = "#8C2F1E"

SANS = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
MONO = "'Source Code Pro', ui-monospace, 'SF Mono', Menlo, Consolas, monospace"

W, H = 1060, 512
out = []

BOX_W, BOX_H = 164, 62
COLS = [48, 248, 448, 648, 848]  # grid de 5 colunas, gap de 36


def caixa(x, y, titulo, sub=None, cor=MARINHO, fill=PAPEL_S, stroke=LINHA, w=BOX_W, h=BOX_H):
    out.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.5"/>'
    )
    cx = x + w / 2
    if sub:
        out.append(
            f'<text x="{cx}" y="{y + h / 2 - 3}" text-anchor="middle" font-family="{SANS}" '
            f'font-size="14.5" font-weight="600" fill="{cor}">{titulo}</text>'
        )
        out.append(
            f'<text x="{cx}" y="{y + h / 2 + 15}" text-anchor="middle" font-family="{MONO}" '
            f'font-size="11" fill="{PEDRA}">{sub}</text>'
        )
    else:
        out.append(
            f'<text x="{cx}" y="{y + h / 2 + 5}" text-anchor="middle" font-family="{SANS}" '
            f'font-size="15" font-weight="600" fill="{cor}">{titulo}</text>'
        )


def seta(x1, y1, x2, y2, cor=PEDRA):
    out.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{cor}" stroke-width="1.6" '
        'marker-end="url(#ponta)"/>'
    )


def caminho(pontos, tracejada=False, cor=PEDRA):
    d = " ".join(f"{x},{y}" for x, y in pontos)
    dash = ' stroke-dasharray="5 4"' if tracejada else ""
    out.append(
        f'<polyline points="{d}" fill="none" stroke="{cor}" stroke-width="1.6"{dash} '
        f'stroke-linejoin="round" marker-end="url(#ponta)"/>'
    )


def rotulo_faixa(x, y, texto):
    out.append(
        f'<text x="{x}" y="{y}" font-family="{MONO}" font-size="11.5" font-weight="600" '
        f'letter-spacing="1.6" fill="{PEDRA}">{texto}</text>'
    )


def nota(x, y, texto, cor=PEDRA, anchor="start"):
    out.append(
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{MONO}" font-size="10.5" '
        f'fill="{cor}">{texto}</text>'
    )


def defesa(n, x, y, texto):
    """Marca uma das quatro defesas no ponto do pipeline onde ela age."""
    out.append(f'<circle cx="{x}" cy="{y}" r="10.5" fill="{TIJOLO}"/>')
    out.append(
        f'<text x="{x}" y="{y + 4.3}" text-anchor="middle" font-family="{SANS}" '
        f'font-size="12" font-weight="700" fill="{PAPEL}">{n}</text>'
    )
    out.append(
        f'<text x="{x + 17}" y="{y + 4.3}" font-family="{SANS}" font-size="12.5" '
        f'font-weight="600" fill="{TIJOLO}">{texto}</text>'
    )


out.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="{PAPEL}"/>')

# ------------------------------------------------------------------ ingestão
rotulo_faixa(48, 40, "INGESTÃO — OFFLINE, UMA VEZ POR CORPUS")

for (titulo, sub), x in zip(
    [
        ("PDFs", "data/raw"),
        ("PyMuPDF", "texto + página"),
        ("chunking", "por cláusula"),
        ("BGE-M3", "embeddings"),
        ("ChromaDB + BM25", "índice híbrido"),
    ],
    COLS,
    strict=True,
):
    caixa(x, 56, titulo, sub)
for x in COLS[:-1]:
    seta(x + BOX_W, 87, x + BOX_W + 36, 87)

out.append(
    f'<line x1="48" y1="158" x2="1012" y2="158" stroke="{LINHA}" stroke-width="1.5" '
    'stroke-dasharray="3 5"/>'
)

# ------------------------------------------------------------------ consulta
rotulo_faixa(48, 196, "CONSULTA — TEMPO REAL, A CADA PERGUNTA")
defesa(1, 800, 192, "abstém sem chamar o modelo")

for (titulo, sub), x in zip(
    [
        ("pergunta", "linguagem natural"),
        ("busca híbrida", "denso + léxico"),
        ("RRF", "top-20"),
        ("reranking", "cross-encoder, top-5"),
        ("limiar de score", "MIN_RERANK_SCORE"),
    ],
    COLS,
    strict=True,
):
    caixa(x, 212, titulo, sub, cor=TIJOLO if titulo == "limiar de score" else MARINHO)
for x in COLS[:-1]:
    seta(x + BOX_W, 243, x + BOX_W + 36, 243)

# o índice construído offline é o que a busca lê
caminho([(930, 118), (930, 138), (330, 138), (330, 210)], tracejada=True)
nota(640, 132, "lê o índice construído offline", anchor="middle")

# wrap: do limiar de score para a segunda linha da consulta
caminho([(930, 274), (930, 310), (130, 310), (130, 348)])
nota(520, 302, "score acima do limiar")

caixa(COLS[0], 350, "prompt fechado", "fontes [1]…[5]", cor=TIJOLO)
caixa(COLS[1], 350, "Claude Haiku", "tool use, JSON")
caixa(COLS[2], 350, "verificação", "casamento ≥ 0,90", cor=TIJOLO)
for x in COLS[:2]:
    seta(x + BOX_W, 381, x + BOX_W + 36, 381)

# ------------------------------------------------------------------ saídas
caixa(800, 330, "Resposta com fontes", None, cor=VERDE, stroke=VERDE, fill="#FFFFFF", w=210, h=54)
caixa(800, 424, "Abstenção", None, cor=TIJOLO, stroke=TIJOLO, fill="#FFFFFF", w=210, h=54)

caminho([(612, 381), (700, 381), (700, 357), (798, 357)], cor=VERDE)
caminho([(612, 381), (700, 381), (700, 451), (798, 451)], cor=TIJOLO)
nota(706, 350, "confere", cor=VERDE)
nota(706, 438, "não confere 2×", cor=TIJOLO)

# a defesa 1 desemboca na mesma abstenção, sem passar pelo modelo
caminho([(1012, 243), (1040, 243), (1040, 451), (1014, 451)], tracejada=True, cor=TIJOLO)

defesa(2, 59, 470, "contexto numerado, sem conhecimento externo")
defesa(3, 459, 470, "citação casada contra o chunk de origem")
defesa(4, 758, 404, "rastro de origem auditável")
out.append(
    f'<line x1="758" y1="393" x2="758" y2="386" stroke="{TIJOLO}" stroke-width="1.4" '
    'stroke-dasharray="2 3"/>'
)

svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
    'role="img" aria-label="Arquitetura do Clausula. Ingestao offline: PDFs, PyMuPDF, chunking '
    'por clausula, embeddings BGE-M3, indice hibrido ChromaDB mais BM25. Consulta em tempo real: '
    'pergunta, busca hibrida, fusao RRF, reranking por cross-encoder, limiar de score, prompt '
    'fechado, Claude Haiku com tool use, verificacao literal da citacao; a saida e uma resposta '
    'com fontes ou uma abstencao. As quatro defesas contra alucinacao estao marcadas nos pontos '
    'do pipeline onde agem.">\n'
    "  <defs>\n"
    '    <marker id="ponta" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
    'markerHeight="7" orient="auto-start-reverse">\n'
    '      <path d="M 0 1 L 10 5 L 0 9 z" fill="context-stroke"/>\n'
    "    </marker>\n"
    "  </defs>\n  " + "\n  ".join(out) + "\n</svg>\n"
)

open("/mnt/user-data/outputs/sistema-rag/assets/arquitetura.svg", "w", encoding="utf-8").write(svg)
print("ok", len(svg), "bytes")
