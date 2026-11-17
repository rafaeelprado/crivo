"""Métricas da avaliação v2 do Crivo.

Só biblioteca padrão, de propósito: este módulo é pura matemática sobre
registros já coletados, então roda no CI e nos testes sem modelo, sem índice
e sem chave de API. A parte que fala com o pipeline fica em
`scripts/run_eval_v2.py`.

Por que existe uma v2: no teste de 50 perguntas, as três configurações
empatavam em 100% de abstenção correta, e com só 10 perguntas sem resposta
isso quase não prova nada (limite inferior de cerca de 72%). A v2 tem 100
perguntas, 40 delas sem resposta, e reporta intervalo de confiança em toda
taxa para que um empate seja visto como empate.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

# Tipos do golden_set_v2.csv. A, B e C têm resposta nos contratos;
# D (a informação não está no contrato citado) e E (fora do domínio)
# devem terminar em abstenção.
TIPOS_COM_RESPOSTA = ("A", "B", "C")
TIPOS_SEM_RESPOSTA = ("D", "E")


# ---------------------------------------------------------------------------
# Estatística básica
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confiança de Wilson (95% por padrão) para uma proporção.

    Melhor que a aproximação normal justamente nos extremos (0% e 100%) e com
    amostra pequena, que é o caso de uma avaliação de RAG.
    """
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def percentile(values: list[float], q: float) -> float | None:
    """Percentil por interpolação linear (q entre 0 e 100). None se vazio."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (q / 100)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[int(pos)]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


# ---------------------------------------------------------------------------
# Recuperação: o trecho certo apareceu entre os primeiros resultados?
# ---------------------------------------------------------------------------


def parse_pages(raw: str) -> set[int]:
    """'3' -> {3}; '2;3' -> {2, 3}; '4-6' -> {4, 5, 6}; '' -> vazio."""
    pages: set[int] = set()
    for part in re.split(r"[;,]", raw or ""):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            if start <= end:
                pages.update(range(start, end + 1))
        elif part.isdigit():
            pages.add(int(part))
    return pages


def is_hit(
    chunk_document: str,
    chunk_page_start: int,
    chunk_page_end: int,
    expected_document: str,
    expected_pages: set[int],
) -> bool:
    """O chunk é a evidência esperada?

    Mesmo documento e, se o gabarito indica páginas, interseção entre as
    páginas do chunk e as páginas de evidência.
    """
    if chunk_document != expected_document:
        return False
    if not expected_pages:
        return True
    return any(chunk_page_start <= p <= chunk_page_end for p in expected_pages)


def retrieval_scores(hits: list[bool], k: int = 5) -> tuple[bool, float]:
    """(recall@k, reciprocal rank) de uma pergunta.

    `hits` é a lista, na ordem do ranking, dizendo se cada chunk é evidência.
    O reciprocal rank considera a lista inteira recebida (ex.: os 20
    candidatos), não só os k primeiros.
    """
    recall = any(hits[:k])
    for i, hit in enumerate(hits, start=1):
        if hit:
            return recall, 1.0 / i
    return recall, 0.0


# ---------------------------------------------------------------------------
# Registro por pergunta e resumo por configuração
# ---------------------------------------------------------------------------


@dataclass
class QuestionRecord:
    config: str
    id: str
    tipo: str
    pergunta: str
    abstencao_esperada: bool
    # None = a geração não rodou (modo --dry-run).
    abstained: bool | None = None
    answer: str = ""
    # 'correta' | 'parcial' | 'incorreta' | None (sem juiz, ou pergunta sem resposta)
    verdict: str | None = None
    citations_total: int = 0
    citations_verified: int = 0
    retries: int = 0
    retrieval_top_score: float | None = None
    # Recuperação (só para perguntas com gabarito de página).
    hit_at_5: bool | None = None
    reciprocal_rank: float | None = None
    retrieval_s: float | None = None
    latency_s: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _rate(successes: int, total: int) -> dict:
    low, high = wilson_interval(successes, total)
    return {
        "k": successes,
        "n": total,
        "rate": (successes / total) if total else None,
        "ci_low": low if total else None,
        "ci_high": high if total else None,
    }


def summarize(
    records: list[QuestionRecord], price_in_per_mtok: float, price_out_per_mtok: float
) -> dict:
    """Resumo de UMA configuração. Todas as taxas vêm com intervalo de Wilson."""
    ok = [r for r in records if r.error is None]
    answerable = [r for r in ok if not r.abstencao_esperada]
    unanswerable = [r for r in ok if r.abstencao_esperada]
    generated = [r for r in ok if r.abstained is not None]

    out: dict = {
        "n_total": len(records),
        "n_errors": len(records) - len(ok),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
    }

    # --- Geração (só quando a API rodou) ---
    gen_answerable = [r for r in answerable if r.abstained is not None]
    gen_unanswerable = [r for r in unanswerable if r.abstained is not None]
    if gen_answerable:
        out["unwarranted_abstention"] = _rate(
            sum(1 for r in gen_answerable if r.abstained), len(gen_answerable)
        )
        judged = [r for r in gen_answerable if not r.abstained and r.verdict]
        if judged or any(r.verdict for r in gen_answerable):
            # "Correta" = respondeu E o juiz aprovou, sobre TODAS as respondíveis:
            # abster-se à toa conta como erro.
            out["answer_correct_strict"] = _rate(
                sum(1 for r in gen_answerable if not r.abstained and r.verdict == "correta"),
                len(gen_answerable),
            )
            out["answer_correct_lenient"] = _rate(
                sum(
                    1
                    for r in gen_answerable
                    if not r.abstained and r.verdict in ("correta", "parcial")
                ),
                len(gen_answerable),
            )
    if gen_unanswerable:
        out["correct_abstention"] = _rate(
            sum(1 for r in gen_unanswerable if r.abstained), len(gen_unanswerable)
        )
        for tipo in TIPOS_SEM_RESPOSTA:
            subset = [r for r in gen_unanswerable if r.tipo == tipo]
            if subset:
                out[f"correct_abstention_{tipo}"] = _rate(
                    sum(1 for r in subset if r.abstained), len(subset)
                )

    answered = [r for r in generated if not r.abstained]
    cit_total = sum(r.citations_total for r in answered)
    if cit_total:
        out["citations_verified"] = _rate(sum(r.citations_verified for r in answered), cit_total)

    # --- Recuperação ---
    with_gold = [r for r in answerable if r.hit_at_5 is not None]
    if with_gold:
        out["recall_at_5"] = _rate(sum(1 for r in with_gold if r.hit_at_5), len(with_gold))
        out["mrr"] = sum(r.reciprocal_rank or 0.0 for r in with_gold) / len(with_gold)

    # --- Latência e custo ---
    lat = [r.latency_s for r in ok if r.latency_s is not None]
    if lat:
        out["latency_s"] = {
            "p50": percentile(lat, 50),
            "p95": percentile(lat, 95),
            "mean": sum(lat) / len(lat),
        }
    ret = [r.retrieval_s for r in ok if r.retrieval_s is not None]
    if ret:
        out["retrieval_s"] = {"p50": percentile(ret, 50), "p95": percentile(ret, 95)}
    if generated:
        costs = [
            (r.input_tokens * price_in_per_mtok + r.output_tokens * price_out_per_mtok) / 1e6
            for r in generated
        ]
        out["cost_per_query_usd"] = sum(costs) / len(costs)
        out["cost_total_usd"] = sum(costs)
        out["tokens_in_mean"] = sum(r.input_tokens for r in generated) / len(generated)
        out["tokens_out_mean"] = sum(r.output_tokens for r in generated) / len(generated)
    return out


# ---------------------------------------------------------------------------
# Relatório em Markdown
# ---------------------------------------------------------------------------

_CONFIG_NAMES = {
    "A": "A · só busca densa",
    "B": "B · híbrida (BM25 + densa)",
    "C": "C · híbrida + reranking",
}


def _fmt_rate(d: dict | None) -> str:
    if not d or d.get("rate") is None:
        return "n/d"
    rate, low, high = d["rate"] * 100, d["ci_low"] * 100, d["ci_high"] * 100
    return f"{rate:.1f}% [{low:.0f}-{high:.0f}] ({d['k']}/{d['n']})"


def _fmt_num(v: float | None, spec: str = ".2f") -> str:
    return "n/d" if v is None else format(v, spec)


def format_markdown(summaries: dict[str, dict]) -> str:
    cfgs = list(summaries)
    lines = [
        "# Crivo, avaliação v2",
        "",
        "Intervalos entre colchetes: Wilson de 95%. Se os intervalos de duas "
        "configurações se sobrepõem, a diferença não é provada por este teste.",
        "",
        "| Métrica | " + " | ".join(_CONFIG_NAMES.get(c, c) for c in cfgs) + " |",
        "|---|" + "---|" * len(cfgs),
    ]

    def row(label: str, fn) -> None:
        lines.append(f"| {label} | " + " | ".join(fn(summaries[c]) for c in cfgs) + " |")

    row(
        "Perguntas com resposta / sem resposta",
        lambda s: f"{s['n_answerable']} / {s['n_unanswerable']}",
    )
    row("Resposta correta (estrita, juiz LLM)", lambda s: _fmt_rate(s.get("answer_correct_strict")))
    row(
        "Resposta correta (aceitando parcial)", lambda s: _fmt_rate(s.get("answer_correct_lenient"))
    )
    row("Abstenção indevida", lambda s: _fmt_rate(s.get("unwarranted_abstention")))
    row("Abstenção correta (D + E)", lambda s: _fmt_rate(s.get("correct_abstention")))
    row(
        "Abstenção correta, tipo D (no contrato, sem a informação)",
        lambda s: _fmt_rate(s.get("correct_abstention_D")),
    )
    row(
        "Abstenção correta, tipo E (fora do domínio)",
        lambda s: _fmt_rate(s.get("correct_abstention_E")),
    )
    row("Citações verificadas", lambda s: _fmt_rate(s.get("citations_verified")))
    row("Recall@5 (evidência nos 5 primeiros)", lambda s: _fmt_rate(s.get("recall_at_5")))
    row("MRR (sobre os 20 candidatos)", lambda s: _fmt_num(s.get("mrr"), ".3f"))
    row(
        "Latência p50 / p95 (s)",
        lambda s: (
            f"{_fmt_num(s['latency_s']['p50'])} / {_fmt_num(s['latency_s']['p95'])}"
            if "latency_s" in s
            else "n/d"
        ),
    )
    row(
        "Só recuperação p50 / p95 (s)",
        lambda s: (
            f"{_fmt_num(s['retrieval_s']['p50'], '.3f')} / "
            f"{_fmt_num(s['retrieval_s']['p95'], '.3f')}"
            if "retrieval_s" in s
            else "n/d"
        ),
    )
    row("Custo médio por consulta (US$)", lambda s: _fmt_num(s.get("cost_per_query_usd"), ".5f"))
    row(
        "Tokens médios (entrada / saída)",
        lambda s: (
            f"{_fmt_num(s.get('tokens_in_mean'), '.0f')} / "
            f"{_fmt_num(s.get('tokens_out_mean'), '.0f')}"
            if "tokens_in_mean" in s
            else "n/d"
        ),
    )
    row("Erros de execução", lambda s: str(s["n_errors"]))
    lines += [
        "",
        "Notas de leitura:",
        "- 'Resposta correta' conta abster-se à toa como erro. O juiz é um LLM comparando "
        "com o gabarito, então vale conferir à mão uma amostra dos vereditos.",
        "- O custo não inclui as chamadas do juiz.",
        "- O preço por milhão de tokens vem dos parâmetros do script, confira na tabela "
        "de preços da Anthropic antes de citar o valor.",
    ]
    return "\n".join(lines) + "\n"
