"""Testes das métricas da avaliação v2 (puro Python, sem modelo e sem API)."""

from __future__ import annotations

import pytest

from rag.eval.metrics_v2 import (
    QuestionRecord,
    format_markdown,
    is_hit,
    parse_pages,
    percentile,
    retrieval_scores,
    summarize,
    wilson_interval,
)


def test_wilson_100_porcento_com_10_nao_prova_muito():
    # É o ponto de partida da v2: 10/10 tem limite inferior perto de 72%.
    low, high = wilson_interval(10, 10)
    assert low == pytest.approx(0.722, abs=0.005)
    assert high == pytest.approx(1.0, abs=1e-9)


def test_wilson_100_porcento_com_40_e_bem_mais_apertado():
    low, _ = wilson_interval(40, 40)
    assert low > 0.91


def test_wilson_total_zero_nao_quebra():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_zero_sucessos():
    low, high = wilson_interval(0, 20)
    assert low == 0.0
    assert 0.1 < high < 0.2


def test_percentile_interpola():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([10], 95) == 10
    assert percentile([], 50) is None
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("3", {3}),
        ("2;3", {2, 3}),
        ("4-6", {4, 5, 6}),
        (" 1 ; 5-6 ", {1, 5, 6}),
        ("", set()),
        ("x", set()),
        ("6-4", set()),
    ],
)
def test_parse_pages(raw, expected):
    assert parse_pages(raw) == expected


def test_is_hit_exige_documento_e_pagina():
    assert is_hit("a.pdf", 2, 4, "a.pdf", {3})
    assert not is_hit("a.pdf", 5, 6, "a.pdf", {3})
    assert not is_hit("b.pdf", 2, 4, "a.pdf", {3})


def test_is_hit_sem_pagina_no_gabarito_basta_o_documento():
    assert is_hit("a.pdf", 9, 9, "a.pdf", set())


def test_retrieval_scores():
    # Evidência na 3ª posição: dentro do top-5, RR = 1/3.
    assert retrieval_scores([False, False, True, False], k=5) == (True, pytest.approx(1 / 3))
    # Na 7ª: fora do top-5, mas o RR ainda conta.
    hits = [False] * 6 + [True]
    recall, rr = retrieval_scores(hits, k=5)
    assert recall is False
    assert rr == pytest.approx(1 / 7)
    assert retrieval_scores([False, False], k=5) == (False, 0.0)


def _rec(**kw) -> QuestionRecord:
    base = dict(config="C", id="X", tipo="A", pergunta="?", abstencao_esperada=False)
    base.update(kw)
    return QuestionRecord(**base)


def test_summarize_conta_abstencao_indevida_como_erro():
    records = [
        _rec(id="1", abstained=False, verdict="correta", citations_total=2, citations_verified=2),
        _rec(id="2", abstained=False, verdict="incorreta", citations_total=1, citations_verified=1),
        _rec(id="3", abstained=True),  # abstenção indevida
        _rec(id="4", abstained=False, verdict="parcial", citations_total=1, citations_verified=1),
    ]
    s = summarize(records, 1.0, 5.0)
    assert s["n_answerable"] == 4
    assert s["unwarranted_abstention"]["k"] == 1
    assert s["answer_correct_strict"]["k"] == 1 and s["answer_correct_strict"]["n"] == 4
    assert s["answer_correct_lenient"]["k"] == 2
    assert s["citations_verified"]["k"] == 4 and s["citations_verified"]["n"] == 4


def test_summarize_abstencao_correta_por_tipo():
    records = [
        _rec(id="d1", tipo="D", abstencao_esperada=True, abstained=True),
        _rec(id="d2", tipo="D", abstencao_esperada=True, abstained=False),  # alucinou
        _rec(id="e1", tipo="E", abstencao_esperada=True, abstained=True),
    ]
    s = summarize(records, 1.0, 5.0)
    assert s["correct_abstention"]["k"] == 2 and s["correct_abstention"]["n"] == 3
    assert s["correct_abstention_D"]["rate"] == pytest.approx(0.5)
    assert s["correct_abstention_E"]["rate"] == pytest.approx(1.0)


def test_summarize_recuperacao_so_nas_perguntas_com_resposta():
    records = [
        _rec(id="1", hit_at_5=True, reciprocal_rank=1.0),
        _rec(id="2", hit_at_5=False, reciprocal_rank=0.25),
        _rec(id="3", tipo="D", abstencao_esperada=True),  # sem gabarito de página
    ]
    s = summarize(records, 1.0, 5.0)
    assert s["recall_at_5"]["k"] == 1 and s["recall_at_5"]["n"] == 2
    assert s["mrr"] == pytest.approx(0.625)


def test_summarize_custo_e_latencia():
    records = [
        _rec(id="1", abstained=False, latency_s=2.0, input_tokens=4000, output_tokens=200),
        _rec(id="2", abstained=True, latency_s=0.5),  # abstenção por limiar: custo zero
    ]
    s = summarize(records, 1.0, 5.0)
    # (4000*1 + 200*5)/1e6 = 0.005 na primeira; 0 na segunda.
    assert s["cost_total_usd"] == pytest.approx(0.005)
    assert s["cost_per_query_usd"] == pytest.approx(0.0025)
    assert s["latency_s"]["p50"] == pytest.approx(1.25)


def test_summarize_dry_run_nao_inventa_metricas_de_geracao():
    records = [_rec(id="1", hit_at_5=True, reciprocal_rank=1.0, retrieval_s=0.2)]
    s = summarize(records, 1.0, 5.0)
    assert "unwarranted_abstention" not in s
    assert "cost_per_query_usd" not in s
    assert s["recall_at_5"]["rate"] == 1.0


def test_summarize_ignora_registros_com_erro():
    records = [_rec(id="1", abstained=True, error="Timeout"), _rec(id="2", abstained=False)]
    s = summarize(records, 1.0, 5.0)
    assert s["n_errors"] == 1
    assert s["n_answerable"] == 1


def test_markdown_tem_uma_coluna_por_configuracao():
    s = summarize([_rec(id="1", abstained=False, verdict="correta")], 1.0, 5.0)
    md = format_markdown({"B": s, "C": s})
    assert "B · híbrida" in md and "C · híbrida + reranking" in md
    assert "Wilson" in md
