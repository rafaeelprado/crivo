"""Testes do chunking estrutural — critério de aceite da Fase 2.

Rodam sobre um contrato sintético em tests/fixtures, com a hierarquia real de um
contrato administrativo brasileiro. Não dependem do corpus, então valem como
rede de segurança desde já.
"""

from pathlib import Path

import pytest

from rag.ingest.chunker import chunk_document, estimate_tokens, find_headings
from rag.ingest.parser import parse_text

FIXTURE = Path(__file__).parent / "fixtures" / "contrato_exemplo.txt"


@pytest.fixture(scope="module")
def doc():
    return parse_text("contrato_exemplo.txt", FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def resultado(doc):
    return chunk_document(doc, doc_type="contrato_administrativo", max_tokens=1200)


# --- detecção de hierarquia ---------------------------------------------------


def test_detecta_todas_as_clausulas(doc):
    marcadores = [h.marker for h in find_headings(doc.text) if h.level == 1]
    assert any("PRIMEIRA" in m.upper() for m in marcadores)
    assert any("DÉCIMA SEGUNDA" in m.upper() for m in marcadores)
    assert len(marcadores) >= 7


def test_detecta_paragrafos_incisos_e_alineas(doc):
    kinds = {h.kind for h in find_headings(doc.text)}
    assert {"clausula", "paragrafo", "inciso", "alinea"} <= kinds


def test_paragrafo_unico_e_reconhecido(doc):
    marcadores = [h.marker.upper() for h in find_headings(doc.text)]
    assert any("PARÁGRAFO ÚNICO" in m for m in marcadores)


def test_nao_confunde_referencia_no_meio_do_texto_com_cabecalho(doc):
    # "...previstas na Cláusula Quarta." é referência, não abertura de cláusula.
    aberturas = [h for h in find_headings(doc.text) if h.kind == "clausula"]
    assert all(doc.text[h.offset : h.offset + 8].upper().startswith("CLÁUSUL") for h in aberturas)


def test_nao_confunde_citacao_de_lei_quebrada_de_linha_com_artigo():
    """Regressão: bug real achado no corpus (clause_path 'art. 137 d',
    'art. 94 d').

    Uma citação como "...conforme o disposto no art. 137 da Lei 14.133/2021"
    pode quebrar de linha bem em "art. 137" por causa do reflow do PDF — a
    linha resultante começa exatamente como um cabeçalho de artigo de
    verdade começaria, e o regex (case-insensitive) ainda captura a letra
    minúscula seguinte ("da") como se fosse o sufixo de um artigo tipo
    "137-A". O resultado sem a correção era um clause_path fantasma como
    'art. 137 d', que nunca existiu como cabeçalho no contrato.
    """
    texto = (
        "CLÁUSULA DÉCIMA TERCEIRA — DA MULTA\n\n"
        "A CONTRATADA sujeita-se à multa moratória prevista neste contrato,\n"
        "sem prejuízo das sanções administrativas cabíveis conforme o disposto no\n"
        "art. 137 da Lei nº 14.133, de 2021, e demais normas aplicáveis à espécie."
    )
    doc = parse_text("citacao_quebrada.txt", texto)
    marcadores_artigo = [h for h in find_headings(doc.text) if h.kind == "artigo"]
    assert not marcadores_artigo, f"citação quebrada virou cabeçalho: {marcadores_artigo}"


def test_ainda_detecta_artigo_de_verdade_apos_linha_em_branco():
    """A correção da citação quebrada não pode custar a detecção real."""
    texto = "CLÁUSULA DÉCIMA NONA\n\nArt. 78. A rescisão observará o contraditório."
    doc = parse_text("artigo_real.txt", texto)
    marcadores_artigo = [h.marker for h in find_headings(doc.text) if h.kind == "artigo"]
    assert any("78" in m for m in marcadores_artigo)


def test_ainda_detecta_artigo_de_verdade_como_primeira_linha_do_documento():
    texto = "Art. 1º Este regulamento se aplica a todos os contratos administrativos."
    doc = parse_text("artigo_primeira_linha.txt", texto)
    marcadores_artigo = [h.marker for h in find_headings(doc.text) if h.kind == "artigo"]
    assert any("1" in m for m in marcadores_artigo)


# --- qualidade dos chunks -----------------------------------------------------


def test_todo_chunk_tem_clause_path(resultado):
    chunks, report = resultado
    assert report.clause_path_rate >= 0.9, f"apenas {report.clause_path_rate:.0%} com clause_path"


def test_nenhum_chunk_estoura_o_teto(resultado):
    chunks, report = resultado
    assert report.oversized_chunks == 0
    assert report.max_tokens_seen <= 1200


def test_nenhum_fallback_num_documento_bem_formatado(resultado):
    _, report = resultado
    assert report.fallback_chunks == 0


def test_texto_do_chunk_e_fatia_literal_do_documento(doc, resultado):
    """A invariante que sustenta a verificação de citação da Fase 4."""
    chunks, _ = resultado
    for chunk in chunks:
        assert chunk.text in doc.text, f"{chunk.chunk_id} não é fatia literal"
        assert doc.text[chunk.char_start : chunk.char_end] == chunk.text


def test_a_clausula_de_rescisao_traz_a_regra_inteira(resultado):
    """Teste de regressão contra o erro clássico: cortar a regra ao meio."""
    chunks, _ = resultado
    rescisao = [c for c in chunks if "DÉCIMA SEGUNDA" in c.clause_path.upper()]
    assert rescisao, "cláusula de rescisão não virou chunk"
    texto = " ".join(c.text for c in rescisao)
    assert "acordo entre as partes" in texto
    assert "art. 78" in texto
    assert "indenização" in texto


def test_a_clausula_de_penalidades_mantem_multa_junto_da_condicao(resultado):
    chunks, _ = resultado
    penalidades = [c for c in chunks if "QUARTA" in c.clause_path.upper()]
    texto = " ".join(c.text for c in penalidades)
    assert "0,5%" in texto and "por dia de atraso" in texto
    assert "15%" in texto


def test_preambulo_nao_e_descartado(resultado):
    chunks, _ = resultado
    preambulo = [c for c in chunks if c.clause_path == "PREÂMBULO"]
    assert preambulo
    assert "CONSTRUTORA MODELO" in preambulo[0].text


def test_clause_path_e_hierarquico_quando_ha_subdivisao(resultado):
    chunks, _ = resultado
    caminhos = [c.clause_path for c in chunks]
    # Ou a cláusula coube inteira, ou os filhos aparecem com caminho composto.
    assert all(">" in c or c.count("CLÁUSULA") <= 1 or c == "PREÂMBULO" for c in caminhos)


def test_chunk_ids_sao_unicos(resultado):
    chunks, _ = resultado
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


# --- fallback -----------------------------------------------------------------


def test_documento_sem_hierarquia_cai_no_fallback():
    texto = "\n\n".join(f"Parágrafo solto número {i} sem nenhuma estrutura." for i in range(40))
    doc = parse_text("sem_estrutura.txt", texto)
    chunks, report = chunk_document(doc, max_tokens=200)
    assert report.fallback_chunks == report.total_chunks
    assert all(c.fallback for c in chunks)
    assert report.clause_path_rate == 0.0


def test_bloco_muito_grande_e_dividido_em_fatias_contiguas():
    corpo = " ".join(["palavra"] * 4000)
    doc = parse_text("gigante.txt", f"CLÁUSULA PRIMEIRA — DO OBJETO\n\n{corpo}")
    chunks, report = chunk_document(doc, max_tokens=300)
    assert len(chunks) > 1
    assert report.oversized_chunks == 0
    for chunk in chunks:
        assert chunk.text in doc.text


def test_texto_de_abertura_longo_antes_do_primeiro_paragrafo_tambem_e_dividido():
    """Regressão: o texto entre o cabeçalho da cláusula e o primeiro § era
    adicionado como um chunk único, sem passar pelo splitter — se fosse maior
    que o teto, estourava sozinho. Só aparece quando a cláusula tem um § (ou
    outro filho de nível 2), senão cai no ramo 'sem filhos' que já divide.
    """
    abertura = " ".join(["cláusula longa antes do parágrafo"] * 400)
    texto = (
        f"CLÁUSULA SEGUNDA — DO VALOR\n\n{abertura}\n\n"
        "§ 1º Disposição final e curta."
    )
    doc = parse_text("abertura_longa.txt", texto)
    chunks, report = chunk_document(doc, max_tokens=300)
    assert report.oversized_chunks == 0
    assert all(estimate_tokens(c.text) <= 300 for c in chunks)
    for chunk in chunks:
        assert chunk.text in doc.text


# --- utilitários --------------------------------------------------------------


def test_estimativa_de_tokens_e_monotonica():
    assert estimate_tokens("abc") < estimate_tokens("abc" * 100)
    assert estimate_tokens("") >= 1
