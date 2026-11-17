#!/usr/bin/env python
"""Avaliação v2 do Crivo: 100 perguntas, 3 configurações, intervalo de confiança.

Configurações (as mesmas de `run_eval.py`, da Fase 7):
  A  só busca densa, sem reranking
  B  busca híbrida (BM25 + densa), sem reranking
  C  busca híbrida + reranking

O que mede, por configuração:
  - resposta correta (juiz LLM contra o gabarito) e abstenção indevida;
  - abstenção correta nas perguntas sem resposta (tipo D: o contrato existe mas
    não tem a informação; tipo E: fora do domínio);
  - citações verificadas;
  - recall@5 e MRR da recuperação (documento certo e página de evidência);
  - latência p50/p95, tokens e custo por consulta.

Uso (com o ambiente do projeto ativado e o índice já construído):

  # 1) Só recuperação, sem chamar a API (grátis, valida o gabarito e o índice):
  python scripts/run_eval_v2.py --dry-run

  # 2) Teste rápido com 10 perguntas:
  python scripts/run_eval_v2.py --config C --limit 10

  # 3) Avaliação completa (A, B e C):
  python scripts/run_eval_v2.py

A execução é retomável: o que já foi gravado em data/eval_v2/ não é repetido,
então uma queda de rede não faz você pagar duas vezes pela mesma pergunta.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag.eval.metrics_v2 import (  # noqa: E402
    QuestionRecord,
    format_markdown,
    is_hit,
    parse_pages,
    retrieval_scores,
    summarize,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("eval_v2")

CONFIGS = ("A", "B", "C")

JUDGE_SYSTEM = (
    "Você corrige respostas de um assistente de análise de contratos públicos "
    "brasileiros. Compare a RESPOSTA DO SISTEMA com o GABARITO e dê um veredito.\n"
    "- correta: traz a informação central do gabarito (valor, prazo, cláusula, "
    "parte etc.) sem contradizê-lo. Detalhes extras corretos são aceitáveis.\n"
    "- parcial: acerta parte do que o gabarito pede ou omite um item importante, "
    "sem afirmar nada errado.\n"
    "- incorreta: erra, contradiz o gabarito ou não traz a informação central.\n"
    "Não use conhecimento próprio sobre o contrato: o gabarito é a verdade."
)
JUDGE_TOOL = {
    "name": "veredito",
    "description": "Registra o veredito da correção.",
    "input_schema": {
        "type": "object",
        "properties": {
            "veredito": {"type": "string", "enum": ["correta", "parcial", "incorreta"]},
            "justificativa": {"type": "string", "description": "Uma frase."},
        },
        "required": ["veredito", "justificativa"],
    },
}


# ---------------------------------------------------------------------------
# Peças que falam com a API (import tardio: o módulo importa sem `anthropic`)
# ---------------------------------------------------------------------------


class UsageCaller:
    """ModelCaller igual ao de produção, mas que soma os tokens de cada chamada.

    O `get_default_model_caller()` do projeto devolve só o input da tool_use e
    descarta `response.usage`. Aqui repetimos a mesma chamada (mesmo modelo,
    mesma tool forçada, mesmos MAX_TOKENS) para poder medir custo.
    """

    def __init__(self) -> None:
        import anthropic

        from rag.generation.client import API_TIMEOUT, MAX_TOKENS
        from rag.generation.schema import ANSWER_TOOL, ANSWER_TOOL_NAME
        from rag.settings import settings

        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key, timeout=API_TIMEOUT)
        self._model = settings.answer_model
        self._max_tokens = MAX_TOKENS
        self._tool = ANSWER_TOOL
        self._tool_name = ANSWER_TOOL_NAME
        self.reset()

    def reset(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    def __call__(self, system: str, messages: list[dict]) -> dict:
        last: Exception | None = None
        for attempt in range(1, 4):
            try:
                resp = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=system,
                    tools=[self._tool],
                    tool_choice={"type": "tool", "name": self._tool_name},
                    messages=messages,
                )
                self.input_tokens += resp.usage.input_tokens
                self.output_tokens += resp.usage.output_tokens
                for block in resp.content:
                    if block.type == "tool_use":
                        return block.input
                raise RuntimeError("resposta sem bloco tool_use")
            except Exception as exc:  # noqa: BLE001 - repete e, no fim, propaga
                last = exc
                if attempt == 3:
                    raise
                time.sleep(2.0 ** (attempt - 1))
        raise last or RuntimeError("erro desconhecido")


class Judge:
    """Juiz LLM com cache em disco (reexecutar não paga duas vezes)."""

    def __init__(self, cache_path: Path) -> None:
        import anthropic

        from rag.settings import settings

        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key, timeout=60.0)
        self._model = settings.answer_model
        self._cache_path = cache_path
        self._cache: dict[str, dict] = {}
        if cache_path.exists():
            self._cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def __call__(self, qid: str, pergunta: str, gabarito: str, resposta: str) -> str:
        key = f"{qid}|{hashlib.sha1(resposta.encode('utf-8')).hexdigest()[:12]}"
        if key in self._cache:
            return self._cache[key]["veredito"]
        prompt = (
            f"PERGUNTA:\n{pergunta}\n\nGABARITO:\n{gabarito}\n\nRESPOSTA DO SISTEMA:\n{resposta}"
        )
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=300,
            system=JUDGE_SYSTEM,
            tools=[JUDGE_TOOL],
            tool_choice={"type": "tool", "name": "veredito"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in resp.content:
            if block.type == "tool_use":
                self._cache[key] = dict(block.input)
                self._cache_path.write_text(
                    json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                return block.input["veredito"]
        raise RuntimeError("juiz não devolveu veredito")


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------


def load_golden(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    needed = {
        "id",
        "tipo",
        "pergunta",
        "resposta_esperada",
        "documento_fonte",
        "paginas_evidencia",
        "abstencao_esperada",
    }
    missing = needed - set(rows[0].keys()) if rows else needed
    if missing:
        raise SystemExit(f"O golden set não tem as colunas: {sorted(missing)}")
    return rows


def retrieval_for(question: dict, config: str) -> tuple[bool | None, float | None, float]:
    """Roda só a recuperação da configuração e mede onde a evidência apareceu."""
    from rag.retrieval.hybrid import search
    from rag.retrieval.rerank import rerank

    t0 = time.perf_counter()
    candidates = search(question["pergunta"], dense_only=(config == "A"))
    if config == "C":
        ordered = [
            rc.chunk for rc in rerank(question["pergunta"], candidates, top_n=len(candidates))
        ]
    else:
        ordered = list(candidates)
    elapsed = time.perf_counter() - t0

    if question["abstencao_esperada"].strip().lower() == "sim":
        return None, None, elapsed
    expected_doc = question["documento_fonte"]
    expected_pages = parse_pages(question["paginas_evidencia"])
    hits = [
        is_hit(c.document, c.page_start, c.page_end, expected_doc, expected_pages) for c in ordered
    ]
    recall, rr = retrieval_scores(hits, k=5)
    return recall, rr, elapsed


def run_question(
    config: str,
    question: dict,
    *,
    caller: UsageCaller | None,
    judge: Judge | None,
) -> QuestionRecord:
    expects_abstention = question["abstencao_esperada"].strip().lower() == "sim"
    rec = QuestionRecord(
        config=config,
        id=question["id"],
        tipo=question["tipo"],
        pergunta=question["pergunta"],
        abstencao_esperada=expects_abstention,
    )
    try:
        rec.hit_at_5, rec.reciprocal_rank, rec.retrieval_s = retrieval_for(question, config)

        if caller is None:  # --dry-run: só recuperação
            return rec

        from rag.generation.answer import answer_question

        caller.reset()
        t0 = time.perf_counter()
        result = answer_question(
            question["pergunta"],
            model_caller=caller,
            log=False,  # não polui data/logs/queries.jsonl, que alimenta /metrics
            disable_reranking=(config in ("A", "B")),
            dense_only=(config == "A"),
        )
        rec.latency_s = time.perf_counter() - t0
        rec.input_tokens = caller.input_tokens
        rec.output_tokens = caller.output_tokens
        rec.abstained = bool(result.abstained or result.insufficient_context)
        rec.answer = result.answer or ""
        rec.retries = result.retries
        rec.retrieval_top_score = result.retrieval_top_score
        rec.citations_total = len(result.citations)
        rec.citations_verified = sum(1 for c in result.citations if c.verified)

        if judge is not None and not expects_abstention and not rec.abstained:
            rec.verdict = judge(
                question["id"], question["pergunta"], question["resposta_esperada"], rec.answer
            )
    except Exception as exc:  # noqa: BLE001 - uma pergunta com erro não derruba a rodada
        log.error("[%s] %s falhou: %s", config, question["id"], exc)
        rec.error = f"{type(exc).__name__}: {exc}"
    return rec


def load_done(path: Path) -> dict[str, QuestionRecord]:
    done: dict[str, QuestionRecord] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            data = json.loads(line)
            rec = QuestionRecord(**data)
            if rec.error is None:  # erro volta para a fila
                done[rec.id] = rec
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação v2 do Crivo")
    parser.add_argument("--config", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--golden", type=Path, default=ROOT / "data" / "golden_set_v2.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "eval_v2")
    parser.add_argument("--limit", type=int, default=None, help="só as N primeiras (teste rápido)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="só recuperação (recall, MRR, latência); não chama a API",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="gera respostas mas não chama o juiz (sem 'resposta correta')",
    )
    parser.add_argument(
        "--price-in",
        type=float,
        default=1.0,
        help="US$ por milhão de tokens de entrada (CONFIRA na tabela da Anthropic)",
    )
    parser.add_argument(
        "--price-out",
        type=float,
        default=5.0,
        help="US$ por milhão de tokens de saída (CONFIRA na tabela da Anthropic)",
    )
    args = parser.parse_args()

    from rag.settings import settings

    questions = load_golden(args.golden)
    if args.limit:
        # Mantém a mistura de tipos: pega fatias iguais de cada tipo.
        tipos = sorted({q["tipo"] for q in questions})
        per = max(1, args.limit // len(tipos))
        picked: list[dict] = []
        for t in tipos:
            picked += [q for q in questions if q["tipo"] == t][:per]
        questions = picked
    log.info(
        "%d perguntas, configurações %s%s",
        len(questions),
        args.config,
        " (dry-run)" if args.dry_run else "",
    )

    raw_dir = settings.raw_dir
    missing_docs = sorted(
        {
            q["documento_fonte"]
            for q in questions
            if q["documento_fonte"] and not (raw_dir / q["documento_fonte"]).exists()
        }
    )
    if missing_docs:
        log.warning("Documentos do gabarito que não estão em data/raw: %s", missing_docs)

    caller: UsageCaller | None = None
    judge: Judge | None = None
    if not args.dry_run:
        if not settings.has_api_key:
            raise SystemExit(
                "ANTHROPIC_API_KEY não configurada no .env (use --dry-run para só recuperação)."
            )
        caller = UsageCaller()
        args.out.mkdir(parents=True, exist_ok=True)
        if not args.no_judge:
            judge = Judge(args.out / "judge_cache.json")
    args.out.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}
    for config in args.config:
        path = args.out / f"results_{config}{'_dry' if args.dry_run else ''}.jsonl"
        done = load_done(path)
        log.info("=== Configuração %s (%d já feitas) ===", config, len(done))
        with path.open("a", encoding="utf-8") as fh:
            for i, q in enumerate(questions, start=1):
                if q["id"] in done:
                    continue
                log.info("[%s] %d/%d %s", config, i, len(questions), q["id"])
                rec = run_question(config, q, caller=caller, judge=judge)
                fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
                fh.flush()
                done[q["id"]] = rec if rec.error is None else done.get(q["id"], rec)
        # Relê tudo (inclusive erros pendentes) para o resumo refletir o arquivo.
        records = [
            QuestionRecord(**json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        latest: dict[str, QuestionRecord] = {}
        for r in records:
            latest[r.id] = r  # a última linha de cada id vale
        wanted = {q["id"] for q in questions}
        summaries[config] = summarize(
            [r for r in latest.values() if r.id in wanted], args.price_in, args.price_out
        )

    suffix = "_dry" if args.dry_run else ""
    (args.out / f"summary{suffix}.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md = format_markdown(summaries)
    (args.out / f"summary{suffix}.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    log.info("Resultados em %s", args.out)


if __name__ == "__main__":
    main()
