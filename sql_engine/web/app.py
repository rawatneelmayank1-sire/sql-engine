"""FastAPI application: the pipeline, visible.

The UI's whole job is to leave nothing implicit. For every request it reports what each
compilation stage produced, what the verifier decided and why, both query plans, and the
rows -- including the attempts that were *refused*, which are the interesting ones.

Two entry points, deliberately:

* ``/api/ask`` runs the full text-to-SQL pipeline.
* ``/api/verify`` takes raw SQL straight from the user. This one exists so a visitor can
  type ``DROP TABLE trades`` themselves and watch it stop, rather than taking the
  README's word for it.

With no ``GROQ_API_KEY`` the app falls back to the committed fixtures, so the demo runs
on any machine. It says so in the header rather than pretending to have a model.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from ..engine import CompiledQuery, Engine
from ..errors import SQLEngineError, TokeniseError
from ..llm import (
    Answer,
    FailureCategory,
    GroqClient,
    LLMClient,
    Prompt,
    ReplayClient,
    TextToSQL,
    VerificationResult,
    Verifier,
)
from ..plan import HashJoin, LogicalPlan, Scan, iter_plan
from ..tokeniser import tokenise

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
STATIC = PACKAGE_ROOT / "static"
WAREHOUSE = PROJECT_ROOT / "data" / "warehouse"
EVALUATION = PROJECT_ROOT / "evaluation"

#: Stage names in pipeline order. This list is the UI's spine: the rail across the top
#: is these six, in this order, and a refusal is reported as "stopped at stage N".
STAGES = ["tokenise", "parse", "bind", "plan", "verify", "execute"]

#: Which stage each failure category belongs to. Reporting "refused at BIND" instead of
#: "verification failed" is the difference between a demo that explains itself and one
#: that asks to be believed.
CATEGORY_STAGE: dict[FailureCategory, str] = {
    # The comment check is lexical and runs on the raw text before the parser sees it.
    FailureCategory.COMMENT_TOKEN: "tokenise",
    FailureCategory.NOT_SELECT: "parse",
    FailureCategory.UNPARSEABLE: "parse",
    FailureCategory.UNKNOWN_TABLE: "bind",
    FailureCategory.UNKNOWN_COLUMN: "bind",
    FailureCategory.AMBIGUOUS_COLUMN: "bind",
    FailureCategory.INVALID_AGGREGATION: "bind",
    FailureCategory.CARTESIAN_PRODUCT: "verify",
    FailureCategory.SCAN_TOO_LARGE: "verify",
    FailureCategory.INTERNAL: "verify",
}

MAX_DISPLAY_ROWS = 200


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class VerifyRequest(BaseModel):
    sql: str = Field(min_length=0, max_length=8000)
    execute: bool = True


@dataclass(slots=True)
class FixtureQuestionClient:
    """Serves recorded responses for the golden-set questions; refuses anything else.

    A demo that silently invented answers for unknown questions would be exactly the
    kind of quiet dishonesty this project is arguing against, so an unrecognised
    question gets a plain error naming the reason.
    """

    replay: ReplayClient
    question_ids: dict[str, str]
    name: str = "fixtures (no GROQ_API_KEY set)"

    def complete(self, prompt: Prompt) -> str:
        return self.replay.complete(prompt)

    def resolve(self, question: str) -> str | None:
        return self.question_ids.get(_normalise(question))


def _normalise(question: str) -> str:
    return " ".join(question.lower().split()).rstrip("?").strip()


def _load_golden() -> list[dict[str, str]]:
    path = EVALUATION / "golden_set.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_client(engine: Engine) -> tuple[LLMClient, FixtureQuestionClient | None, str]:
    """Prefer a live model; fall back to fixtures so the demo always runs."""
    groq = GroqClient()
    if groq.available:
        return groq, None, groq.name
    fixtures = EVALUATION / "fixtures" / "reference.json"
    replay = ReplayClient.from_file(fixtures)
    lookup = {_normalise(row["question"]): row["id"] for row in _load_golden()}
    client = FixtureQuestionClient(replay=replay, question_ids=lookup)
    return client, client, client.name


def create_app(warehouse: Path | None = None) -> FastAPI:
    engine = Engine.from_directory(warehouse or WAREHOUSE)
    verifier = Verifier(engine)
    client, fixture_client, model_name = build_client(engine)
    pipeline = TextToSQL(engine, client, verifier=verifier)
    golden = _load_golden()

    app = FastAPI(title="SQL Engine", docs_url=None, redoc_url=None)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/schema")
    def schema() -> dict[str, Any]:
        return {
            "model": model_name,
            "live_model": fixture_client is None,
            "tables": [
                {
                    "name": table.name,
                    "format": table.file_format,
                    "rows": table.row_count,
                    "columns": [
                        {"name": column.name, "type": column.dtype.value}
                        for column in table.columns
                    ],
                }
                for table in sorted(engine.catalog, key=lambda t: t.name)
            ],
            "examples": [row["question"] for row in golden[:12]],
            "attack_examples": [
                "DROP TABLE trades",
                "SELECT * FROM trades; DROP TABLE trades;",
                "DELETE FROM traders WHERE 1 = 1",
                "SELECT lifetime_value FROM traders",
                "SELECT t.trade_id FROM trades t INNER JOIN instruments i ON 1 = 1",
                "SELECT side, price, COUNT(*) FROM trades GROUP BY side",
            ],
        }

    @app.post("/api/ask")
    def ask(request: AskRequest) -> JSONResponse:
        question = request.question.strip()
        key = "adhoc"
        if fixture_client is not None:
            resolved = fixture_client.resolve(question)
            if resolved is None:
                return JSONResponse(
                    status_code=200,
                    content={
                        "ok": False,
                        "no_model": True,
                        "message": (
                            "No language model is configured, so only the recorded "
                            "questions can be answered. Pick one below, or set "
                            "GROQ_API_KEY and restart."
                        ),
                    },
                )
            key = resolved

        started = time.perf_counter()
        answer = pipeline.answer(question, key=key)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return JSONResponse(content=_render_answer(answer, elapsed_ms))

    @app.post("/api/verify")
    def verify(request: VerifyRequest) -> JSONResponse:
        sql = request.sql
        started = time.perf_counter()
        verdict = verifier.verify(sql)
        result = None
        if verdict.ok and request.execute and verdict.compiled is not None:
            try:
                result = engine.execute(verdict.compiled)
            except SQLEngineError as error:
                return JSONResponse(
                    content={
                        "ok": False,
                        "sql": verdict.sql,
                        "stages": _stages_for(sql, verdict, None),
                        "failures": [
                            {
                                "category": "runtime",
                                "message": str(error),
                                "stage": "execute",
                            }
                        ],
                        "repairs": [],
                        "elapsed_ms": (time.perf_counter() - started) * 1000,
                    }
                )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return JSONResponse(content=_render_verdict(sql, verdict, result, elapsed_ms))

    return app


# --------------------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------------------


def _token_count(sql: str) -> int | None:
    try:
        return max(0, len(tokenise(sql)) - 1)  # drop the EOF token
    except TokeniseError:
        return None


def _plan_summary(plan: LogicalPlan) -> dict[str, int]:
    nodes = list(iter_plan(plan))
    return {
        "scans": sum(1 for n in nodes if isinstance(n, Scan)),
        "joins": sum(1 for n in nodes if isinstance(n, HashJoin)),
        "nodes": len(nodes),
    }


def _stages_for(
    sql: str, verdict: VerificationResult, compiled: CompiledQuery | None
) -> list[dict[str, Any]]:
    """Per-stage status and one measured readout each.

    Stages after a failure are ``skipped`` rather than ``failed``: the pipeline stopped,
    it did not try and fail six times. That distinction is the point of showing the rail
    at all.
    """
    tokens = _token_count(sql)
    failed_stage = (
        CATEGORY_STAGE.get(verdict.failures[0].category) if verdict.failures else None
    )
    if tokens is None:
        failed_stage = "tokenise"

    readouts: dict[str, str] = {"tokenise": f"{tokens} tokens" if tokens is not None else "-"}
    if compiled is not None:
        readouts["parse"] = "SELECT"
        count = len(compiled.output_names)
        readouts["bind"] = f"{count} column" + ("s" if count != 1 else "")
        summary = _plan_summary(compiled.optimised_plan)
        readouts["plan"] = (
            f"{summary['scans']} scan" + ("s" if summary["scans"] != 1 else "")
            + (f", {summary['joins']} join" if summary["joins"] else "")
        )
        readouts["verify"] = f"{verdict.estimated_scan_bytes / 1e6:.1f} MB est."

    stages: list[dict[str, Any]] = []
    stopped = False
    for name in STAGES:
        if stopped:
            status = "skipped"
        elif name == failed_stage:
            status = "refused"
            stopped = True
        else:
            status = "passed"
        stages.append({"name": name, "status": status, "readout": readouts.get(name, "")})
    return stages


def _render_rows(result: Any) -> dict[str, Any]:
    if result is None:
        return {"columns": [], "rows": [], "truncated": False, "total": 0}
    rows = [list(row) for row in result.rows[:MAX_DISPLAY_ROWS]]
    return {
        "columns": list(result.column_names),
        "rows": rows,
        "truncated": len(result.rows) > MAX_DISPLAY_ROWS,
        "total": len(result.rows),
        "stats": {
            "rows_scanned": result.stats.rows_scanned,
            "rows_after_scan_filter": result.stats.rows_after_scan_filter,
            "columns_read": result.stats.columns_read,
        },
    }


def _render_plans(compiled: CompiledQuery | None) -> dict[str, Any]:
    if compiled is None:
        return {"before": "", "after": "", "summary": ""}
    return {
        "before": compiled.logical_plan.explain(compiled.estimator),
        "after": compiled.optimised_plan.explain(compiled.estimator),
        "summary": compiled.report.summary(),
        "estimated_rows": compiled.estimated_rows,
    }


def _render_verdict(
    sql: str, verdict: VerificationResult, result: Any, elapsed_ms: float
) -> dict[str, Any]:
    return {
        "ok": verdict.ok,
        "sql": verdict.sql,
        "original_sql": verdict.original_sql,
        "stages": _stages_for(sql, verdict, verdict.compiled),
        "failures": [
            {
                "category": failure.category.value,
                "description": failure.category.description,
                "message": failure.message,
                "stage": CATEGORY_STAGE.get(failure.category, "verify"),
            }
            for failure in verdict.failures
        ],
        "repairs": [{"kind": r.kind, "detail": r.detail} for r in verdict.repairs],
        "plan": _render_plans(verdict.compiled),
        "result": _render_rows(result),
        "elapsed_ms": elapsed_ms,
    }


def _render_answer(answer: Answer, elapsed_ms: float) -> dict[str, Any]:
    final = answer.attempts[-1].verification if answer.attempts else None
    payload = {
        "ok": answer.ok,
        "question": answer.question,
        "sql": answer.sql,
        "error": answer.error,
        "elapsed_ms": elapsed_ms,
        "attempts": [
            {
                "index": attempt.index,
                "sql": attempt.sql,
                "ok": attempt.ok,
                "failures": [
                    {
                        "category": failure.category.value,
                        "description": failure.category.description,
                        "message": failure.message,
                        "stage": CATEGORY_STAGE.get(failure.category, "verify"),
                    }
                    for failure in attempt.verification.failures
                ],
                "repairs": [
                    {"kind": r.kind, "detail": r.detail}
                    for r in attempt.verification.repairs
                ],
            }
            for attempt in answer.attempts
        ],
        "repair_count": answer.repair_count,
    }
    if final is not None:
        payload["stages"] = _stages_for(
            answer.attempts[-1].sql, final, answer.compiled
        )
        payload["failures"] = payload["attempts"][-1]["failures"]
        payload["repairs"] = payload["attempts"][-1]["repairs"]
    payload["plan"] = _render_plans(answer.compiled)
    payload["result"] = _render_rows(answer.result)
    return payload


app = create_app()
