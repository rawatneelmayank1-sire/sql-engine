"""Build the data behind the static demo in ``web/`` by running the real pipeline.

GitHub Pages serves static files, so the published demo cannot run Python. Rather than
re-implement anything in JavaScript, this script drives the real FastAPI app in-process
and saves the exact JSON each endpoint returns:

* ``GET  /api/schema``                     -> ``web/data/schema.js``
* ``POST /api/ask``    for all 45 questions -> ``web/data/answers.js``
* ``POST /api/verify`` for a curated set    -> ``web/data/queries.js``

Every stage readout, refusal, plan and row in the static demo is therefore the engine's
own output. The page only looks results up.

**No model is called and no key is read.** Language-model answers are replayed from
``evaluation/fixtures/recorded.json`` (real responses recorded from Groq). Before the app
is imported, every ``*_API_KEY`` variable is removed from this process's environment and
DNS resolution of anything but localhost is disabled, so a live call cannot happen even
by accident.

Each answer also carries the evaluation's verdict for that question -- does the verified
answer return the same rows as the hand-written gold query? -- taken from
``evaluation/run_eval.py`` itself, so the demo cannot disagree with
``evaluation/results.md``.

The files are JavaScript that assign to ``window.SQL_ENGINE_DEMO`` rather than JSON, so
the page also works when opened straight from disk (``file://`` forbids ``fetch``).

Run:  python scripts/export_static_demo.py
"""

from __future__ import annotations

import json
import os
import socket
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
WAREHOUSE = ROOT / "data" / "warehouse"
RECORDED = ROOT / "evaluation" / "fixtures" / "recorded.json"
OUTPUT = ROOT / "web" / "data"
REPOSITORY = "https://github.com/rawatneelmayank1-sire/sql-engine"
MAX_DATA_BYTES = 5 * 1024 * 1024

#: The "Write SQL yourself" examples, in display order: refusals first, because watching
#: a query stop is the point of the demo. ``halts_at`` is asserted against the engine's
#: actual verdict, so a label can never drift from what the page shows.
QUERY_EXAMPLES: list[dict[str, str | None]] = [
    {"kind": "refusal", "halts_at": "parse", "why": "destructive statement",
     "sql": "DROP TABLE trades"},
    {"kind": "refusal", "halts_at": "parse", "why": "stacked statements",
     "sql": "SELECT * FROM trades; DROP TABLE trades;"},
    {"kind": "refusal", "halts_at": "parse", "why": "destructive statement",
     "sql": "DELETE FROM traders WHERE 1 = 1"},
    {"kind": "refusal", "halts_at": "tokenise", "why": "a comment deletes a filter",
     "sql": "SELECT trade_id, notional FROM trades WHERE side = 'BUY' -- AND trader_id = 7"},
    {"kind": "refusal", "halts_at": "bind", "why": "invented column",
     "sql": "SELECT lifetime_value FROM traders"},
    {"kind": "refusal", "halts_at": "verify", "why": "cartesian join",
     "sql": "SELECT t.trade_id FROM trades t INNER JOIN instruments i ON 1 = 1"},
    {"kind": "refusal", "halts_at": "bind", "why": "bare column beside an aggregate",
     "sql": "SELECT side, price, COUNT(*) FROM trades GROUP BY side"},
    {"kind": "valid", "halts_at": None, "why": "group-by; a LIMIT is injected",
     "sql": "SELECT side, COUNT(*) FROM trades GROUP BY side"},
    {"kind": "valid", "halts_at": None, "why": "join and aggregate",
     "sql": "SELECT asset_class, AVG(price) AS avg_price FROM trades t "
            "JOIN instruments i ON t.instrument_id = i.instrument_id GROUP BY asset_class"},
    {"kind": "valid", "halts_at": None, "why": "join, group-by, top 5",
     "sql": "SELECT tr.desk, COUNT(*) AS n FROM trades t "
            "JOIN traders tr ON t.trader_id = tr.trader_id "
            "GROUP BY tr.desk ORDER BY n DESC LIMIT 5"},
    {"kind": "valid", "halts_at": None, "why": "left join, least active traders",
     "sql": "SELECT tr.trader_name, COUNT(t.trade_id) AS n FROM traders tr "
            "LEFT JOIN trades t ON tr.trader_id = t.trader_id "
            "GROUP BY tr.trader_name ORDER BY n LIMIT 5"},
    {"kind": "valid", "halts_at": None, "why": "two-key order-by",
     "sql": "SELECT symbol, exchange, tick_size FROM instruments "
            "WHERE asset_class = 'Energy' ORDER BY tick_size DESC, symbol LIMIT 10"},
    {"kind": "valid", "halts_at": None, "why": "the word DROP inside a string is just data",
     "sql": "SELECT trade_id, side, price FROM trades WHERE side = 'DROP TABLE trades'"},
]


def _isolate_from_models() -> None:
    """Make a live model call impossible for the rest of this process."""
    for name in [key for key in os.environ if key.upper().endswith("_API_KEY")]:
        del os.environ[name]

    real_getaddrinfo = socket.getaddrinfo

    def local_only(host, *args, **kwargs):  # type: ignore[no-untyped-def]
        if host not in (None, "localhost", "127.0.0.1", "::1", "testserver"):
            raise RuntimeError(f"network access is disabled during the export (host {host!r})")
        return real_getaddrinfo(host, *args, **kwargs)

    socket.getaddrinfo = local_only  # type: ignore[assignment]


def _write_js(path: Path, assignments: dict[str, Any]) -> int:
    """Write ``window.SQL_ENGINE_DEMO.<name> = <json>;`` lines atomically."""
    lines = [
        "// Generated by scripts/export_static_demo.py -- do not edit by hand.",
        "window.SQL_ENGINE_DEMO = window.SQL_ENGINE_DEMO || {};",
    ]
    for name, value in assignments.items():
        body = json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        lines.append(f"window.SQL_ENGINE_DEMO.{name} = {body};")
    text = "\n".join(lines) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)
    return len(text.encode("utf-8"))


def main() -> int:
    _isolate_from_models()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "evaluation"))

    from data.generate import generate

    generate(WAREHOUSE)  # no-op when the warehouse already exists

    from fastapi.testclient import TestClient

    import run_eval
    from sql_engine import __version__
    from sql_engine.llm import GroqClient, ReplayClient
    from sql_engine.web import app as web_app

    OUTPUT.mkdir(parents=True, exist_ok=True)
    recorded = ReplayClient.from_file(RECORDED)
    golden = web_app._load_golden()

    # The server chooses its model in build_client(): Groq when a key is set, otherwise
    # the hand-authored reference fixtures. The demo must replay the *recorded* responses
    # instead, so that one choice is swapped for the duration of create_app(). The request
    # handling and serialisation are the server's own, unchanged.
    fixture_client = web_app.FixtureQuestionClient(
        replay=recorded,
        question_ids={web_app._normalise(row["question"]): row["id"] for row in golden},
        name=recorded.name,
    )
    original_build_client = web_app.build_client
    web_app.build_client = lambda engine: (fixture_client, fixture_client, fixture_client.name)
    try:
        app = web_app.create_app(WAREHOUSE)
    finally:
        web_app.build_client = original_build_client
    assert not isinstance(fixture_client.replay, GroqClient)
    client = TestClient(app)

    def call(method: str, path: str, payload: dict[str, str] | None = None) -> dict[str, Any]:
        response = client.request(method, path, json=payload)
        if response.status_code != 200:
            raise SystemExit(f"{method} {path} returned HTTP {response.status_code}")
        return response.json()

    written: dict[str, int] = {}

    # -- schema -----------------------------------------------------------------------
    schema = call("GET", "/api/schema")
    meta = {
        "generated_on": date.today().isoformat(),
        "engine_version": __version__,
        "recorded_from": recorded.name,
        "question_count": len(golden),
        "query_count": len(QUERY_EXAMPLES),
        "repository": REPOSITORY,
    }
    written["schema.js"] = _write_js(OUTPUT / "schema.js", {"meta": meta, "schema": schema})

    # -- the 45 questions, answered from recorded model responses ---------------------
    report = run_eval.evaluate(RECORDED)
    outcomes = {outcome.case.question_id: outcome for outcome in report.outcomes}
    questions: list[dict[str, Any]] = []
    answers: dict[str, dict[str, Any]] = {}
    for row in golden:
        payload = call("POST", "/api/ask", {"question": row["question"]})
        if payload.get("no_model") or str(payload.get("error") or "").startswith(
            "model call failed"
        ):
            raise SystemExit(f"{row['id']}: no recorded response could be replayed: {payload}")
        outcome = outcomes[row["id"]]
        # The page and the evaluation must be describing the same run.
        evaluation_ok = outcome.verified_note in ("verified", "verified, wrong rows")
        if payload["ok"] != evaluation_ok or len(payload["attempts"]) != outcome.attempts:
            raise SystemExit(f"{row['id']}: web answer and evaluation disagree")
        answers[row["id"]] = payload
        questions.append(
            {
                "id": row["id"],
                "question": row["question"],
                "key": web_app._normalise(row["question"]),
                "gold_sql": row["gold_sql"],
                "gold_match": outcome.verified_correct,
            }
        )
    missing = [key for key in recorded.calls if key not in recorded]
    if missing:
        raise SystemExit(f"requested keys with no recording: {missing}")
    written["answers.js"] = _write_js(
        OUTPUT / "answers.js", {"questions": questions, "answers": answers}
    )

    # -- curated SQL for the "Write SQL yourself" tab ---------------------------------
    queries: dict[str, dict[str, Any]] = {}
    for example in QUERY_EXAMPLES:
        sql = str(example["sql"])
        payload = call("POST", "/api/verify", {"sql": sql})
        halted = [stage["name"] for stage in payload["stages"] if stage["status"] == "refused"]
        expected_ok = example["kind"] == "valid"
        if payload["ok"] is not expected_ok or halted != (
            [] if expected_ok else [example["halts_at"]]
        ):
            raise SystemExit(f"unexpected verdict for {sql!r}: ok={payload['ok']} halted={halted}")
        queries[sql] = payload
    written["queries.js"] = _write_js(
        OUTPUT / "queries.js", {"query_examples": QUERY_EXAMPLES, "queries": queries}
    )

    total = sum(written.values())
    for name, size in written.items():
        print(f"{name:<12} {size / 1024:8.1f} KB")
    print(f"{'total':<12} {total / 1024:8.1f} KB")
    print(
        f"questions {len(answers)} (verified {sum(p['ok'] for p in answers.values())}, "
        f"match gold {sum(q['gold_match'] for q in questions)}); "
        f"model calls replayed {len(recorded.calls)}; queries {len(queries)}"
    )
    if total > MAX_DATA_BYTES:
        raise SystemExit(f"web/data is {total / 1e6:.1f} MB, over the {MAX_DATA_BYTES / 1e6:.0f} MB cap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
