"""Build the committed reference fixture set.

READ THIS BEFORE BELIEVING ANY NUMBER THE EVALUATION PRODUCES.

No LLM API key was available when this project was built, so the responses in
``fixtures/reference.json`` are **hand-authored, not recorded from a live model**. They
are written here, in code, rather than pasted into a JSON blob, so that the composition
is auditable: you can see exactly which failure was planted where and in what
proportion.

What that means for the evaluation:

* The **verification catch-rate by category** is a real measurement. It measures this
  repository's verifier against a set of candidate SQL strings. The verifier does not
  know or care where those strings came from.
* The **execution accuracy with and without verification** is *not* a measurement of any
  model. It is a measurement of this fixture set. The gap between the two numbers is a
  property of the errors planted below.

To get real numbers: set ``GROQ_API_KEY`` and run ``evaluation/record_fixtures.py``,
which drives the same pipeline against a live model and writes
``fixtures/recorded.json``. ``run_eval.py --fixtures recorded`` then reports genuine
figures with the same command.

The planted error distribution follows the failure taxonomy reported in the text-to-SQL
literature -- invented column names dominate, followed by unsupported functions and
missing GROUP BY entries. Four questions are answered with SQL that is *valid but
semantically wrong*, because a verifier that only checks structure cannot catch those,
and an evaluation that hides that limitation would be dishonest.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "fixtures" / "reference.json"
GOLDEN = HERE / "golden_set.jsonl"

#: Candidate responses that deviate from the gold SQL, keyed by question id.
#: Each entry is a list: index 0 is the first attempt, index 1 the response to the
#: verifier's feedback, and so on. A question absent from this table is answered
#: correctly on the first attempt with its gold SQL.
PLANTED: dict[str, list[str]] = {
    # ---- invented column names (the most common real failure) -------------------
    "golden-017": [
        "SELECT r.desk, COUNT(*) AS n FROM trades t INNER JOIN traders r "
        "ON t.trader_id = r.trader_id GROUP BY r.desk_name HAVING COUNT(*) > 20000 "
        "ORDER BY r.desk_name ASC LIMIT 200",
        "SELECT r.desk, COUNT(*) AS n FROM trades t INNER JOIN traders r "
        "ON t.trader_id = r.trader_id GROUP BY r.desk HAVING COUNT(*) > 20000 "
        "ORDER BY r.desk ASC LIMIT 200",
    ],
    "golden-026": [
        "SELECT i.exchange, MAX(p.close) AS highest_close FROM daily_prices p "
        "INNER JOIN instruments i ON p.instrument_id = i.instrument_id "
        "GROUP BY i.exchange ORDER BY i.exchange ASC LIMIT 200",
        "SELECT i.exchange, MAX(p.close_price) AS highest_close FROM daily_prices p "
        "INNER JOIN instruments i ON p.instrument_id = i.instrument_id "
        "GROUP BY i.exchange ORDER BY i.exchange ASC LIMIT 200",
    ],
    "golden-034": [
        "SELECT instrument_id, price_date, high - low AS range_size FROM daily_prices "
        "ORDER BY range_size DESC LIMIT 10",
        "SELECT instrument_id, price_date, high_price - low_price AS range_size "
        "FROM daily_prices ORDER BY high_price - low_price DESC, instrument_id ASC, "
        "price_date ASC LIMIT 10",
    ],
    "golden-045": [
        # Two invented names in a row: the model corrects the first and invents another.
        "SELECT i.symbol, SUM(t.quantity) AS total_quantity FROM trades t "
        "INNER JOIN instruments i ON t.instrument_id = i.id WHERE i.exchange = 'EUREX' "
        "GROUP BY i.symbol ORDER BY total_quantity DESC LIMIT 10",
        "SELECT i.ticker, SUM(t.quantity) AS total_quantity FROM trades t "
        "INNER JOIN instruments i ON t.instrument_id = i.instrument_id "
        "WHERE i.exchange = 'EUREX' GROUP BY i.ticker "
        "ORDER BY total_quantity DESC LIMIT 10",
        "SELECT i.symbol, SUM(t.quantity) AS total_quantity FROM trades t "
        "INNER JOIN instruments i ON t.instrument_id = i.instrument_id "
        "WHERE i.exchange = 'EUREX' GROUP BY i.symbol "
        "ORDER BY total_quantity DESC, i.symbol ASC LIMIT 10",
    ],

    # ---- functions the dialect does not have -------------------------------------
    "golden-003": [
        "SELECT ROUND(AVG(price), 2) AS avg_price FROM trades",
        "SELECT AVG(price) AS avg_price FROM trades LIMIT 200",
    ],
    "golden-033": [
        "SELECT desk, AVG(CAST(seniority AS REAL)) AS avg_seniority FROM traders "
        "GROUP BY desk ORDER BY desk ASC",
        "SELECT desk, AVG(seniority) AS avg_seniority FROM traders GROUP BY desk "
        "ORDER BY desk ASC LIMIT 200",
    ],
    "golden-013": [
        "SELECT COUNT(*) AS n FROM trades WHERE YEAR(trade_date) = 2025 "
        "AND MONTH(trade_date) = 3",
        # Repairs into something that parses and binds -- and is still wrong, because it
        # excludes the 31st. The verifier cannot catch this, and the evaluation says so.
        "SELECT COUNT(*) AS n FROM trades WHERE trade_date >= '2025-03-01' "
        "AND trade_date < '2025-03-31' LIMIT 200",
    ],

    # ---- comma joins and other join syntax the grammar refuses --------------------
    "golden-015": [
        "SELECT i.asset_class, COUNT(*) AS n FROM trades t, instruments i "
        "WHERE t.instrument_id = i.instrument_id GROUP BY i.asset_class",
        "SELECT i.asset_class, COUNT(*) AS n FROM trades t INNER JOIN instruments i "
        "ON t.instrument_id = i.instrument_id GROUP BY i.asset_class "
        "ORDER BY i.asset_class ASC LIMIT 200",
    ],
    "golden-042": [
        # Never recovers: three different unsupported constructs in a row.
        "SELECT i.asset_class, AVG(p.volume) AS avg_volume FROM daily_prices p "
        "CROSS JOIN instruments i GROUP BY i.asset_class",
        "SELECT i.asset_class, AVG(p.daily_volume) AS avg_volume FROM daily_prices p "
        "INNER JOIN instruments i ON p.instrument_id = i.instrument_id "
        "GROUP BY i.asset_class ORDER BY i.asset_class ASC LIMIT 200",
        "SELECT DISTINCT i.asset_class, AVG(p.volume) OVER (PARTITION BY i.asset_class) "
        "AS avg_volume FROM daily_prices p INNER JOIN instruments i "
        "ON p.instrument_id = i.instrument_id LIMIT 200",
    ],

    # ---- aggregation rules --------------------------------------------------------
    "golden-020": [
        "SELECT instrument_id, AVG(price) AS avg_price FROM trades "
        "ORDER BY avg_price DESC LIMIT 10",
        "SELECT instrument_id, AVG(price) AS avg_price FROM trades "
        "GROUP BY instrument_id ORDER BY avg_price DESC, instrument_id ASC LIMIT 10",
    ],
    "golden-040": [
        "SELECT exchange, currency, COUNT(*) AS n FROM instruments GROUP BY exchange "
        "ORDER BY exchange ASC LIMIT 200",
        "SELECT exchange, currency, COUNT(*) AS n FROM instruments "
        "GROUP BY exchange, currency ORDER BY exchange ASC, currency ASC LIMIT 200",
    ],
    "golden-011": [
        "SELECT t.trader_id, r.trader_name, COUNT(*) AS n FROM trades t "
        "INNER JOIN traders r ON t.trader_id = r.trader_id GROUP BY t.trader_id "
        "ORDER BY n DESC LIMIT 10",
        "SELECT trader_id, COUNT(*) AS n FROM trades GROUP BY trader_id "
        "ORDER BY n DESC, trader_id ASC LIMIT 10",
    ],
    "golden-037": [
        # Never recovers: aggregate in WHERE, then a comma join, then USING.
        "SELECT i.asset_class, COUNT(*) AS n, AVG(t.price) AS avg_price FROM trades t "
        "INNER JOIN instruments i ON t.instrument_id = i.instrument_id "
        "WHERE COUNT(*) > 30000 GROUP BY i.asset_class",
        "SELECT i.asset_class, COUNT(*) AS n, AVG(t.price) AS avg_price "
        "FROM trades t, instruments i WHERE t.instrument_id = i.instrument_id "
        "GROUP BY i.asset_class HAVING COUNT(*) > 30000",
        "SELECT i.asset_class, COUNT(*) AS n, AVG(t.price) AS avg_price FROM trades t "
        "INNER JOIN instruments i USING (instrument_id) GROUP BY i.asset_class "
        "HAVING COUNT(*) > 30000",
    ],

    # ---- ambiguous column ----------------------------------------------------------
    "golden-024": [
        "SELECT t.trade_id, i.symbol, t.price FROM trades t INNER JOIN instruments i "
        "ON instrument_id = instrument_id ORDER BY t.notional DESC LIMIT 20",
        "SELECT t.trade_id, i.symbol, t.price FROM trades t INNER JOIN instruments i "
        "ON t.instrument_id = i.instrument_id ORDER BY t.notional DESC, t.trade_id ASC "
        "LIMIT 20",
    ],

    # ---- subquery ------------------------------------------------------------------
    "golden-029": [
        "SELECT r.trader_name, (SELECT COUNT(*) FROM trades t "
        "WHERE t.trader_id = r.trader_id) AS n FROM traders r ORDER BY r.trader_name",
        "SELECT r.trader_name, COUNT(t.trade_id) AS n FROM traders r "
        "LEFT JOIN trades t ON r.trader_id = t.trader_id GROUP BY r.trader_name "
        "ORDER BY r.trader_name ASC LIMIT 200",
    ],

    # ---- valid SQL, wrong answer: the verifier's blind spot, stated openly ---------
    "golden-008": [
        # `= NULL` is never true in SQL. Parses, binds, verifies, returns 0.
        "SELECT COUNT(*) AS n FROM traders WHERE desk = NULL LIMIT 200",
    ],
    "golden-019": [
        # Empty string is not NULL.
        "SELECT COUNT(*) AS n FROM trades WHERE venue = '' LIMIT 200",
    ],
    "golden-021": [
        # Forgets DISTINCT.
        "SELECT COUNT(trader_id) AS n FROM trades WHERE venue = 'SCREEN' LIMIT 200",
    ],
    "golden-035": [
        # `<> 'SCREEN' OR IS NULL` silently changes the answer by including unknowns.
        "SELECT COUNT(*) AS n FROM trades WHERE venue <> 'SCREEN' OR venue IS NULL "
        "LIMIT 200",
    ],
}


def build() -> dict[str, str]:
    """Produce the ``{prompt key: response}`` mapping the ReplayClient consumes."""
    golden = [
        json.loads(line)
        for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    responses: dict[str, str] = {}
    for entry in golden:
        question_id = entry["id"]
        candidates = PLANTED.get(question_id, [entry["gold_sql"]])
        for index, sql in enumerate(candidates):
            responses[f"{question_id}/attempt-{index}"] = sql
        # A model that keeps failing keeps answering; repeating the last candidate for
        # the remaining attempts models a model that has stopped making progress.
        for index in range(len(candidates), 3):
            responses[f"{question_id}/attempt-{index}"] = candidates[-1]
    return responses


def main() -> None:
    responses = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "recorded_from": "hand-authored (no LLM API key available at build time)",
                "provenance": (
                    "These responses were written by hand in "
                    "evaluation/build_reference_fixtures.py to exercise the verifier's "
                    "failure taxonomy. They are NOT recorded from a live model. The "
                    "verification catch-rate they produce is a real measurement of the "
                    "verifier; the execution-accuracy gap is a property of this fixture "
                    "set, not of any model."
                ),
                "responses": responses,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"{OUTPUT}  {len(responses)} responses")


if __name__ == "__main__":
    main()
