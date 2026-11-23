"""Author + sanity-check the adversarial cases, then emit cases.jsonl.

Each case says what the sql-engine verification layer *should* do with the SQL.

The sanity check runs `EXPLAIN <sql>` against a **read-only** connection to the
golden database. EXPLAIN parses and resolves every table/column name without
executing anything, so it is safe even for the DROP/DELETE/ATTACH cases, and it
proves that an "invented" column or table really is absent from the schema
rather than something the author merely assumed was absent.

The recorded `sqlite_parses` flag matters for the engine: cases where it is true
(not_select, cartesian, unbounded) are perfectly legal SQL, so rejecting them has
to come from sql-engine's own static analysis and policy - SQLite itself will not
complain. Cases where it is false can be caught by parse/name resolution alone.
"""
from __future__ import annotations

import json
import pathlib
import sqlite3

HERE = pathlib.Path(__file__).resolve().parent
DB_PATH = HERE.parent / "golden" / "db" / "market.sqlite"
OUT_PATH = HERE / "cases.jsonl"

# id, sql, expect, reason, note
CASES: list[tuple[str, str, str, str, str]] = [
    # ------------------------------------------------- invented_column (6)
    ("a001", "SELECT ticker, market_cap FROM instruments", "reject", "invented_column",
     "market_cap is the single most plausible hallucination for this schema; it does not exist"),
    ("a002", "SELECT instrument_id, closing_price FROM daily_bars", "reject", "invented_column",
     "column is called close, not closing_price"),
    ("a003", "SELECT name, sector_name FROM instruments", "reject", "invented_column",
     "sector_name lives on sectors; instruments only carries sector_id"),
    ("a004", "SELECT ticker, dividend_yield FROM instruments WHERE dividend_yield > 0.03",
     "reject", "invented_column",
     "invented column used in both projection and predicate"),
    ("a005", "SELECT e.code, e.num_listings FROM exchanges e", "reject", "invented_column",
     "a derived quantity the model assumed was stored"),
    ("a006",
     "SELECT i.ticker, AVG(b.vwap) AS avg_vwap FROM daily_bars b "
     "JOIN instruments i ON b.instrument_id = i.instrument_id GROUP BY i.ticker",
     "reject", "invented_column",
     "invented column hidden inside an aggregate in an otherwise well-formed join"),
    ("a007", "SELECT ticker, exchange_name FROM instruments WHERE is_active = 1",
     "reject", "invented_column",
     "denormalised column that would require a join to obtain"),

    # ------------------------------------------------- invented_table (5)
    ("a008", "SELECT * FROM securities", "reject", "invented_table",
     "plausible synonym for instruments"),
    ("a009", "SELECT instrument_id, close FROM price_history", "reject", "invented_table",
     "plausible synonym for daily_bars"),
    ("a010", "SELECT t.symbol FROM tickers t", "reject", "invented_table",
     "invented table plus invented column"),
    ("a011",
     "SELECT i.ticker, f.eps FROM instruments i "
     "JOIN fundamentals f ON f.instrument_id = i.instrument_id",
     "reject", "invented_table",
     "invented table joined correctly onto a real one - join shape looks right"),
    ("a012", "SELECT * FROM daily_bar", "reject", "invented_table",
     "near-miss singular/plural typo of daily_bars"),

    # ------------------------------------------------- not_select (8)
    ("a013", "DROP TABLE instruments", "reject", "not_select", "destructive DDL"),
    ("a014", "DELETE FROM daily_bars WHERE close < 10", "reject", "not_select",
     "destructive DML with a predicate, so it looks deliberate"),
    ("a015", "UPDATE instruments SET is_active = 0", "reject", "not_select",
     "unfiltered mass update"),
    ("a016", "INSERT INTO sectors (sector_id, name) VALUES (99, 'Fabricated')",
     "reject", "not_select", "data insertion"),
    ("a017", "ALTER TABLE instruments ADD COLUMN market_cap REAL", "reject", "not_select",
     "schema mutation - note it would 'fix' case a001, which is exactly why it must be blocked"),
    ("a018", "ATTACH DATABASE 'evil.db' AS evil", "reject", "not_select",
     "attaches an external database file into the session"),
    ("a019", "PRAGMA table_info(instruments)", "reject", "not_select",
     "introspection pragma; not a SELECT and can leak schema"),
    ("a020", "CREATE TABLE backdoor (id INTEGER, payload TEXT)", "reject", "not_select",
     "DDL creating new state"),

    # ------------------------------------------------- cartesian (4)
    ("a021", "SELECT i.ticker, e.name FROM instruments i, exchanges e",
     "reject", "cartesian",
     "comma join with no ON/WHERE equi-predicate: 120 x 8 = 960 rows"),
    ("a022", "SELECT i.ticker, s.name FROM instruments i JOIN sectors s ON 1 = 1",
     "reject", "cartesian",
     "ON clause present but is a constant, not an equi-predicate"),
    ("a023", "SELECT i.ticker, b.close FROM instruments i CROSS JOIN daily_bars b",
     "reject", "cartesian",
     "explicit CROSS JOIN: 120 x 14400 = 1,728,000 rows"),
    ("a024",
     "SELECT i.ticker, a.name FROM instruments i, analysts a WHERE i.is_active = 1",
     "reject", "cartesian",
     "has a WHERE clause, but it filters one table only - still no join predicate"),

    # ------------------------------------------------- unparseable (8)
    ("a025", "SELECT ticker FROM WHERE instruments", "reject", "unparseable",
     "FROM and WHERE transposed"),
    ("a026", "SELCT ticker FROM instruments", "reject", "unparseable",
     "misspelled keyword"),
    ("a027",
     "Show me all the instruments listed on Nasdaq together with their latest closing price.",
     "reject", "unparseable",
     "the model echoed the question back instead of emitting SQL"),
    ("a028", "```sql\nSELECT ticker FROM instruments\n```", "reject", "unparseable",
     "valid SQL wrapped in markdown fences - the single most common LLM output defect"),
    ("a029",
     "Here is the query you asked for:\n\n```sql\nSELECT COUNT(*) FROM instruments;\n```\n\n"
     "This counts every instrument in the table.",
     "reject", "unparseable",
     "SQL buried in prose and fences"),
    ("a030", "SELECT ticker, FROM instruments", "reject", "unparseable",
     "trailing comma in the select list"),
    ("a031", "", "reject", "unparseable", "empty string"),
    ("a032", "SELECT * FROM instruments WHERE ticker = 'ABC", "reject", "unparseable",
     "unterminated string literal"),

    # ------------------------------------------------- injection_attempt (6)
    ("a033", "SELECT ticker FROM instruments; DROP TABLE instruments",
     "reject", "injection_attempt",
     "stacked statement: benign SELECT followed by destructive DDL"),
    ("a034", "SELECT ticker FROM instruments WHERE ticker = 'A' OR '1' = '1' --",
     "reject", "injection_attempt",
     "classic tautology plus trailing comment terminator"),
    ("a035", "SELECT ticker FROM instruments; DELETE FROM ratings; --",
     "reject", "injection_attempt",
     "multiple stacked statements with a comment terminator"),
    ("a036",
     "SELECT ticker FROM instruments WHERE exchange_id = 1 /* bypass */ "
     "UNION SELECT name FROM sectors",
     "reject", "injection_attempt",
     "inline comment plus UNION exfiltration; UNION is also outside the supported subset"),
    ("a037", "SELECT ticker FROM instruments -- AND is_active = 1",
     "reject", "injection_attempt",
     "comment terminator silently removes the intended filter; parses fine and returns MORE rows"),
    ("a038", "SELECT ticker FROM instruments WHERE ticker = ''; ATTACH DATABASE 'eve.db' AS eve; --",
     "reject", "injection_attempt",
     "stacked ATTACH to pull in an external file"),

    # ------------------------------------------------- unbounded (4)
    ("a039", "SELECT * FROM daily_bars", "reject", "unbounded",
     "full 14,400-row table scan, all columns, no LIMIT"),
    ("a040",
     "SELECT b.bar_id, b.trade_date, b.open, b.high, b.low, b.close, b.volume, "
     "i.ticker, i.name, e.name AS exchange_name FROM daily_bars b "
     "JOIN instruments i ON b.instrument_id = i.instrument_id "
     "JOIN exchanges e ON i.exchange_id = e.exchange_id",
     "reject", "unbounded",
     "properly joined but returns every bar with wide columns and no LIMIT"),
    ("a041", "SELECT * FROM daily_bars ORDER BY close DESC", "reject", "unbounded",
     "full sort of the largest table with no LIMIT - the ORDER BY implies a top-N was meant"),
    ("a042", "SELECT instrument_id, trade_date, close, volume FROM daily_bars WHERE volume > 0",
     "reject", "unbounded",
     "predicate that excludes nothing, so it is a disguised full scan"),

    # ------------------------------------------------- valid (6)
    ("a043", "SELECT ticker FROM instruments WHERE exchange_id = 1 LIMIT 10",
     "allow", "valid", "bounded, single table, filtered"),
    ("a044", "SELECT COUNT(*) AS n FROM instruments", "allow", "valid",
     "aggregate returning exactly one row, so no LIMIT is needed"),
    ("a045",
     "SELECT s.name AS sector_name, COUNT(*) AS n FROM instruments i "
     "JOIN sectors s ON i.sector_id = s.sector_id GROUP BY s.name",
     "allow", "valid", "proper equi-join, grouped, bounded by cardinality of sectors"),
    ("a046",
     "SELECT i.ticker, b.close FROM daily_bars b "
     "JOIN instruments i ON b.instrument_id = i.instrument_id "
     "WHERE b.trade_date = '2024-03-01' ORDER BY b.close DESC LIMIT 5",
     "allow", "valid", "equi-join, filtered, ordered and explicitly limited"),
    ("a047", "SELECT DISTINCT currency FROM instruments", "allow", "valid",
     "small distinct projection"),
    ("a048",
     "SELECT i.ticker, s.name AS sector_name FROM instruments i "
     "LEFT JOIN sectors s ON i.sector_id = s.sector_id WHERE s.sector_id IS NULL",
     "allow", "valid",
     "LEFT JOIN with an ON equi-predicate - must NOT be flagged cartesian"),
]


def main() -> None:
    ids = [c[0] for c in CASES]
    assert len(ids) == len(set(ids)), "duplicate case ids"

    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    records: list[dict] = []
    mismatches: list[str] = []

    for case_id, sql, expect, reason, note in CASES:
        try:
            conn.execute("EXPLAIN " + sql)
            parses, error = True, None
        except Exception as exc:  # noqa: BLE001
            parses, error = False, f"{type(exc).__name__}: {exc}"

        # Self-checks: these assertions are what make the labels trustworthy.
        if reason in {"invented_column", "invented_table", "unparseable"} and parses:
            mismatches.append(f"{case_id} ({reason}) unexpectedly PARSED - label is wrong")
        if reason == "valid" and not parses:
            mismatches.append(f"{case_id} (valid) failed to parse: {error}")
        if reason in {"cartesian", "unbounded"} and not parses:
            mismatches.append(f"{case_id} ({reason}) should be legal SQL but did not parse: {error}")

        records.append(
            {
                "id": case_id,
                "sql": sql,
                "expect": expect,
                "reason": reason,
                "note": note,
                "sqlite_parses": parses,
                "sqlite_error": error,
            }
        )
    conn.close()

    with OUT_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_reason: dict[str, int] = {}
    for record in records:
        by_reason[record["reason"]] = by_reason.get(record["reason"], 0) + 1
    n_reject = sum(1 for r in records if r["expect"] == "reject")
    print(f"total cases: {len(records)}  (reject {n_reject} / allow {len(records) - n_reject})")
    print("by reason:", dict(sorted(by_reason.items())))
    print("parses OK:", sum(1 for r in records if r["sqlite_parses"]),
          "| rejected by SQLite itself:", sum(1 for r in records if not r["sqlite_parses"]))
    if mismatches:
        print("LABEL MISMATCHES:")
        for m in mismatches:
            print("   ", m)
    else:
        print("label self-check: all consistent")
    print(f"written -> {OUT_PATH}")


if __name__ == "__main__":
    main()
