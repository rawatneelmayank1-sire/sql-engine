"""Author + verify the golden question/SQL pairs, then emit questions.jsonl.

Every gold SQL is executed against db/market.sqlite. For each one we record the
row count, the column names, an order-insensitive hash of the result set and an
order-sensitive hash (which is what ORDER BY / LIMIT cases are really asserting).

Any query that fails to execute is reported and NOT written to the JSONL - a gold
query that does not run is worse than useless.

Supported-subset rule for sql-engine: SELECT only, no subqueries, no CTEs, no window
functions, no set operations, no self-joins. `check_subset()` enforces this here so
an out-of-subset query can never quietly enter the file.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sqlite3

HERE = pathlib.Path(__file__).resolve().parent
DB_PATH = HERE / "db" / "market.sqlite"
OUT_PATH = HERE / "questions.jsonl"

# id, question, sql, difficulty, features
CASES: list[tuple[str, str, str, str, list[str]]] = [
    # ---------------------------------------------------------------- projection
    ("g001", "List the codes of all exchanges.",
     "SELECT code FROM exchanges", "easy", ["select"]),
    ("g002", "Show the name and country of every exchange.",
     "SELECT name, country FROM exchanges", "easy", ["select"]),
    ("g003", "List the ticker and name of every instrument.",
     "SELECT ticker, name FROM instruments", "easy", ["select"]),
    ("g004", "What distinct currencies do instruments trade in?",
     "SELECT DISTINCT currency FROM instruments", "easy", ["select", "distinct"]),
    ("g005", "What are the distinct instrument types?",
     "SELECT DISTINCT instrument_type FROM instruments", "easy", ["select", "distinct"]),
    ("g006", "List every sector name and its description.",
     "SELECT name, description FROM sectors", "easy", ["select"]),
    ("g007", "Show the name and firm of each analyst.",
     "SELECT name, firm FROM analysts", "easy", ["select"]),

    # ---------------------------------------------------------------- simple WHERE
    ("g008", "Which exchanges are in the United States?",
     "SELECT code, name FROM exchanges WHERE country = 'United States'",
     "easy", ["select", "where"]),
    ("g009", "Which exchanges were founded before 1900?",
     "SELECT name, founded_year FROM exchanges WHERE founded_year < 1900",
     "easy", ["select", "where"]),
    ("g010", "List the tickers of instruments on exchange 1.",
     "SELECT ticker FROM instruments WHERE exchange_id = 1",
     "easy", ["select", "where"]),
    ("g011", "Which instruments are exchange traded funds?",
     "SELECT ticker, name FROM instruments WHERE instrument_type = 'etf'",
     "easy", ["select", "where"]),
    ("g012", "Show analysts who started in 2015 or later.",
     "SELECT name, start_year FROM analysts WHERE start_year >= 2015",
     "easy", ["select", "where"]),

    # ---------------------------------------------------------------- AND / OR / NOT
    ("g013", "Which instruments are active equities?",
     "SELECT ticker FROM instruments WHERE instrument_type = 'equity' AND is_active = 1",
     "medium", ["select", "where", "and"]),
    ("g014", "Which instruments trade in US dollars or British pounds?",
     "SELECT ticker, currency FROM instruments WHERE currency = 'USD' OR currency = 'GBP'",
     "medium", ["select", "where", "or"]),
    ("g015", "List instruments that are not equities.",
     "SELECT ticker, instrument_type FROM instruments WHERE NOT instrument_type = 'equity'",
     "medium", ["select", "where", "not"]),
    ("g016", "Show active instruments on exchange 1 that are not ETFs.",
     "SELECT ticker FROM instruments "
     "WHERE exchange_id = 1 AND is_active = 1 AND NOT instrument_type = 'etf'",
     "medium", ["select", "where", "and", "not"]),
    ("g017", "Which instruments are inactive or trade in Japanese yen?",
     "SELECT ticker, currency, is_active FROM instruments "
     "WHERE is_active = 0 OR currency = 'JPY'",
     "medium", ["select", "where", "or"]),

    # ---------------------------------------------------------------- IN / NOT IN
    ("g018", "Which instruments trade in euros or yen?",
     "SELECT ticker, currency FROM instruments WHERE currency IN ('EUR', 'JPY')",
     "medium", ["select", "where", "in"]),
    ("g019", "List instruments whose currency is not USD, EUR or GBP.",
     "SELECT ticker, currency FROM instruments "
     "WHERE currency NOT IN ('USD', 'EUR', 'GBP')",
     "medium", ["select", "where", "in", "not"]),
    ("g020", "Show ratings that are buy or strong buy.",
     "SELECT rating_id, rating FROM ratings WHERE rating IN ('buy', 'strong_buy')",
     "medium", ["select", "where", "in"]),

    # ---------------------------------------------------------------- BETWEEN
    ("g021", "How many daily bars closed between 100 and 200?",
     "SELECT COUNT(*) AS n FROM daily_bars WHERE close BETWEEN 100 AND 200",
     "medium", ["select", "where", "between", "count"]),
    ("g022", "Show bars for instrument 3 traded during February 2024.",
     "SELECT trade_date, close FROM daily_bars "
     "WHERE instrument_id = 3 AND trade_date BETWEEN '2024-02-01' AND '2024-02-29'",
     "medium", ["select", "where", "between", "and"]),
    ("g023", "Which analysts started between 2005 and 2012?",
     "SELECT name, start_year FROM analysts WHERE start_year BETWEEN 2005 AND 2012",
     "medium", ["select", "where", "between"]),

    # ---------------------------------------------------------------- LIKE
    ("g024", "Which instruments have a ticker starting with B?",
     "SELECT ticker FROM instruments WHERE ticker LIKE 'B%'",
     "medium", ["select", "where", "like"]),
    ("g025", "Find instruments whose name contains 'Capital'.",
     "SELECT ticker, name FROM instruments WHERE name LIKE '%Capital%'",
     "medium", ["select", "where", "like"]),
    ("g026", "Which instruments have a name ending in 'Technologies'?",
     "SELECT ticker, name FROM instruments WHERE name LIKE '%Technologies'",
     "medium", ["select", "where", "like"]),
    ("g027", "List analysts whose firm does not contain 'Capital'.",
     "SELECT name, firm FROM analysts WHERE firm NOT LIKE '%Capital%'",
     "medium", ["select", "where", "like", "not"]),

    # ---------------------------------------------------------------- IS NULL
    ("g028", "Which instruments have not been assigned a sector?",
     "SELECT ticker, name FROM instruments WHERE sector_id IS NULL",
     "medium", ["select", "where", "is_null"]),
    ("g029", "Which instruments have an unknown share count?",
     "SELECT ticker FROM instruments WHERE shares_outstanding IS NULL",
     "medium", ["select", "where", "is_null"]),
    ("g030", "Which sectors have no description recorded?",
     "SELECT name FROM sectors WHERE description IS NULL",
     "easy", ["select", "where", "is_null"]),
    ("g031", "How many ratings were published without a price target?",
     "SELECT COUNT(*) AS n FROM ratings WHERE price_target IS NULL",
     "medium", ["select", "where", "is_null", "count"]),
    ("g032", "List analysts whose country is known, with that country.",
     "SELECT name, country FROM analysts WHERE country IS NOT NULL",
     "medium", ["select", "where", "is_null", "not"]),

    # ---------------------------------------------------------------- aggregates
    ("g033", "How many instruments are there in total?",
     "SELECT COUNT(*) AS n FROM instruments", "easy", ["select", "count"]),
    ("g034", "How many distinct exchanges do instruments trade on?",
     "SELECT COUNT(DISTINCT exchange_id) AS n FROM instruments",
     "easy", ["select", "count", "distinct"]),
    ("g035", "What is the total volume traded across all daily bars?",
     "SELECT SUM(volume) AS total_volume FROM daily_bars", "easy", ["select", "sum"]),
    ("g036", "What is the average closing price across all bars?",
     "SELECT AVG(close) AS avg_close FROM daily_bars", "easy", ["select", "avg"]),
    ("g037", "What is the lowest low price recorded?",
     "SELECT MIN(low) AS min_low FROM daily_bars", "easy", ["select", "min"]),
    ("g038", "What is the highest high price recorded?",
     "SELECT MAX(high) AS max_high FROM daily_bars", "easy", ["select", "max"]),

    # ---------------------------------------------------------------- GROUP BY
    ("g039", "How many instruments are listed on each exchange?",
     "SELECT exchange_id, COUNT(*) AS n FROM instruments GROUP BY exchange_id",
     "medium", ["select", "group_by", "count"]),
    ("g040", "What is the total volume traded per instrument?",
     "SELECT instrument_id, SUM(volume) AS total_volume FROM daily_bars "
     "GROUP BY instrument_id",
     "medium", ["select", "group_by", "sum"]),
    ("g041", "What is the average closing price per instrument?",
     "SELECT instrument_id, AVG(close) AS avg_close FROM daily_bars "
     "GROUP BY instrument_id",
     "medium", ["select", "group_by", "avg"]),
    ("g042", "What are the lowest and highest closing prices per instrument?",
     "SELECT instrument_id, MIN(close) AS min_close, MAX(close) AS max_close "
     "FROM daily_bars GROUP BY instrument_id",
     "medium", ["select", "group_by", "min", "max"]),
    ("g043", "How many instruments are there of each type in each currency?",
     "SELECT currency, instrument_type, COUNT(*) AS n FROM instruments "
     "GROUP BY currency, instrument_type",
     "medium", ["select", "group_by", "count"]),
    ("g044", "How many ratings of each kind were issued?",
     "SELECT rating, COUNT(*) AS n FROM ratings GROUP BY rating",
     "medium", ["select", "group_by", "count"]),
    ("g045", "What is the average price target per rating category?",
     "SELECT rating, AVG(price_target) AS avg_target FROM ratings GROUP BY rating",
     "medium", ["select", "group_by", "avg"]),

    # ---------------------------------------------------------------- HAVING
    ("g046", "Which exchanges list more than 12 instruments?",
     "SELECT exchange_id, COUNT(*) AS n FROM instruments "
     "GROUP BY exchange_id HAVING COUNT(*) > 12",
     "hard", ["select", "group_by", "having", "count"]),
    ("g047", "Which instruments have an average closing price above 150?",
     "SELECT instrument_id, AVG(close) AS avg_close FROM daily_bars "
     "GROUP BY instrument_id HAVING AVG(close) > 150",
     "hard", ["select", "group_by", "having", "avg"]),
    ("g048", "Which instruments traded a total volume above 500 million?",
     "SELECT instrument_id, SUM(volume) AS total_volume FROM daily_bars "
     "GROUP BY instrument_id HAVING SUM(volume) > 500000000",
     "hard", ["select", "group_by", "having", "sum"]),
    ("g049", "Which analysts issued at least 20 ratings?",
     "SELECT analyst_id, COUNT(*) AS n FROM ratings "
     "GROUP BY analyst_id HAVING COUNT(*) >= 20",
     "hard", ["select", "group_by", "having", "count"]),
    ("g050", "Which currencies have more than 10 instruments and a max share count above 1 billion?",
     "SELECT currency, COUNT(*) AS n, MAX(shares_outstanding) AS max_shares "
     "FROM instruments GROUP BY currency "
     "HAVING COUNT(*) > 10 AND MAX(shares_outstanding) > 1000000000",
     "hard", ["select", "group_by", "having", "count", "max", "and"]),

    # ---------------------------------------------------------------- ORDER BY / LIMIT
    ("g051", "List exchanges from oldest to newest.",
     "SELECT name, founded_year FROM exchanges ORDER BY founded_year ASC",
     "easy", ["select", "order_by"]),
    ("g052", "List analysts by start year, newest first.",
     "SELECT name, start_year FROM analysts ORDER BY start_year DESC, name ASC",
     "medium", ["select", "order_by"]),
    ("g053", "Show instruments ordered by currency then ticker.",
     "SELECT currency, ticker FROM instruments ORDER BY currency ASC, ticker ASC",
     "medium", ["select", "order_by"]),
    ("g054", "What are the 5 highest closing prices recorded?",
     "SELECT instrument_id, trade_date, close FROM daily_bars "
     "ORDER BY close DESC, instrument_id ASC, trade_date ASC LIMIT 5",
     "medium", ["select", "order_by", "limit"]),
    ("g055", "Show the first 10 instruments by ticker.",
     "SELECT ticker FROM instruments ORDER BY ticker ASC LIMIT 10",
     "easy", ["select", "order_by", "limit"]),
    ("g056", "Show instruments 11 to 20 when ordered by ticker.",
     "SELECT ticker FROM instruments ORDER BY ticker ASC LIMIT 10 OFFSET 10",
     "medium", ["select", "order_by", "limit", "offset"]),
    ("g057", "Which 3 instruments have the largest total traded volume?",
     "SELECT instrument_id, SUM(volume) AS total_volume FROM daily_bars "
     "GROUP BY instrument_id ORDER BY total_volume DESC, instrument_id ASC LIMIT 3",
     "hard", ["select", "group_by", "sum", "order_by", "limit"]),

    # ---------------------------------------------------------------- INNER JOIN
    ("g058", "Show each instrument's ticker together with its exchange name.",
     "SELECT i.ticker, e.name AS exchange_name FROM instruments i "
     "JOIN exchanges e ON i.exchange_id = e.exchange_id",
     "medium", ["select", "join"]),
    ("g059", "Show each instrument's ticker with its sector name.",
     "SELECT i.ticker, s.name AS sector_name FROM instruments i "
     "JOIN sectors s ON i.sector_id = s.sector_id",
     "medium", ["select", "join"]),
    ("g060", "Which instruments are listed on exchanges in Japan?",
     "SELECT i.ticker, e.country FROM instruments i "
     "JOIN exchanges e ON i.exchange_id = e.exchange_id "
     "WHERE e.country = 'Japan'",
     "medium", ["select", "join", "where"]),
    ("g061", "Show the analyst name, rating and instrument ticker for each rating.",
     "SELECT a.name AS analyst_name, r.rating, i.ticker FROM ratings r "
     "JOIN analysts a ON r.analyst_id = a.analyst_id "
     "JOIN instruments i ON r.instrument_id = i.instrument_id",
     "hard", ["select", "join"]),
    ("g062", "How many instruments does each exchange list, by exchange name?",
     "SELECT e.name AS exchange_name, COUNT(*) AS n FROM instruments i "
     "JOIN exchanges e ON i.exchange_id = e.exchange_id "
     "GROUP BY e.name",
     "hard", ["select", "join", "group_by", "count"]),
    ("g063", "What is the average closing price per sector?",
     "SELECT s.name AS sector_name, AVG(b.close) AS avg_close "
     "FROM daily_bars b "
     "JOIN instruments i ON b.instrument_id = i.instrument_id "
     "JOIN sectors s ON i.sector_id = s.sector_id "
     "GROUP BY s.name",
     "hard", ["select", "join", "group_by", "avg"]),
    ("g064", "Which sectors contain more than 10 instruments?",
     "SELECT s.name AS sector_name, COUNT(*) AS n FROM instruments i "
     "JOIN sectors s ON i.sector_id = s.sector_id "
     "GROUP BY s.name HAVING COUNT(*) > 10",
     "hard", ["select", "join", "group_by", "having", "count"]),
    ("g065", "Which 5 sectors have the highest total traded volume?",
     "SELECT s.name AS sector_name, SUM(b.volume) AS total_volume "
     "FROM daily_bars b "
     "JOIN instruments i ON b.instrument_id = i.instrument_id "
     "JOIN sectors s ON i.sector_id = s.sector_id "
     "GROUP BY s.name ORDER BY total_volume DESC, s.name ASC LIMIT 5",
     "hard", ["select", "join", "group_by", "sum", "order_by", "limit"]),
    ("g066", "Show buy ratings on US-listed instruments with the analyst firm.",
     "SELECT i.ticker, a.firm, r.rating FROM ratings r "
     "JOIN instruments i ON r.instrument_id = i.instrument_id "
     "JOIN analysts a ON r.analyst_id = a.analyst_id "
     "JOIN exchanges e ON i.exchange_id = e.exchange_id "
     "WHERE e.country = 'United States' AND r.rating = 'buy'",
     "hard", ["select", "join", "where", "and"]),

    # ---------------------------------------------------------------- LEFT JOIN
    ("g067", "List every instrument with its sector name, including unclassified ones.",
     "SELECT i.ticker, s.name AS sector_name FROM instruments i "
     "LEFT JOIN sectors s ON i.sector_id = s.sector_id",
     "hard", ["select", "left_join"]),
    ("g068", "Which sectors have no instruments assigned to them?",
     "SELECT s.name AS sector_name FROM sectors s "
     "LEFT JOIN instruments i ON i.sector_id = s.sector_id "
     "WHERE i.instrument_id IS NULL",
     "hard", ["select", "left_join", "where", "is_null"]),
    ("g069", "How many instruments does each sector have, including empty sectors?",
     "SELECT s.name AS sector_name, COUNT(i.instrument_id) AS n FROM sectors s "
     "LEFT JOIN instruments i ON i.sector_id = s.sector_id "
     "GROUP BY s.name",
     "hard", ["select", "left_join", "group_by", "count"]),
    ("g070", "Which sectors have fewer than 10 instruments, counting empty sectors?",
     "SELECT s.name AS sector_name, COUNT(i.instrument_id) AS n FROM sectors s "
     "LEFT JOIN instruments i ON i.sector_id = s.sector_id "
     "GROUP BY s.name HAVING COUNT(i.instrument_id) < 10",
     "hard", ["select", "left_join", "group_by", "having", "count"]),
    ("g071", "Show unclassified instruments' tickers alongside the null sector name.",
     "SELECT i.ticker, s.name AS sector_name FROM instruments i "
     "LEFT JOIN sectors s ON i.sector_id = s.sector_id "
     "WHERE s.sector_id IS NULL",
     "hard", ["select", "left_join", "where", "is_null"]),
    ("g072", "For each analyst, how many ratings did they issue, including any with none?",
     "SELECT a.name AS analyst_name, COUNT(r.rating_id) AS n FROM analysts a "
     "LEFT JOIN ratings r ON r.analyst_id = a.analyst_id "
     "GROUP BY a.name ORDER BY n DESC, a.name ASC",
     "hard", ["select", "left_join", "group_by", "count", "order_by"]),

    # ---------------------------------------------------------------- combined
    ("g073", "Average volume per exchange for active instruments only, highest first.",
     "SELECT e.code, AVG(b.volume) AS avg_volume FROM daily_bars b "
     "JOIN instruments i ON b.instrument_id = i.instrument_id "
     "JOIN exchanges e ON i.exchange_id = e.exchange_id "
     "WHERE i.is_active = 1 "
     "GROUP BY e.code ORDER BY avg_volume DESC, e.code ASC",
     "hard", ["select", "join", "where", "group_by", "avg", "order_by"]),
    ("g074", "For USD or EUR instruments, show the 5 with the highest maximum close.",
     "SELECT i.ticker, MAX(b.close) AS max_close FROM daily_bars b "
     "JOIN instruments i ON b.instrument_id = i.instrument_id "
     "WHERE i.currency IN ('USD', 'EUR') "
     "GROUP BY i.ticker ORDER BY max_close DESC, i.ticker ASC LIMIT 5",
     "hard", ["select", "join", "where", "in", "group_by", "max", "order_by", "limit"]),
    ("g075", "Count ratings per instrument for tickers starting with C, only where at least 5.",
     "SELECT i.ticker, COUNT(*) AS n FROM ratings r "
     "JOIN instruments i ON r.instrument_id = i.instrument_id "
     "WHERE i.ticker LIKE 'C%' "
     "GROUP BY i.ticker HAVING COUNT(*) >= 5 ORDER BY n DESC, i.ticker ASC",
     "hard", ["select", "join", "where", "like", "group_by", "having", "count", "order_by"]),
]

# --------------------------------------------------------------------------- checks

FORBIDDEN = [
    (r"\bWITH\b", "cte"),
    (r"\bUNION\b", "set_operation"),
    (r"\bINTERSECT\b", "set_operation"),
    (r"\bEXCEPT\b", "set_operation"),
    (r"\bOVER\s*\(", "window_function"),
    (r"\bSELECT\b[\s\S]*\(\s*SELECT\b", "subquery"),
]


def check_subset(sql: str) -> list[str]:
    """Return a list of supported-subset violations for `sql` (empty == fine)."""
    upper = sql.upper()
    problems = [label for pattern, label in FORBIDDEN if re.search(pattern, upper)]
    if not upper.lstrip().startswith("SELECT"):
        problems.append("not_select")
    return problems


def canonical(value) -> str:
    if value is None:
        return "\x00NULL"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def hash_rows(rows: list[tuple], sort: bool) -> str:
    encoded = [[canonical(v) for v in row] for row in rows]
    if sort:
        encoded.sort()
    return hashlib.sha256(
        json.dumps(encoded, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> None:
    ids = [c[0] for c in CASES]
    assert len(ids) == len(set(ids)), "duplicate case ids"

    conn = sqlite3.connect(DB_PATH)
    verified: list[dict] = []
    failures: list[tuple[str, str]] = []
    empty: list[str] = []

    for case_id, question, sql, difficulty, features in CASES:
        violations = check_subset(sql)
        if violations:
            failures.append((case_id, f"subset violation: {violations}"))
            continue
        try:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            columns = [d[0] for d in cursor.description]
        except Exception as exc:  # noqa: BLE001 - we want to report anything
            failures.append((case_id, f"{type(exc).__name__}: {exc}"))
            continue
        if not rows:
            empty.append(case_id)
        verified.append(
            {
                "id": case_id,
                "question": question,
                "sql": sql,
                "difficulty": difficulty,
                "features": features,
                "expected_row_count": len(rows),
                "expected_columns": columns,
                "expected_hash": hash_rows(rows, sort=True),
                "expected_ordered_hash": hash_rows(rows, sort=False),
            }
        )
    conn.close()

    with OUT_PATH.open("w", encoding="utf-8") as handle:
        for record in verified:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"authored:  {len(CASES)}")
    print(f"VERIFIED (executed OK against SQLite): {len(verified)}")
    print(f"failed:    {len(failures)}")
    for case_id, why in failures:
        print(f"   FAIL {case_id}: {why}")
    print(f"empty result sets (suspicious): {empty if empty else 'none'}")
    by_difficulty: dict[str, int] = {}
    for record in verified:
        by_difficulty[record["difficulty"]] = by_difficulty.get(record["difficulty"], 0) + 1
    print("difficulty:", dict(sorted(by_difficulty.items())))
    feats: dict[str, int] = {}
    for record in verified:
        for f in record["features"]:
            feats[f] = feats.get(f, 0) + 1
    print("features:", dict(sorted(feats.items())))
    print(f"written -> {OUT_PATH}")


if __name__ == "__main__":
    main()
