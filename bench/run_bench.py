"""Benchmark sql-engine against SQLite and pandas, and against itself with the optimiser off.

Read the "How to read these numbers" section of the generated report before quoting any
figure from it. The short version:

* **sql-engine and pandas both read Parquet from disk on every single run.** Neither
  caches anything between iterations.
* **SQLite is measured two ways**: from an in-memory database (its fastest mode, load
  time excluded and reported separately) and from a file-backed database. The in-memory
  number is *not* a like-for-like comparison and is labelled as such -- it is included
  because it is the number a reader would otherwise assume, and hiding it would look
  like avoiding it.
* sql-engine is a pure-Python engine over NumPy. SQLite is thirty years of C. The point of
  the comparison is not to win it; it is to know the size of the gap and to be able to
  explain it.

The optimiser-on versus optimiser-off column is the one this project actually owns: same
engine, same code path, one flag.

Run: python bench/run_bench.py [--repeats 7]
"""

from __future__ import annotations

import argparse
import gc
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sql_engine import Catalog, Engine, Optimiser  # noqa: E402
from sql_engine.sqlite_oracle import build_sqlite  # noqa: E402

WAREHOUSE = ROOT / "data" / "warehouse"
RESULTS = HERE / "results.md"
SQLITE_FILE = HERE / "_bench.sqlite"


@dataclass(frozen=True, slots=True)
class Query:
    """One benchmark case: the same question expressed for each engine."""

    name: str
    description: str
    sql: str
    #: pandas implementation, written by hand because pandas has no SQL front-end.
    pandas: Callable[[Path], object]


# --------------------------------------------------------------------------------------
# pandas implementations
#
# Each reads only the columns it needs, so pandas gets the benefit of projection pruning
# for free -- a hand-written pipeline always does. Not giving it that would be a rigged
# comparison.
# --------------------------------------------------------------------------------------


def pandas_filter_project(warehouse: Path):
    frame = pd.read_parquet(warehouse / "trades.parquet", columns=["trade_id", "price"])
    return frame[frame["price"] > 3900].head(1000)


def pandas_count_filtered(warehouse: Path):
    frame = pd.read_parquet(warehouse / "trades.parquet", columns=["trader_id"])
    return len(frame[frame["trader_id"] == 7])


def pandas_group_by(warehouse: Path):
    frame = pd.read_parquet(warehouse / "trades.parquet", columns=["side", "quantity", "price"])
    return frame.groupby("side", dropna=False).agg(
        n=("quantity", "size"), q=("quantity", "sum"), p=("price", "mean")
    )


def pandas_group_by_many(warehouse: Path):
    frame = pd.read_parquet(
        warehouse / "trades.parquet", columns=["trader_id", "quantity", "price"]
    )
    return frame.groupby("trader_id", dropna=False).agg(
        n=("quantity", "size"), q=("quantity", "sum"), p=("price", "mean")
    )


def pandas_selective_scan(warehouse: Path):
    frame = pd.read_parquet(
        warehouse / "trades.parquet", columns=["trade_id", "trader_id", "side", "price"]
    )
    selected = frame[
        (frame["trader_id"] == 7) & (frame["side"] == "BUY") & (frame["price"] > 2000)
    ]
    return selected[["trade_id", "price"]]


def pandas_join_group(warehouse: Path):
    trades = pd.read_parquet(
        warehouse / "trades.parquet", columns=["instrument_id", "quantity"]
    )
    instruments = pd.read_csv(
        warehouse / "instruments.csv", usecols=["instrument_id", "asset_class"]
    )
    merged = trades.merge(instruments, on="instrument_id", how="inner")
    return merged.groupby("asset_class").agg(n=("quantity", "size"), q=("quantity", "sum"))


def pandas_join_filter(warehouse: Path):
    trades = pd.read_parquet(
        warehouse / "trades.parquet", columns=["trade_id", "instrument_id", "price"]
    )
    instruments = pd.read_csv(
        warehouse / "instruments.csv", usecols=["instrument_id", "exchange", "symbol"]
    )
    instruments = instruments[instruments["exchange"] == "CME"]
    merged = trades.merge(instruments, on="instrument_id", how="inner")
    return merged[merged["price"] > 3000][["trade_id", "symbol", "price"]]


def pandas_sort_limit(warehouse: Path):
    frame = pd.read_parquet(
        warehouse / "trades.parquet", columns=["trade_id", "price", "notional"]
    )
    return frame.sort_values(["notional", "trade_id"], ascending=[False, True]).head(50)


def pandas_distinct(warehouse: Path):
    frame = pd.read_parquet(warehouse / "trades.parquet", columns=["trader_id", "venue"])
    return frame.drop_duplicates().sort_values(["trader_id", "venue"])


QUERIES: list[Query] = [
    Query(
        "filter_project",
        "Narrow filter, two columns out of nine",
        "SELECT trade_id, price FROM trades WHERE price > 3900 LIMIT 1000",
        pandas_filter_project,
    ),
    Query(
        "count_filtered",
        "Count with an equality filter",
        "SELECT COUNT(*) AS n FROM trades WHERE trader_id = 7",
        pandas_count_filtered,
    ),
    Query(
        "group_by_small",
        "Aggregate into 2 groups",
        "SELECT side, COUNT(*) AS n, SUM(quantity) AS q, AVG(price) AS p "
        "FROM trades GROUP BY side ORDER BY side",
        pandas_group_by,
    ),
    Query(
        "group_by_wide",
        "Aggregate into 40 groups",
        "SELECT trader_id, COUNT(*) AS n, SUM(quantity) AS q, AVG(price) AS p "
        "FROM trades GROUP BY trader_id ORDER BY trader_id",
        pandas_group_by_many,
    ),
    Query(
        "selective_scan",
        "Three conjuncts, highly selective -- the pushdown case",
        "SELECT trade_id, price FROM trades "
        "WHERE trader_id = 7 AND side = 'BUY' AND price > 2000",
        pandas_selective_scan,
    ),
    Query(
        "join_group",
        "200k x 120 hash join, then aggregate",
        "SELECT i.asset_class, COUNT(*) AS n, SUM(t.quantity) AS q FROM trades t "
        "INNER JOIN instruments i ON t.instrument_id = i.instrument_id "
        "GROUP BY i.asset_class ORDER BY i.asset_class",
        pandas_join_group,
    ),
    Query(
        "join_filter",
        "Join with filters on both sides",
        "SELECT t.trade_id, i.symbol, t.price FROM trades t "
        "INNER JOIN instruments i ON t.instrument_id = i.instrument_id "
        "WHERE i.exchange = 'CME' AND t.price > 3000",
        pandas_join_filter,
    ),
    Query(
        "sort_limit",
        "Full sort of 200k rows, top 50",
        "SELECT trade_id, price FROM trades ORDER BY notional DESC, trade_id ASC LIMIT 50",
        pandas_sort_limit,
    ),
    Query(
        "distinct",
        "DISTINCT over two columns",
        "SELECT DISTINCT trader_id, venue FROM trades ORDER BY trader_id, venue",
        pandas_distinct,
    ),
]


@dataclass(slots=True)
class Timing:
    """Timings for one query under one engine configuration."""

    samples: list[float] = field(default_factory=list)

    @property
    def median(self) -> float:
        return statistics.median(self.samples)

    @property
    def best(self) -> float:
        return min(self.samples)


def measure(function: Callable[[], object], repeats: int) -> Timing:
    """Time ``function``, discarding one warm-up run.

    The warm-up matters here: the first Parquet read pays for OS page-cache misses and
    for pyarrow's lazy imports, and including it would measure the filesystem rather
    than the engine. Garbage collection is disabled during the timed region so a
    collection triggered by one engine is not charged to another.
    """
    function()  # warm-up, discarded
    timing = Timing()
    for _ in range(repeats):
        gc.collect()
        gc.disable()
        try:
            start = time.perf_counter()
            function()
            timing.samples.append(time.perf_counter() - start)
        finally:
            gc.enable()
    return timing


def build_file_database(catalog: Catalog) -> tuple[sqlite3.Connection, float]:
    """A file-backed SQLite database, so one comparison reads from disk like sql-engine."""
    if SQLITE_FILE.exists():
        SQLITE_FILE.unlink()
    start = time.perf_counter()
    connection = build_sqlite(catalog, str(SQLITE_FILE))
    return connection, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sql-engine.")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()

    catalog = Catalog.from_directory(WAREHOUSE)
    optimised = Engine(catalog)
    unoptimised = Engine(
        catalog, optimiser=Optimiser(predicate_pushdown=False, projection_pruning=False)
    )

    memory_start = time.perf_counter()
    memory_db = build_sqlite(catalog)
    memory_load = time.perf_counter() - memory_start
    file_db, file_load = build_file_database(catalog)

    rows: list[dict[str, object]] = []
    for query in QUERIES:
        print(f"  {query.name} ...", flush=True)
        on = measure(lambda q=query: optimised.query(q.sql), args.repeats)
        off = measure(lambda q=query: unoptimised.query(q.sql), args.repeats)
        mem = measure(
            lambda q=query: memory_db.execute(q.sql).fetchall(), args.repeats
        )
        fil = measure(lambda q=query: file_db.execute(q.sql).fetchall(), args.repeats)
        pnd = measure(lambda q=query: q.pandas(WAREHOUSE), args.repeats)

        compiled_on = optimised.compile(query.sql)
        compiled_off = unoptimised.compile(query.sql)
        result = optimised.query(query.sql)

        rows.append(
            {
                "query": query,
                "sql_engine_on": on,
                "sql_engine_off": off,
                "sqlite_memory": mem,
                "sqlite_file": fil,
                "pandas": pnd,
                "columns_on": compiled_on.report.columns_read_after,
                "columns_off": compiled_off.report.columns_read_before,
                "pushed": compiled_on.report.predicates_pushed_into_scans,
                "rows_out": len(result.rows),
                "rows_scanned": result.stats.rows_scanned,
                "rows_after_filter": result.stats.rows_after_scan_filter,
            }
        )

    memory_db.close()
    file_db.close()
    if SQLITE_FILE.exists():
        SQLITE_FILE.unlink()

    args.output.write_text(
        render(rows, args.repeats, memory_load, file_load), encoding="utf-8"
    )
    print(f"\nwritten to {args.output}")


def _speedup(slow: Timing, fast: Timing) -> str:
    if fast.median <= 0:
        return "-"
    return f"{slow.median / fast.median:.2f}x"


def render(
    rows: list[dict[str, object]], repeats: int, memory_load: float, file_load: float
) -> str:
    import platform

    lines: list[str] = []
    lines.append("# Benchmark results")
    lines.append("")
    lines.append(
        f"Generated by `python bench/run_bench.py` on {date.today().isoformat()}. "
        "Every number below came out of that command; none is typed by hand."
    )
    lines.append("")
    lines.append(
        f"Machine: {platform.python_implementation()} {platform.python_version()} on "
        f"{platform.system()} {platform.release()}, {platform.machine()}. "
        f"Median of {repeats} runs after one discarded warm-up."
    )
    lines.append("")
    lines.append("Dataset: `trades` 200,000 rows (Parquet, 9 columns), "
                 "`instruments` 120 rows (CSV), `traders` 40 rows (CSV), "
                 "`daily_prices` 60,000 rows (Parquet).")
    lines.append("")

    lines.append("## How to read these numbers")
    lines.append("")
    lines.append(
        "- **sql-engine** and **pandas** read Parquet from disk on every run. Nothing is "
        "cached between iterations."
    )
    lines.append(
        f"- **SQLite (memory)** runs against a pre-loaded in-memory database. Loading it "
        f"took {memory_load:.2f} s and that cost is *excluded* from every figure in its "
        "column. This is not a like-for-like comparison and should not be quoted as one."
    )
    lines.append(
        f"- **SQLite (file)** runs against a file-backed database, built in "
        f"{file_load:.2f} s, also excluded. The OS page cache warms up across runs, "
        "just as it does for the Parquet files."
    )
    lines.append(
        "- **pandas** implementations are hand-written in `bench/run_bench.py` and read "
        "only the columns they need, so pandas gets projection pruning for free."
    )
    lines.append(
        "- sql-engine is pure Python over NumPy; SQLite is a mature C engine. Where SQLite "
        "wins, it should. The number worth arguing about is the optimiser column, which "
        "compares sql-engine against itself."
    )
    lines.append("")

    lines.append("## Wall-clock time (seconds, median)")
    lines.append("")
    lines.append(
        "| query | sql-engine | sql-engine (no optimiser) | SQLite (memory) | SQLite (file) "
        "| pandas | rows out |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        query: Query = row["query"]  # type: ignore[assignment]
        lines.append(
            f"| `{query.name}` "
            f"| {row['sql_engine_on'].median:.4f} "  # type: ignore[union-attr]
            f"| {row['sql_engine_off'].median:.4f} "  # type: ignore[union-attr]
            f"| {row['sqlite_memory'].median:.4f} "  # type: ignore[union-attr]
            f"| {row['sqlite_file'].median:.4f} "  # type: ignore[union-attr]
            f"| {row['pandas'].median:.4f} "  # type: ignore[union-attr]
            f"| {row['rows_out']:,} |"
        )
    lines.append("")

    lines.append("## What the optimiser bought")
    lines.append("")
    lines.append(
        "Same engine, same query, one flag. `columns read` is the number of columns the "
        "Parquet reader was asked for; `rows kept at scan` is how many survived the "
        "pushed predicates out of 200,000 scanned."
    )
    lines.append("")
    lines.append(
        "| query | speedup | columns read (off -> on) | conjuncts pushed | "
        "rows kept at scan |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        query: Query = row["query"]  # type: ignore[assignment]
        kept = (
            f"{row['rows_after_filter']:,} / {row['rows_scanned']:,}"
            if row["rows_scanned"]
            else "-"
        )
        lines.append(
            f"| `{query.name}` "
            f"| {_speedup(row['sql_engine_off'], row['sql_engine_on'])} "  # type: ignore[arg-type]
            f"| {row['columns_off']} -> {row['columns_on']} "
            f"| {row['pushed']} "
            f"| {kept} |"
        )
    lines.append("")

    lines.append("## Relative to SQLite and pandas")
    lines.append("")
    lines.append("Ratio of median times. Above 1.0 means sql-engine is slower.")
    lines.append("")
    lines.append("| query | vs SQLite (memory) | vs SQLite (file) | vs pandas |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in rows:
        query: Query = row["query"]  # type: ignore[assignment]
        gb = row["sql_engine_on"].median  # type: ignore[union-attr]
        lines.append(
            f"| `{query.name}` "
            f"| {gb / row['sqlite_memory'].median:.2f}x "  # type: ignore[union-attr]
            f"| {gb / row['sqlite_file'].median:.2f}x "  # type: ignore[union-attr]
            f"| {gb / row['pandas'].median:.2f}x |"  # type: ignore[union-attr]
        )
    lines.append("")

    lines.append("## Query descriptions")
    lines.append("")
    for row in rows:
        query: Query = row["query"]  # type: ignore[assignment]
        lines.append(f"**`{query.name}`** — {query.description}")
        lines.append("")
        lines.append("```sql")
        lines.append(query.sql)
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
