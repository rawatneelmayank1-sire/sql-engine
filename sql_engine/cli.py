"""Command line interface.

    sql-engine query   "SELECT side, COUNT(*) FROM trades GROUP BY side"
    sql-engine explain "SELECT trade_id FROM trades WHERE trader_id = 7"
    sql-engine verify  "DROP TABLE trades"
    sql-engine schema
    sql-engine serve

``explain`` and ``verify`` exist as first-class commands because they are the two things
worth showing someone: what the optimiser did, and what the verifier refused.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .catalog import Catalog
from .engine import Engine
from .errors import SQLEngineError
from .llm import Verifier

DEFAULT_WAREHOUSE = Path(__file__).resolve().parent.parent / "data" / "warehouse"


def _engine(directory: Path) -> Engine:
    return Engine(Catalog.from_directory(directory))


def _print_rows(result, limit: int) -> None:
    """Print a result as a plain aligned table.

    Deliberately not a rich/tabulate dependency: the output of a database engine is
    something people pipe into other tools, and a plain table stays greppable.
    """
    if not result.rows:
        print("(no rows)")
        return
    rows = result.rows[:limit]
    headers = list(result.column_names)
    cells = [[("NULL" if v is None else str(v)) for v in row] for row in rows]
    widths = [
        max(len(headers[i]), max((len(row[i]) for row in cells), default=0))
        for i in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in cells:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))
    if len(result.rows) > limit:
        print(f"... {len(result.rows) - limit:,} more row(s)")
    print(
        f"\n{len(result.rows):,} row(s); "
        f"{result.stats.rows_scanned:,} scanned, "
        f"{result.stats.columns_read} column(s) read"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sql-engine", description=__doc__)
    parser.add_argument(
        "--warehouse", type=Path, default=DEFAULT_WAREHOUSE, help="data directory"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("query", help="run a SELECT and print the rows")
    run.add_argument("sql")
    run.add_argument("--limit", type=int, default=50, help="rows to print")

    explain = sub.add_parser("explain", help="print the plan before and after optimisation")
    explain.add_argument("sql")

    verify = sub.add_parser("verify", help="run the verification checks without executing")
    verify.add_argument("sql")

    sub.add_parser("schema", help="print the catalog")

    serve = sub.add_parser("serve", help="start the web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        from .web.app import create_app

        uvicorn.run(create_app(args.warehouse), host=args.host, port=args.port)
        return 0

    try:
        engine = _engine(args.warehouse)
    except SQLEngineError as error:
        print(f"error: {error}", file=sys.stderr)
        print("hint: run `python data/generate.py` first", file=sys.stderr)
        return 2

    try:
        if args.command == "schema":
            print(engine.schema_text())
            return 0
        if args.command == "explain":
            print(engine.explain(args.sql))
            return 0
        if args.command == "verify":
            verdict = Verifier(engine).verify(args.sql)
            print("VERIFIED" if verdict.ok else "REFUSED")
            for repair in verdict.repairs:
                print(f"  repaired  {repair}")
            for failure in verdict.failures:
                print(f"  {failure}")
            if verdict.ok:
                print(f"  sql       {verdict.sql}")
                print(f"  est scan  {verdict.estimated_scan_bytes / 1e6:.1f} MB")
            return 0 if verdict.ok else 1
        if args.command == "query":
            _print_rows(engine.query(args.sql), args.limit)
            return 0
    except SQLEngineError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1

    return 0  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
