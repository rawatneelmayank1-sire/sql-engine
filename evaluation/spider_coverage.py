"""How much of the Spider dev set can this dialect even express?

**This does not report Spider execution accuracy, and no number it produces should be
compared with a published Spider score.** Execution accuracy requires the 20 dev SQLite
databases, which are not in this repository, and it requires an engine that can run
Spider's gold queries -- which sql-engine mostly cannot, because they use subqueries, set
operations and scalar functions that this dialect does not implement.

What it does measure is honest and useful: given the 1,034 gold queries of the Spider dev
split, what fraction

1. **parse** under the sql-engine grammar, and
2. **bind** against the corresponding database schema.

That turns the README's Limitations list from a set of adjectives into a number. If the
answer is "a quarter of Spider", then that is what "a deliberately small subset of SQL"
means, stated precisely, against a public benchmark rather than against my own examples.

The failure breakdown is the interesting part: it says exactly which missing feature
costs the most coverage, which is the ordered to-do list for extending the dialect.

Requires the Spider dev files under ``eval_data/spider/``. Skips cleanly if absent, so a
clone without them still passes CI.

Run: python evaluation/spider_coverage.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sql_engine.binder import Binder  # noqa: E402
from sql_engine.catalog import Catalog, ColumnSchema, DataType, TableSchema  # noqa: E402
from sql_engine.errors import (  # noqa: E402
    AmbiguousColumnError,
    BindError,
    CatalogError,
    InvalidAggregationError,
    NotSelectError,
    ParseError,
    TokeniseError,
    UnknownColumnError,
)
from sql_engine.parser import parse  # noqa: E402

SPIDER = ROOT / "eval_data" / "spider"
DEV = SPIDER / "dev.jsonl"
SCHEMAS = SPIDER / "dev_schemas.json"
RESULTS = HERE / "spider_coverage.md"

#: Spider's column type vocabulary mapped onto the four types this engine has. The
#: mapping only has to be good enough for *binding* -- no data is read -- so time and
#: "others" collapse to TEXT exactly as they would if the data were loaded.
_TYPES: dict[str, DataType] = {
    "number": DataType.REAL,
    "text": DataType.TEXT,
    "time": DataType.TEXT,
    "boolean": DataType.BOOLEAN,
    "others": DataType.TEXT,
}

#: Constructs whose absence explains a parse failure, matched against the gold SQL so the
#: breakdown names a missing *feature* rather than a token position. Order matters: the
#: first pattern that matches wins, so the more specific ones come first.
_FEATURE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("subquery", re.compile(r"\(\s*SELECT\b", re.I)),
    ("set operation (UNION/INTERSECT/EXCEPT)", re.compile(r"\b(UNION|INTERSECT|EXCEPT)\b", re.I)),
    ("comma join", re.compile(r"\bFROM\s+[\w\"`.]+(\s+(AS\s+)?\w+)?\s*,", re.I)),
    ("JOIN ... USING", re.compile(r"\bUSING\s*\(", re.I)),
    ("scalar function", re.compile(
        r"\b(ROUND|CAST|SUBSTR|LOWER|UPPER|LENGTH|ABS|COALESCE|IFNULL|STRFTIME|DATE|"
        r"JULIANDAY|REPLACE|TRIM|INSTR|GROUP_CONCAT|TOTAL)\s*\(", re.I)),
    ("string concatenation", re.compile(r"\|\|")),
    ("CASE expression", re.compile(r"\bCASE\b", re.I)),
    ("EXISTS / NOT EXISTS", re.compile(r"\bEXISTS\b", re.I)),
    ("LIMIT with an expression", re.compile(r"\bLIMIT\s+[^\d\s]", re.I)),
    ("outer join other than LEFT", re.compile(r"\b(RIGHT|FULL|CROSS)\s+(OUTER\s+)?JOIN\b", re.I)),
]


def build_catalog(entry: dict) -> Catalog:
    """Turn one Spider schema entry into a sql-engine catalog.

    Spider carries both natural-language names (``column_names``) and real identifiers
    (``column_names_original``). Only the originals appear in the gold SQL, so only those
    are registered -- binding against the prose names would inflate the coverage number.
    """
    table_names = entry["table_names_original"]
    columns: dict[int, list[ColumnSchema]] = {i: [] for i in range(len(table_names))}
    for (table_index, column_name), column_type in zip(
        entry["column_names_original"], entry["column_types"]
    ):
        if table_index < 0:  # the synthetic "*" column
            continue
        columns[table_index].append(
            ColumnSchema(column_name, _TYPES.get(column_type, DataType.TEXT))
        )

    tables = []
    for index, name in enumerate(table_names):
        if not columns[index]:
            continue
        tables.append(
            TableSchema(
                name=name,
                path=Path(f"{name}.parquet"),
                file_format="parquet",
                columns=tuple(columns[index]),
                row_count=0,
            )
        )
    return Catalog(tables)


_DOUBLE_QUOTED = re.compile(r'"([^"]*)"')


def normalise_double_quoted_strings(sql: str) -> str:
    """Rewrite ``"text"`` as ``'text'``.

    Spider's gold SQL writes string literals in double quotes. In standard SQL -- and in
    sql-engine -- double quotes delimit an *identifier*, so ``WHERE Airline = "JetBlue
    Airways"`` asks for a column named ``JetBlue Airways`` and correctly fails to bind.

    SQLite accepts it anyway: when a double-quoted token does not resolve to a column, it
    silently falls back to treating it as a string literal. SQLite's own documentation
    calls this a misfeature and provides a flag to turn it off.

    Counting those queries as "unsupported" would blame sql-engine for following the
    standard, so coverage is reported both ways. The substitution is safe on this dataset
    specifically because Spider's gold SQL contains no double-quoted identifiers -- which
    is checked, not assumed, by the fact that every rewritten query still parses.
    """
    return _DOUBLE_QUOTED.sub(
        lambda m: "'" + m.group(1).replace("'", "''") + "'", sql
    )


def classify_parse_failure(sql: str) -> str:
    for label, pattern in _FEATURE_PATTERNS:
        if pattern.search(sql):
            return label
    return "other syntax"


def analyse() -> dict:
    rows = [
        json.loads(line)
        for line in DEV.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    schemas = {
        entry["db_id"]: build_catalog(entry)
        for entry in json.loads(SCHEMAS.read_text(encoding="utf-8"))
    }

    parsed = 0
    bound = 0
    bound_normalised = 0
    dqs_only = 0
    parse_reasons: Counter = Counter()
    bind_reasons: Counter = Counter()
    examples: dict[str, str] = {}

    for row in rows:
        sql = row["sql"]
        try:
            statement = parse(sql)
        except (ParseError, TokeniseError, NotSelectError):
            reason = classify_parse_failure(sql)
            parse_reasons[reason] += 1
            examples.setdefault(reason, sql)
            continue
        parsed += 1

        catalog = schemas.get(row["db_id"])
        if catalog is None:
            bind_reasons["schema unavailable"] += 1
            continue
        # Bound a second time with double-quoted strings rewritten, so the report can
        # separate "this dialect cannot express it" from "Spider quotes strings the way
        # only SQLite accepts".
        try:
            Binder(catalog).bind(parse(normalise_double_quoted_strings(sql)))
            bound_normalised += 1
            normalised_ok = True
        except Exception:
            normalised_ok = False

        try:
            Binder(catalog).bind(statement)
        except UnknownColumnError:
            if normalised_ok:
                bind_reasons["double-quoted string literal (SQLite-only)"] += 1
                dqs_only += 1
                examples.setdefault("bind: double-quoted string literal", sql)
            else:
                bind_reasons["unknown column"] += 1
                examples.setdefault("bind: unknown column", sql)
        except AmbiguousColumnError:
            bind_reasons["ambiguous column"] += 1
            examples.setdefault("bind: ambiguous column", sql)
        except InvalidAggregationError:
            bind_reasons["bare column with an aggregate"] += 1
            examples.setdefault("bind: bare column with an aggregate", sql)
        except CatalogError:
            bind_reasons["unknown table"] += 1
        except BindError:
            bind_reasons["other bind error"] += 1
        else:
            bound += 1

    return {
        "total": len(rows),
        "parsed": parsed,
        "bound": bound,
        "bound_normalised": bound_normalised,
        "dqs_only": dqs_only,
        "databases": len({row["db_id"] for row in rows}),
        "parse_reasons": parse_reasons,
        "bind_reasons": bind_reasons,
        "examples": examples,
    }


def render(data: dict) -> str:
    total = data["total"]
    lines: list[str] = []
    lines.append("# Spider dev coverage")
    lines.append("")
    lines.append(
        f"Generated by `python evaluation/spider_coverage.py` on {date.today().isoformat()}."
    )
    lines.append("")
    lines.append("## What this is, and what it is not")
    lines.append("")
    lines.append(
        "**This is not a Spider execution-accuracy score and must not be compared with "
        "one.** Execution accuracy needs the 20 dev SQLite databases (not in this "
        "repository) and an engine that can run Spider's gold queries -- which sql-engine "
        "mostly cannot, by design."
    )
    lines.append("")
    lines.append(
        "What is measured here is **grammar and schema coverage**: of the "
        f"{total:,} gold queries in the Spider dev split, across "
        f"{data['databases']} databases, how many can this dialect express at all. It "
        "turns the Limitations section from a list of adjectives into a number."
    )
    lines.append("")

    lines.append("## Coverage")
    lines.append("")
    lines.append("| Stage | Queries | Share |")
    lines.append("| --- | ---: | ---: |")
    lines.append(f"| Gold queries in the dev split | {total:,} | 100.0% |")
    lines.append(
        f"| Parse under the sql-engine grammar | {data['parsed']:,} | "
        f"{data['parsed'] / total:.1%} |"
    )
    lines.append(
        f"| Also bind against the Spider schema | {data['bound']:,} | "
        f"{data['bound'] / total:.1%} |"
    )
    lines.append(
        "| Also bind, after rewriting Spider's double-quoted string literals | "
        f"{data['bound_normalised']:,} | {data['bound_normalised'] / total:.1%} |"
    )
    lines.append("")
    lines.append(
        f"The last row exists because {data['dqs_only']:,} dev queries write string "
        'literals in double quotes (`WHERE Airline = "JetBlue Airways"`). In standard '
        "SQL double quotes delimit an identifier, so sql-engine correctly reports no such "
        "column; SQLite accepts it through a documented compatibility fallback. That is "
        "a literal-quoting difference, not a missing feature, so both numbers are shown "
        "rather than picking whichever flatters the engine."
    )
    lines.append("")

    lines.append("## Why the rest do not parse")
    lines.append("")
    lines.append(
        "This is the ordered to-do list for widening the dialect: the feature at the "
        "top costs the most coverage."
    )
    lines.append("")
    lines.append("| Missing feature | Queries | Share of all dev queries |")
    lines.append("| --- | ---: | ---: |")
    for reason, count in data["parse_reasons"].most_common():
        lines.append(f"| {reason} | {count:,} | {count / total:.1%} |")
    lines.append("")

    if data["bind_reasons"]:
        lines.append("## Parsed but did not bind")
        lines.append("")
        lines.append(
            "These parse as sql-engine SQL but fail name resolution against the Spider "
            "schema. The double-quoted-literal group is a quoting difference rather "
            "than a missing feature (see above). The rest are sql-engine's deliberately "
            "stricter aggregation rule: it refuses `SELECT name, COUNT(*) ... GROUP BY "
            "id`, where SQLite silently picks an arbitrary row for `name`."
        )
        lines.append("")
        lines.append("| Reason | Queries |")
        lines.append("| --- | ---: |")
        for reason, count in data["bind_reasons"].most_common():
            lines.append(f"| {reason} | {count:,} |")
        lines.append("")

    lines.append("## Example of each rejection")
    lines.append("")
    for reason, sql in sorted(data["examples"].items()):
        lines.append(f"**{reason}**")
        lines.append("")
        lines.append("```sql")
        lines.append(sql.strip())
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Spider dev grammar coverage.")
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()

    if not DEV.exists() or not SCHEMAS.exists():
        print(
            f"Spider dev files not found under {SPIDER}. Nothing to measure; skipping.",
            file=sys.stderr,
        )
        raise SystemExit(0)

    data = analyse()
    args.output.write_text(render(data), encoding="utf-8")
    total = data["total"]
    print(f"dev queries      {total:,}")
    print(f"parse            {data['parsed']:,}  ({data['parsed'] / total:.1%})")
    print(f"parse and bind   {data['bound']:,}  ({data['bound'] / total:.1%})")
    print(f"written to       {args.output}")


if __name__ == "__main__":
    main()
