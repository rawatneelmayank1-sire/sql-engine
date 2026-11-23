"""Loader for the sql-engine evaluation datasets.

Standard library only - no third-party imports - so the engine can depend on this
without pulling pandas/pyarrow into its runtime.

    from eval_data.loader import load_golden, load_adversarial, load_spider, golden_db_path

    for case in load_golden():
        rows = run(case.sql)
        assert len(rows) == case.expected_row_count

Run ``python loader.py`` to print dataset statistics.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Iterator

__all__ = [
    "GoldenCase",
    "AdversarialCase",
    "load_golden",
    "load_adversarial",
    "load_spider",
    "golden_db_path",
    "golden_schema",
    "spider_schemas",
    "DATA_ROOT",
]

DATA_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parent

GOLDEN_DIR: pathlib.Path = DATA_ROOT / "golden"
ADVERSARIAL_DIR: pathlib.Path = DATA_ROOT / "adversarial"
SPIDER_DIR: pathlib.Path = DATA_ROOT / "spider"


# --------------------------------------------------------------------------- models


@dataclass(frozen=True)
class GoldenCase:
    """One natural-language question paired with its gold SQL.

    ``expected_row_count`` / ``expected_hash`` are populated only for cases whose
    gold SQL was actually executed against a database (the hand-written golden
    set). They are ``None`` for Spider, where the SQLite database files are not
    distributed with this repository - see ``spider/PROVENANCE.md``.
    """

    id: str
    question: str
    sql: str
    difficulty: str
    features: tuple[str, ...] = ()
    expected_row_count: int | None = None
    expected_columns: tuple[str, ...] | None = None
    expected_hash: str | None = None
    expected_ordered_hash: str | None = None
    db_id: str | None = None
    source: str = "golden"

    @property
    def is_verified(self) -> bool:
        """True when a recorded expected result exists to compare against."""
        return self.expected_hash is not None

    def has_feature(self, name: str) -> bool:
        return name in self.features


@dataclass(frozen=True)
class AdversarialCase:
    """One SQL string the verification layer is expected to accept or reject."""

    id: str
    sql: str
    expect: str
    reason: str
    note: str = ""
    sqlite_parses: bool | None = None
    sqlite_error: str | None = None

    @property
    def should_reject(self) -> bool:
        return self.expect == "reject"

    @property
    def needs_static_analysis(self) -> bool:
        """True when SQLite itself accepts the SQL, so only sql-engine can catch it.

        Applies to `not_select`, `cartesian` and `unbounded` cases: they are
        perfectly legal SQL and will not raise on execution.
        """
        return self.should_reject and self.sqlite_parses is True


# --------------------------------------------------------------------------- helpers


def _read_jsonl(path: pathlib.Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _tuple_or_none(value: Any) -> tuple[str, ...] | None:
    return tuple(value) if value is not None else None


# --------------------------------------------------------------------------- public API


def golden_db_path() -> pathlib.Path:
    """Absolute path to the golden SQLite database.

    Raises:
        FileNotFoundError: if the database has not been generated yet.
    """
    path = GOLDEN_DIR / "db" / "market.sqlite"
    if not path.exists():
        raise FileNotFoundError(
            f"golden database missing at {path}. "
            f"Regenerate it with: python {GOLDEN_DIR / 'generate_db.py'}"
        )
    return path


def golden_schema() -> dict[str, Any]:
    """The machine-readable schema description of the golden database."""
    return json.loads((GOLDEN_DIR / "db" / "schema.json").read_text(encoding="utf-8"))


def load_golden() -> list[GoldenCase]:
    """Load the hand-written golden question/SQL pairs.

    Every case carries a verified expected row count and result-set hash.
    """
    path = GOLDEN_DIR / "questions.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Regenerate it with: "
            f"python {GOLDEN_DIR / 'build_questions.py'}"
        )
    return [
        GoldenCase(
            id=record["id"],
            question=record["question"],
            sql=record["sql"],
            difficulty=record["difficulty"],
            features=tuple(record.get("features", ())),
            expected_row_count=record.get("expected_row_count"),
            expected_columns=_tuple_or_none(record.get("expected_columns")),
            expected_hash=record.get("expected_hash"),
            expected_ordered_hash=record.get("expected_ordered_hash"),
            source="golden",
        )
        for record in _read_jsonl(path)
    ]


def load_adversarial() -> list[AdversarialCase]:
    """Load the adversarial cases for the verification layer."""
    path = ADVERSARIAL_DIR / "cases.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Regenerate it with: "
            f"python {ADVERSARIAL_DIR / 'build_adversarial.py'}"
        )
    return [
        AdversarialCase(
            id=record["id"],
            sql=record["sql"],
            expect=record["expect"],
            reason=record["reason"],
            note=record.get("note", ""),
            sqlite_parses=record.get("sqlite_parses"),
            sqlite_error=record.get("sqlite_error"),
        )
        for record in _read_jsonl(path)
    ]


def load_spider() -> list[GoldenCase] | None:
    """Load the Spider dev set, or ``None`` if it was not obtained.

    Spider cases have no expected results: the benchmark's SQLite database files
    are not distributed here, so ``is_verified`` is False for every case and
    execution accuracy cannot be computed from this directory. Spider gold SQL
    also routinely uses constructs outside the sql-engine supported subset, so
    filter before use.
    """
    path = SPIDER_DIR / "dev.jsonl"
    if not path.exists():
        return None
    cases = [
        GoldenCase(
            id=record["id"],
            question=record["question"],
            sql=record["sql"],
            difficulty="unknown",
            features=(),
            db_id=record.get("db_id"),
            source="spider",
        )
        for record in _read_jsonl(path)
    ]
    return cases or None


def spider_schemas() -> list[dict[str, Any]] | None:
    """Schemas for the 20 databases used by the Spider dev split, or ``None``."""
    path = SPIDER_DIR / "dev_schemas.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- stats


def _counter(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _main() -> None:
    print("=" * 62)
    print("sql-engine evaluation datasets")
    print("=" * 62)
    print(f"root: {DATA_ROOT}")

    golden = load_golden()
    print(f"\nGOLDEN  {len(golden)} cases")
    print(f"  verified (expected result recorded): "
          f"{sum(1 for c in golden if c.is_verified)}/{len(golden)}")
    print(f"  difficulty: {_counter(c.difficulty for c in golden)}")
    feature_counts = _counter(f for c in golden for f in c.features)
    print(f"  features ({len(feature_counts)} distinct):")
    for name, count in feature_counts.items():
        print(f"     {name:<12} {count}")
    try:
        schema = golden_schema()
        rows = sum(t["row_count"] for t in schema["tables"])
        print(f"  database: {golden_db_path().name}  "
              f"{len(schema['tables'])} tables, {rows} rows, seed {schema['seed']}")
    except FileNotFoundError as exc:
        print(f"  database: MISSING ({exc})")

    adversarial = load_adversarial()
    reject = [c for c in adversarial if c.should_reject]
    print(f"\nADVERSARIAL  {len(adversarial)} cases "
          f"({len(reject)} reject / {len(adversarial) - len(reject)} allow)")
    for name, count in _counter(c.reason for c in adversarial).items():
        print(f"     {name:<20} {count}")
    print(f"  legal SQL that only static analysis can catch: "
          f"{sum(1 for c in adversarial if c.needs_static_analysis)}")

    spider = load_spider()
    if spider is None:
        print("\nSPIDER  not available - see spider/NOT_AVAILABLE.md")
    else:
        print(f"\nSPIDER  {len(spider)} dev cases across "
              f"{len({c.db_id for c in spider})} databases")
        schemas = spider_schemas()
        print(f"  schemas loaded: {len(schemas) if schemas else 0}")
        print("  expected results: none (no DB files shipped) - "
              "execution accuracy CANNOT be computed from this directory")
    print()


if __name__ == "__main__":
    _main()
