"""SQLite as a correctness oracle.

Every claim this project makes about correctness reduces to one procedure: run the same
query over the same data in sql-engine and in SQLite, and compare. SQLite is the right
oracle because it is the most-tested SQL implementation in existence and it is in the
standard library, so the comparison costs nothing and can run in CI on any machine.

The interesting work is in :func:`results_match`. A naive ``==`` would report failures
that are not failures (SQLite returns an int where NumPy returns an int64; float sums
differ in the last bit; row order is unspecified without ORDER BY) and would hide ones
that are (a NULL compared against 0). The comparison rules below are the ones that make
the oracle meaningful rather than noisy, and each is justified where it is defined.
"""

from __future__ import annotations

import csv
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow.parquet as pq

from .catalog import Catalog, DataType, TableSchema

#: Relative tolerance for float comparison. Set by the worst case that actually occurs:
#: SQLite sums in a different order than NumPy's pairwise summation, so a SUM over
#: 200k float64 values can disagree in roughly the last two decimal digits.
FLOAT_RELATIVE_TOLERANCE = 1e-9
FLOAT_ABSOLUTE_TOLERANCE = 1e-9

_SQLITE_TYPES: dict[DataType, str] = {
    DataType.INTEGER: "INTEGER",
    DataType.REAL: "REAL",
    DataType.TEXT: "TEXT",
    DataType.BOOLEAN: "INTEGER",
}


def build_sqlite(catalog: Catalog, database: str = ":memory:") -> sqlite3.Connection:
    """Load every table in ``catalog`` into a SQLite database.

    Loading from the same files the engine reads (rather than from a second fixture)
    is what makes the comparison a test of the engine instead of a test of two
    independently-written fixtures agreeing.
    """
    connection = sqlite3.connect(database)
    for table in catalog:
        _create_table(connection, table)
        _load_rows(connection, table)
    connection.commit()
    return connection


def _create_table(connection: sqlite3.Connection, table: TableSchema) -> None:
    columns = ", ".join(
        f'"{column.name}" {_SQLITE_TYPES[column.dtype]}' for column in table.columns
    )
    connection.execute(f'CREATE TABLE "{table.name}" ({columns})')


def _load_rows(connection: sqlite3.Connection, table: TableSchema) -> None:
    placeholders = ", ".join("?" for _ in table.columns)
    statement = f'INSERT INTO "{table.name}" VALUES ({placeholders})'
    connection.executemany(statement, _read_rows(table))


def _read_rows(table: TableSchema) -> Iterable[tuple[Any, ...]]:
    if table.file_format == "parquet":
        arrow = pq.read_table(table.path)
        columns = [arrow.column(name).to_pylist() for name in table.column_names]
        return list(zip(*columns))
    return list(_read_csv_rows(table))


def _read_csv_rows(table: TableSchema) -> Iterable[tuple[Any, ...]]:
    """Read a CSV with the same NULL and type rules the engine's reader applies.

    An empty field is NULL, not the empty string. If the two readers disagreed here,
    every differential test involving a nullable CSV column would fail for a reason that
    has nothing to do with the engine.
    """
    with Path(table.path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)  # header
        for row in reader:
            converted: list[Any] = []
            for value, column in zip(row, table.columns):
                if value == "":
                    converted.append(None)
                elif column.dtype is DataType.INTEGER:
                    converted.append(int(value))
                elif column.dtype is DataType.REAL:
                    converted.append(float(value))
                elif column.dtype is DataType.BOOLEAN:
                    converted.append(1 if value.lower() in ("true", "1") else 0)
                else:
                    converted.append(value)
            yield tuple(converted)


def run_sqlite(connection: sqlite3.Connection, sql: str) -> list[tuple[Any, ...]]:
    """Execute ``sql`` and return the rows as plain tuples."""
    cursor = connection.execute(sql)
    return [tuple(row) for row in cursor.fetchall()]


# --------------------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------------------


def _normalise(value: Any) -> Any:
    """Reduce a cell to a form where equal values compare equal across both engines.

    Booleans become integers because SQLite has no boolean type, and an integral float
    becomes an integer because SQLite's ``SUM`` over integers returns an integer while a
    float path may return ``6.0``. Neither collapse can mask a real difference: 6 and 6.0
    are the same number, and no query in scope can distinguish them.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if value.is_integer() and abs(value) < 2**53:
            return int(value)
        return value
    return value


def _cells_match(left: Any, right: Any) -> bool:
    left, right = _normalise(left), _normalise(right)
    if left is None or right is None:
        # NULL only matches NULL. This is the check that catches an engine returning 0
        # for an empty SUM, which is the most common way to be quietly wrong.
        return left is None and right is None
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(
            float(left),
            float(right),
            rel_tol=FLOAT_RELATIVE_TOLERANCE,
            abs_tol=FLOAT_ABSOLUTE_TOLERANCE,
        )
    return left == right


def rows_match(left: Sequence[Any], right: Sequence[Any]) -> bool:
    """Compare two rows cell by cell."""
    if len(left) != len(right):
        return False
    return all(_cells_match(a, b) for a, b in zip(left, right))


#: Significant digits a float is bucketed to when building a hashable comparison key.
#: Twelve is comfortably tighter than any difference this project should tolerate and
#: comfortably looser than the summation-order difference between NumPy and SQLite.
FLOAT_KEY_DIGITS = 12


def _row_key(row: Sequence[Any]) -> tuple[Any, ...]:
    """A hashable key for multiset comparison, with floats bucketed by *significant*
    digits rather than decimal places.

    Decimal-place rounding was the original implementation and it was wrong: a SUM in
    the hundreds of millions has no decimal places left to round, so two values agreeing
    to fourteen significant digits landed in different buckets. See FAILURES.md #5.

    Bucketing can still split two values that straddle a boundary, so a Counter
    mismatch is treated as inconclusive and rechecked tolerantly by
    :func:`results_match` rather than reported as a failure.
    """
    key: list[Any] = []
    for value in row:
        value = _normalise(value)
        if isinstance(value, float):
            key.append(float(f"{value:.{FLOAT_KEY_DIGITS}g}"))
        else:
            key.append(value)
    return tuple(key)


def _tolerant_multiset_match(
    left: Sequence[Sequence[Any]], right: Sequence[Sequence[Any]]
) -> bool:
    """Greedy pairwise match under :func:`rows_match`.

    Quadratic, so it only ever runs over the rows the fast Counter path could not
    reconcile -- which is a handful of near-boundary floats, never a whole result set.
    """
    remaining = list(right)
    for row in left:
        for index, candidate in enumerate(remaining):
            if rows_match(row, candidate):
                remaining.pop(index)
                break
        else:
            return False
    return not remaining


def results_match(
    left: Sequence[Sequence[Any]],
    right: Sequence[Sequence[Any]],
    *,
    ordered: bool,
) -> tuple[bool, str]:
    """Compare two result sets. Returns ``(matched, explanation)``.

    ``ordered`` must be True only when the query has an ORDER BY that fully determines
    the order. Without one, SQL leaves row order undefined, so comparing as an ordered
    sequence would test an implementation detail rather than correctness -- and would
    fail intermittently as soon as either engine changed its internal ordering.
    """
    if len(left) != len(right):
        return False, f"row count differs: sql-engine={len(left)} sqlite={len(right)}"

    if ordered:
        for index, (a, b) in enumerate(zip(left, right)):
            if not rows_match(a, b):
                return False, f"row {index} differs: sql-engine={a!r} sqlite={b!r}"
        return True, ""

    left_counter = Counter(_row_key(row) for row in left)
    right_counter = Counter(_row_key(row) for row in right)
    if left_counter == right_counter:
        return True, ""

    # The exact-key comparison is a fast filter, not the verdict. Rows it could not pair
    # are rechecked under the float tolerance before anything is called a failure.
    only_left_keys = left_counter - right_counter
    only_right_keys = right_counter - left_counter
    residual_left = [row for row in left if _row_key(row) in only_left_keys]
    residual_right = [row for row in right if _row_key(row) in only_right_keys]
    if _tolerant_multiset_match(residual_left, residual_right):
        return True, ""

    return False, (
        f"multiset differs; only in sql-engine: {residual_left[:3]}; "
        f"only in sqlite: {residual_right[:3]}"
    )


__all__ = [
    "build_sqlite",
    "results_match",
    "rows_match",
    "run_sqlite",
]
