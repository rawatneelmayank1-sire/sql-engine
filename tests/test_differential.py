"""Differential tests: sql-engine versus SQLite over identical data.

This is the file that makes every other claim in the project checkable. If a query
returns different rows here than SQLite returns, sql-engine is wrong -- there is no
appeal, no "well, our semantics differ", no tolerance for interpretation. SQLite is the
specification.

Three passes over the same corpus:

1. sql-engine (optimised) vs SQLite -- correctness;
2. sql-engine (optimised) vs sql-engine (rules disabled) -- the optimiser preserves meaning;
3. property-based tests over generated predicates -- shapes nobody thought to write down.
"""

from __future__ import annotations

import sqlite3

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from sql_engine import Engine
from sql_engine.sqlite_oracle import results_match, run_sqlite

from .corpus import CORPUS, is_ordered


@pytest.mark.parametrize("name,sql", CORPUS, ids=[name for name, _ in CORPUS])
def test_matches_sqlite(
    name: str, sql: str, engine: Engine, sqlite_connection: sqlite3.Connection
):
    """Every corpus query must return what SQLite returns."""
    actual = engine.query(sql)
    expected = run_sqlite(sqlite_connection, sql)
    matched, explanation = results_match(actual.rows, expected, ordered=is_ordered(sql))
    assert matched, f"{name}: {explanation}"


@pytest.mark.parametrize("name,sql", CORPUS, ids=[name for name, _ in CORPUS])
def test_optimiser_preserves_results(
    name: str, sql: str, engine: Engine, unoptimised_engine: Engine
):
    """Predicate pushdown and projection pruning must not change a single row.

    Run against the *same engine code* with the rules switched off, so a difference can
    only come from the rewrite rules themselves.
    """
    optimised = engine.query(sql)
    plain = unoptimised_engine.query(sql)
    matched, explanation = results_match(optimised.rows, plain.rows, ordered=is_ordered(sql))
    assert matched, f"{name}: optimiser changed the answer -- {explanation}"


def test_optimiser_actually_does_something(engine: Engine, unoptimised_engine: Engine):
    """Guards the test above from being vacuous.

    If the optimiser were a no-op, `test_optimiser_preserves_results` would pass
    trivially and prove nothing.
    """
    sql = "SELECT trade_id FROM trades WHERE trader_id = 3 LIMIT 100"
    assert engine.compile(sql).report.columns_pruned > 0
    assert engine.compile(sql).report.predicates_pushed_into_scans > 0
    assert unoptimised_engine.compile(sql).report.predicates_pushed_into_scans == 0


# --------------------------------------------------------------------------------------
# Property-based: generated predicates, both engines, same answer
# --------------------------------------------------------------------------------------

NUMERIC_COLUMNS = ["quantity", "price", "trade_id", "trader_id", "instrument_id"]
TEXT_COLUMNS = ["side", "venue", "trade_date"]
COMPARISONS = ["=", "<>", "<", "<=", ">", ">="]

numeric_atom = st.builds(
    lambda column, op, value: f"{column} {op} {value}",
    st.sampled_from(NUMERIC_COLUMNS),
    st.sampled_from(COMPARISONS),
    st.integers(min_value=-600, max_value=4000),
)

text_atom = st.builds(
    lambda column, op, value: f"{column} {op} '{value}'",
    st.sampled_from(TEXT_COLUMNS),
    st.sampled_from(["=", "<>"]),
    st.sampled_from(["BUY", "SELL", "SCREEN", "BLOCK", "TAS", "2025-06-01", ""]),
)

null_atom = st.builds(
    lambda column, negated: f"{column} IS {'NOT ' if negated else ''}NULL",
    st.sampled_from(["venue", "side", "price"]),
    st.booleans(),
)

between_atom = st.builds(
    lambda column, low, span: f"{column} BETWEEN {low} AND {low + span}",
    st.sampled_from(NUMERIC_COLUMNS),
    st.integers(min_value=-500, max_value=3000),
    st.integers(min_value=0, max_value=800),
)

in_atom = st.builds(
    lambda column, values: f"{column} IN ({', '.join(str(v) for v in values)})",
    st.sampled_from(["trader_id", "instrument_id", "quantity"]),
    st.lists(st.integers(min_value=-50, max_value=120), min_size=1, max_size=5),
)

atom = st.one_of(numeric_atom, text_atom, null_atom, between_atom, in_atom)


def _combine(parts: list[str], operators: list[str]) -> str:
    expression = parts[0]
    for operator, part in zip(operators, parts[1:]):
        expression = f"({expression} {operator} {part})"
    return expression


predicate = st.builds(
    _combine,
    st.lists(atom, min_size=1, max_size=4),
    st.lists(st.sampled_from(["AND", "OR"]), min_size=3, max_size=3),
).map(lambda text: text)

negated_predicate = st.one_of(predicate, predicate.map(lambda p: f"NOT ({p})"))


@settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(where=negated_predicate)
def test_generated_predicates_match_sqlite(
    where: str, engine: Engine, sqlite_connection: sqlite3.Connection
):
    """Randomly assembled WHERE clauses must agree with SQLite.

    Three-valued logic is where hand-written tests run out of imagination: NULL
    interacts with NOT, OR and IN in ways that are easy to reason about one at a time
    and easy to get wrong in combination. Generating the combinations is the only way to
    cover them.
    """
    sql = f"SELECT COUNT(*) AS n, SUM(quantity) AS q FROM trades WHERE {where}"
    actual = engine.query(sql)
    expected = run_sqlite(sqlite_connection, sql)
    matched, explanation = results_match(actual.rows, expected, ordered=False)
    assert matched, f"WHERE {where}: {explanation}"


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    where=predicate,
    group=st.sampled_from(["side", "venue", "trader_id", "instrument_id"]),
)
def test_generated_group_by_matches_sqlite(
    where: str, group: str, engine: Engine, sqlite_connection: sqlite3.Connection
):
    sql = (
        f"SELECT {group}, COUNT(*) AS n, SUM(quantity) AS q, AVG(price) AS p, "
        f"MIN(price) AS lo, MAX(price) AS hi FROM trades WHERE {where} "
        f"GROUP BY {group} ORDER BY {group}"
    )
    actual = engine.query(sql)
    expected = run_sqlite(sqlite_connection, sql)
    matched, explanation = results_match(actual.rows, expected, ordered=True)
    assert matched, f"{sql}: {explanation}"


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    limit=st.integers(min_value=0, max_value=25),
    offset=st.integers(min_value=0, max_value=25),
    direction=st.sampled_from(["ASC", "DESC"]),
    column=st.sampled_from(["price", "quantity", "side", "venue"]),
)
def test_generated_order_limit_matches_sqlite(
    limit: int,
    offset: int,
    direction: str,
    column: str,
    engine: Engine,
    sqlite_connection: sqlite3.Connection,
):
    """Ordering, NULL placement and paging together.

    ``trade_id`` breaks every tie so the expected order is total, which is what makes an
    *ordered* comparison legitimate here.
    """
    sql = (
        f"SELECT trade_id, {column} FROM trades "
        f"ORDER BY {column} {direction}, trade_id ASC LIMIT {limit} OFFSET {offset}"
    )
    actual = engine.query(sql)
    expected = run_sqlite(sqlite_connection, sql)
    matched, explanation = results_match(actual.rows, expected, ordered=True)
    assert matched, f"{sql}: {explanation}"
