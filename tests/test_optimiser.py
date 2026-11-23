"""Optimiser tests.

Two things need proving about a rewrite rule, and only one of them is interesting:

1. it fires (easy, and the weaker claim);
2. it never changes an answer (the claim that matters).

Claim 2 is covered here for the tricky outer-join cases and, exhaustively, by
``test_differential.py``, which runs the whole corpus twice with the rules on and off.
"""

from __future__ import annotations

import pytest

from sql_engine import Engine
from sql_engine.ast_nodes import JoinType
from sql_engine.optimiser import Optimiser
from sql_engine.plan import Filter, HashJoin, Scan, iter_plan


def scans(plan) -> list[Scan]:
    return [node for node in iter_plan(plan) if isinstance(node, Scan)]


def joins(plan) -> list[HashJoin]:
    return [node for node in iter_plan(plan) if isinstance(node, HashJoin)]


def filters(plan) -> list[Filter]:
    return [node for node in iter_plan(plan) if isinstance(node, Filter)]


# --------------------------------------------------------------------------------------
# Predicate pushdown
# --------------------------------------------------------------------------------------


def test_single_table_predicate_reaches_the_scan(engine: Engine):
    compiled = engine.compile("SELECT trade_id FROM trades WHERE trader_id = 7 LIMIT 10")
    assert filters(compiled.logical_plan), "unoptimised plan should have a Filter"
    assert not filters(compiled.optimised_plan), "Filter should have dissolved into the scan"
    assert len(scans(compiled.optimised_plan)[0].pushed_predicates) == 1


def test_conjuncts_are_split_and_routed_independently(engine: Engine):
    """`a.x = 1 AND b.y = 2` must reach two different scans."""
    compiled = engine.compile(
        "SELECT t.trade_id FROM trades t INNER JOIN instruments i "
        "ON t.instrument_id = i.instrument_id "
        "WHERE t.trader_id = 7 AND i.exchange = 'CME' LIMIT 10"
    )
    pushed = {scan.alias: len(scan.pushed_predicates) for scan in scans(compiled.optimised_plan)}
    assert pushed == {"t": 1, "i": 1}
    assert not filters(compiled.optimised_plan)


def test_predicate_spanning_both_sides_stays_above_the_join(engine: Engine):
    compiled = engine.compile(
        "SELECT t.trade_id FROM trades t INNER JOIN instruments i "
        "ON t.instrument_id = i.instrument_id WHERE t.price > i.tick_size LIMIT 10"
    )
    assert all(not scan.pushed_predicates for scan in scans(compiled.optimised_plan))


def test_predicate_is_never_pushed_through_limit(engine: Engine):
    """Filter-then-limit and limit-then-filter are different queries.

    SQL's clause order means a plan built from source text never puts a Filter above a
    Limit, so this builds that shape by hand and checks the rule refuses to move the
    predicate down. Without the guard the rewrite would be silently wrong on every
    query the optimiser is later extended to handle.
    """
    from sql_engine.ast_nodes import BinaryOp, BinaryOperator, ColumnRef, Literal
    from sql_engine.optimiser import _push_predicates
    from sql_engine.plan import Limit

    scan = Scan(table=engine.catalog.table("trades"), alias="trades")
    limited = Limit(scan, limit=5)
    predicate = BinaryOp(BinaryOperator.GT, ColumnRef("price", "trades"), Literal(100))

    rewritten = _push_predicates(limited, (predicate,))

    assert isinstance(rewritten, Filter), "the predicate must stay above the LIMIT"
    assert isinstance(rewritten.child, Limit)
    assert not scans(rewritten)[0].pushed_predicates


def test_having_on_a_group_key_is_pushed_below_the_aggregate(engine: Engine):
    """`GROUP BY side HAVING side = 'BUY'` is really a WHERE clause."""
    compiled = engine.compile(
        "SELECT side, COUNT(*) FROM trades GROUP BY side HAVING side = 'BUY' LIMIT 10"
    )
    assert len(scans(compiled.optimised_plan)[0].pushed_predicates) == 1


def test_having_on_an_aggregate_is_not_pushed(engine: Engine):
    compiled = engine.compile(
        "SELECT side, COUNT(*) FROM trades GROUP BY side HAVING COUNT(*) > 5 LIMIT 10"
    )
    assert not scans(compiled.optimised_plan)[0].pushed_predicates
    assert filters(compiled.optimised_plan)


# --------------------------------------------------------------------------------------
# Outer-join asymmetry -- the classic pushdown trap
# --------------------------------------------------------------------------------------


def test_where_on_the_preserved_side_of_a_left_join_is_pushed(engine: Engine):
    compiled = engine.compile(
        "SELECT r.trader_id FROM traders r LEFT JOIN trades t "
        "ON r.trader_id = t.trader_id WHERE r.region = 'EMEA' LIMIT 10"
    )
    by_alias = {s.alias: s for s in scans(compiled.optimised_plan)}
    assert len(by_alias["r"].pushed_predicates) == 1


def test_where_on_the_null_supplying_side_of_a_left_join_is_not_pushed(engine: Engine):
    """Pushing it would discard rows the outer join is required to keep.

    `WHERE t.price > 3990` after a LEFT JOIN removes unmatched left rows, because NULL
    fails the comparison. Push it into the right scan and those left rows come back with
    NULLs instead -- a different answer.
    """
    compiled = engine.compile(
        "SELECT r.trader_id FROM traders r LEFT JOIN trades t "
        "ON r.trader_id = t.trader_id WHERE t.price > 3990 LIMIT 10"
    )
    by_alias = {s.alias: s for s in scans(compiled.optimised_plan)}
    assert not by_alias["t"].pushed_predicates
    assert filters(compiled.optimised_plan)


def test_on_clause_predicate_on_the_null_supplying_side_is_pushed(engine: Engine):
    """The mirror image: an ON predicate filters matches, so it *may* be pushed right."""
    compiled = engine.compile(
        "SELECT r.trader_id FROM traders r LEFT JOIN trades t "
        "ON r.trader_id = t.trader_id AND t.price > 3990 LIMIT 10"
    )
    by_alias = {s.alias: s for s in scans(compiled.optimised_plan)}
    assert len(by_alias["t"].pushed_predicates) == 1


def test_on_clause_predicate_on_the_preserved_side_is_not_pushed_for_left_join(engine: Engine):
    compiled = engine.compile(
        "SELECT r.trader_id FROM traders r LEFT JOIN trades t "
        "ON r.trader_id = t.trader_id AND r.seniority > 3 LIMIT 10"
    )
    by_alias = {s.alias: s for s in scans(compiled.optimised_plan)}
    assert not by_alias["r"].pushed_predicates
    assert joins(compiled.optimised_plan)[0].residual


# --------------------------------------------------------------------------------------
# Projection pruning
# --------------------------------------------------------------------------------------


def test_only_referenced_columns_are_read(engine: Engine):
    compiled = engine.compile("SELECT trade_id, price FROM trades LIMIT 10")
    assert scans(compiled.optimised_plan)[0].projection == ("trade_id", "price")
    assert scans(compiled.logical_plan)[0].projection is None


def test_columns_used_only_in_where_are_still_read(engine: Engine):
    compiled = engine.compile("SELECT trade_id FROM trades WHERE price > 10 LIMIT 10")
    assert set(scans(compiled.optimised_plan)[0].projection) == {"trade_id", "price"}


def test_join_keys_are_kept(engine: Engine):
    compiled = engine.compile(
        "SELECT i.symbol FROM trades t INNER JOIN instruments i "
        "ON t.instrument_id = i.instrument_id LIMIT 10"
    )
    by_alias = {s.alias: set(s.projection) for s in scans(compiled.optimised_plan)}
    assert by_alias["t"] == {"instrument_id"}
    assert by_alias["i"] == {"instrument_id", "symbol"}


def test_count_star_reads_exactly_one_column(engine: Engine):
    """No column values are needed, but a row count still requires reading something."""
    compiled = engine.compile("SELECT COUNT(*) FROM trades LIMIT 10")
    assert len(scans(compiled.optimised_plan)[0].projection) == 1


def test_aggregate_is_a_barrier_for_column_requirements(engine: Engine):
    compiled = engine.compile(
        "SELECT side, SUM(quantity) FROM trades GROUP BY side ORDER BY side LIMIT 10"
    )
    assert set(scans(compiled.optimised_plan)[0].projection) == {"side", "quantity"}


def test_report_counts_what_it_claims(engine: Engine):
    compiled = engine.compile("SELECT trade_id FROM trades WHERE trader_id = 1 LIMIT 10")
    report = compiled.report
    assert report.predicates_pushed_into_scans == 1
    assert report.columns_read_before == 9  # every column of `trades`
    assert report.columns_read_after == 2  # trade_id + trader_id
    assert report.columns_pruned == 7


# --------------------------------------------------------------------------------------
# Estimation and EXPLAIN
# --------------------------------------------------------------------------------------


def test_pushdown_lowers_the_row_estimate(engine: Engine):
    compiled = engine.compile("SELECT trade_id FROM trades WHERE trader_id = 1 LIMIT 1000")
    unoptimised = compiled.logical_plan.estimated_rows(compiled.estimator)
    optimised = compiled.optimised_plan.estimated_rows(compiled.estimator)
    assert optimised <= unoptimised


def test_explain_shows_both_plans_and_the_summary(engine: Engine):
    text = engine.explain("SELECT trade_id FROM trades WHERE trader_id = 1 LIMIT 10")
    assert "before optimisation" in text
    assert "after optimisation" in text
    assert "est. rows" in text
    assert "predicate pushdown" in text


def test_disabling_the_rules_leaves_the_plan_alone(catalog):
    engine = Engine(catalog, optimiser=Optimiser(predicate_pushdown=False, projection_pruning=False))
    compiled = engine.compile("SELECT trade_id FROM trades WHERE trader_id = 1 LIMIT 10")
    assert filters(compiled.optimised_plan)
    assert scans(compiled.optimised_plan)[0].projection is None
    assert compiled.report.predicates_pushed_into_scans == 0


def test_cartesian_join_is_representable_and_labelled(engine: Engine):
    """The plan must be able to *say* cartesian, or the verifier has nothing to detect."""
    compiled = engine.compile(
        "SELECT t.trade_id FROM trades t INNER JOIN instruments i ON 1 = 1 LIMIT 10"
    )
    join = joins(compiled.optimised_plan)[0]
    assert join.is_cartesian
    assert "CARTESIAN" in join.describe(compiled.estimator)
    assert join.join_type is JoinType.INNER
