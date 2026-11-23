"""Executor unit tests: batches, three-valued logic, and the vectorised primitives.

These test the pieces in isolation. Whole-query correctness is the differential suite's
job; this file exists to make a failure there *diagnosable*, by pinning down which
primitive broke.
"""

from __future__ import annotations

import numpy as np
import pytest

from sql_engine import Engine
from sql_engine.ast_nodes import BinaryOp, BinaryOperator, ColumnRef, Literal
from sql_engine.errors import ExecutionError
from sql_engine.executor.batch import Batch, Column, concat_batches, split_batch
from sql_engine.executor.expressions import evaluate, sql_like_to_regex, truth_mask
from sql_engine.executor.operators import _ragged_expand, KeyEncoder, to_masked_series

T, F, N = True, False, None


def make_column(values: list, dtype=np.int64) -> Column:
    """Build a column from a list where ``None`` means NULL."""
    valid = np.array([v is not None for v in values], dtype=bool)
    filler = "" if dtype == object else 0
    filled = [filler if v is None else v for v in values]
    array = np.array(filled, dtype=dtype)
    return Column(array, valid)


def make_batch(**columns: Column) -> Batch:
    length = len(next(iter(columns.values()))) if columns else 0
    return Batch(dict(columns), length)


def three_valued(column: Column) -> list:
    return [None if not ok else bool(v) for v, ok in zip(column.values, column.valid)]


# --------------------------------------------------------------------------------------
# Column and Batch
# --------------------------------------------------------------------------------------


def test_column_length_must_match_its_mask():
    with pytest.raises(ExecutionError):
        Column(np.zeros(3), np.ones(2, dtype=bool))


def test_to_pylist_maps_invalid_positions_to_none():
    assert make_column([1, None, 3]).to_pylist() == [1, None, 3]


def test_constant_null_is_invalid_everywhere():
    column = Column.constant(None, 4)
    assert not column.valid.any()


def test_filter_and_take():
    column = make_column([10, 20, 30, 40])
    assert column.filter(np.array([T, F, T, F])).to_pylist() == [10, 30]
    assert column.take(np.array([3, 0])).to_pylist() == [40, 10]


def test_concat_of_all_empty_batches_keeps_the_schema():
    """The bug from FAILURES.md #2: no rows must not mean no columns."""
    empty = make_batch(a=make_column([])).take(np.empty(0, dtype=np.int64))
    merged = concat_batches([empty, empty])
    assert merged.num_rows == 0
    assert set(merged.columns) == {"a"}


def test_split_batch_respects_the_batch_size():
    batch = make_batch(a=make_column(list(range(10))))
    sizes = [chunk.num_rows for chunk in split_batch(batch, 4)]
    assert sizes == [4, 4, 2]


# --------------------------------------------------------------------------------------
# Three-valued logic
# --------------------------------------------------------------------------------------


def boolean_column(values: list) -> Column:
    valid = np.array([v is not None for v in values], dtype=bool)
    filled = np.array([False if v is None else v for v in values], dtype=bool)
    return Column(filled, valid)


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ([T, T, F, F, N, N, T, F, N], [T, F, T, F, T, F, N, N, N],
         [T, F, F, F, N, F, N, F, N]),
    ],
)
def test_and_truth_table(left, right, expected):
    """FALSE AND NULL is FALSE. Getting this wrong changes results silently."""
    batch = make_batch(a=boolean_column(left), b=boolean_column(right))
    result = evaluate(BinaryOp(BinaryOperator.AND, ColumnRef("a"), ColumnRef("b")), batch)
    assert three_valued(result) == expected


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ([T, T, F, F, N, N, T, F, N], [T, F, T, F, T, F, N, N, N],
         [T, T, T, F, T, N, T, N, N]),
    ],
)
def test_or_truth_table(left, right, expected):
    """TRUE OR NULL is TRUE."""
    batch = make_batch(a=boolean_column(left), b=boolean_column(right))
    result = evaluate(BinaryOp(BinaryOperator.OR, ColumnRef("a"), ColumnRef("b")), batch)
    assert three_valued(result) == expected


def test_where_keeps_true_only():
    """UNKNOWN is discarded by WHERE, exactly like FALSE."""
    mask = truth_mask(boolean_column([T, F, N]))
    assert mask.tolist() == [True, False, False]


def test_comparison_with_null_operand_is_null():
    batch = make_batch(a=make_column([1, None, 3]))
    result = evaluate(BinaryOp(BinaryOperator.EQ, ColumnRef("a"), Literal(1)), batch)
    assert three_valued(result) == [True, None, False]


def test_comparison_against_a_null_literal_is_null_not_an_error():
    """`x = NULL` is the classic mistake; it must return nothing, not raise.

    See FAILURES.md #6 -- this raised an ExecutionError before the validity mask was
    checked ahead of the dtype guard.
    """
    batch = make_batch(a=make_column(["x", "y"], dtype=object))
    result = evaluate(BinaryOp(BinaryOperator.EQ, ColumnRef("a"), Literal(None)), batch)
    assert three_valued(result) == [None, None]
    assert truth_mask(result).sum() == 0


def test_text_versus_number_comparison_raises_rather_than_guessing():
    batch = make_batch(a=make_column(["x", "y"], dtype=object))
    with pytest.raises(ExecutionError):
        evaluate(BinaryOp(BinaryOperator.EQ, ColumnRef("a"), Literal(1)), batch)


# --------------------------------------------------------------------------------------
# Arithmetic that has to match SQLite
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "left,right,expected",
    [(-5, 2, -2), (5, 2, 2), (-5, -2, 2), (5, -2, -2), (7, 7, 1)],
)
def test_integer_division_truncates_toward_zero(left, right, expected):
    """Python floors, C and SQLite truncate. See FAILURES.md #4."""
    batch = make_batch(a=make_column([left]), b=make_column([right]))
    result = evaluate(BinaryOp(BinaryOperator.DIV, ColumnRef("a"), ColumnRef("b")), batch)
    assert result.to_pylist() == [expected]


@pytest.mark.parametrize(
    "left,right,expected", [(-5, 2, -1), (5, 2, 1), (-5, -2, -1), (5, -2, 1)]
)
def test_modulo_sign_follows_the_dividend(left, right, expected):
    batch = make_batch(a=make_column([left]), b=make_column([right]))
    result = evaluate(BinaryOp(BinaryOperator.MOD, ColumnRef("a"), ColumnRef("b")), batch)
    assert result.to_pylist() == [expected]


def test_division_by_zero_is_null():
    batch = make_batch(a=make_column([1, 2]), b=make_column([0, 2]))
    result = evaluate(BinaryOp(BinaryOperator.DIV, ColumnRef("a"), ColumnRef("b")), batch)
    assert result.to_pylist() == [None, 1]


def test_real_division_stays_real():
    batch = make_batch(a=make_column([5.0], dtype=np.float64), b=make_column([2]))
    result = evaluate(BinaryOp(BinaryOperator.DIV, ColumnRef("a"), ColumnRef("b")), batch)
    assert result.to_pylist() == [2.5]


# --------------------------------------------------------------------------------------
# LIKE
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern,value,expected",
    [
        ("a%", "abc", True),
        ("a%", "bac", False),
        ("%c", "abc", True),
        ("%b%", "abc", True),
        ("a_c", "abc", True),
        ("a_c", "ac", False),
        ("abc", "ABC", True),  # SQLite's LIKE is ASCII case-insensitive
        ("a.c", "abc", False),  # '.' is literal in LIKE, not a regex wildcard
    ],
)
def test_like_semantics(pattern, value, expected):
    assert bool(sql_like_to_regex(pattern).fullmatch(value)) is expected


# --------------------------------------------------------------------------------------
# Vectorised join primitives
# --------------------------------------------------------------------------------------


def test_ragged_expand_flattens_group_ranges():
    result = _ragged_expand(np.array([10, 40]), np.array([2, 3]))
    assert result.tolist() == [10, 11, 40, 41, 42]


def test_ragged_expand_handles_empty_input():
    assert _ragged_expand(np.array([], dtype=np.int64), np.array([], dtype=np.int64)).size == 0


def test_key_encoder_maps_unseen_probe_values_to_no_match():
    encoder = KeyEncoder([make_column([1, 2, 3])])
    codes, has_null = encoder.encode([make_column([2, 99])])
    assert codes[1] == -1  # 99 was never on the build side
    assert codes[0] != -1
    assert not has_null.any()


def test_key_encoder_marks_null_keys():
    encoder = KeyEncoder([make_column([1, 2])])
    _codes, has_null = encoder.encode([make_column([1, None])])
    assert has_null.tolist() == [False, True]


def test_key_encoder_handles_composite_keys():
    encoder = KeyEncoder([make_column([1, 1, 2]), make_column(["a", "b", "a"], dtype=object)])
    codes, _ = encoder.encode(
        [make_column([1, 2, 1]), make_column(["b", "a", "a"], dtype=object)]
    )
    # (1,'b') and (1,'a') are distinct composite keys and must not collide.
    assert codes[0] != codes[2]
    assert len(set(codes.tolist())) == 3


def test_masked_series_uses_the_fast_path_when_there_are_no_nulls():
    column = make_column([1, 2, 3])
    series = to_masked_series(column)
    assert series.dtype == np.int64  # not promoted to object or float


# --------------------------------------------------------------------------------------
# Batching behaviour
# --------------------------------------------------------------------------------------


#: Queries exercised at every batch size. They cover a streaming pipeline, a blocking
#: aggregate, a join and a sort, because each has its own way of going wrong at a chunk
#: boundary. Run over the small tables so that a batch size of 1 stays affordable.
BATCH_SENSITIVE_QUERIES = [
    "SELECT exchange, COUNT(*) AS n FROM instruments GROUP BY exchange ORDER BY exchange",
    "SELECT symbol FROM instruments WHERE tick_size > 0.1 ORDER BY symbol",
    "SELECT r.region, COUNT(*) AS n FROM traders r GROUP BY r.region ORDER BY r.region",
    (
        "SELECT i.exchange, COUNT(*) AS n FROM instruments i "
        "INNER JOIN instruments j ON i.exchange = j.exchange "
        "GROUP BY i.exchange ORDER BY i.exchange"
    ),
    "SELECT DISTINCT currency FROM instruments ORDER BY currency",
    "SELECT symbol FROM instruments ORDER BY symbol LIMIT 7 OFFSET 11",
]


@pytest.mark.parametrize("size", [1, 3, 7, 64, 4096, 65536])
def test_results_are_independent_of_batch_size(catalog, size):
    """A batch boundary must never change an answer.

    Off-by-one bugs at chunk edges are the characteristic failure of a batched executor,
    and they hide behind any single batch size -- including the default.
    """
    reference = Engine(catalog, batch_size=4096)
    candidate = Engine(catalog, batch_size=size)
    for sql in BATCH_SENSITIVE_QUERIES:
        assert candidate.query(sql).rows == reference.query(sql).rows, (size, sql)


def test_large_table_is_unaffected_by_batch_size(catalog):
    """The same property on the 200k-row fact table, at realistic batch sizes."""
    sql = (
        "SELECT side, COUNT(*) AS n, SUM(quantity) AS q FROM trades "
        "WHERE price > 1000 GROUP BY side ORDER BY side"
    )
    baseline = Engine(catalog, batch_size=4096).query(sql).rows
    for size in (997, 65536):
        assert Engine(catalog, batch_size=size).query(sql).rows == baseline, size


def test_limit_stops_early(catalog):
    """LIMIT must not read the whole table."""
    engine = Engine(catalog, batch_size=1024)
    result = engine.query("SELECT trade_id FROM trades LIMIT 10")
    assert len(result.rows) == 10
    assert result.stats.rows_scanned < 200_000


def test_stats_report_the_pushdown_effect(catalog):
    engine = Engine(catalog)
    result = engine.query("SELECT trade_id FROM trades WHERE trader_id = 1")
    assert result.stats.rows_scanned == 200_000
    assert result.stats.rows_after_scan_filter < result.stats.rows_scanned
    assert result.stats.columns_read == 2
