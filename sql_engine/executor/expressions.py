"""Vectorised expression evaluation with SQL three-valued logic.

Every function here takes whole columns and returns a whole column. There is no
`for row in rows` anywhere in this module, and that is the point: an expression tree is
walked once per *batch*, not once per row, so the interpreter overhead is paid 4096
times less often.

Three-valued logic is implemented explicitly rather than by leaning on NaN propagation,
because SQL's rules are not NaN's rules. ``FALSE AND NULL`` is FALSE, not NULL, and
getting that wrong silently changes results rather than raising -- exactly the kind of
bug the SQLite differential suite exists to catch.
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd

from ..ast_nodes import (
    Between,
    BinaryOp,
    BinaryOperator,
    ColumnRef,
    Expression,
    FunctionCall,
    InList,
    IsNull,
    Like,
    Literal,
    Star,
    UnaryOp,
)
from ..errors import ExecutionError
from .batch import Batch, Column


def evaluate(expr: Expression, batch: Batch) -> Column:
    """Evaluate ``expr`` over every row of ``batch`` at once.

    Pre-computed columns win: if the batch already carries a column named exactly as
    this expression renders, it is returned untouched. That is how aggregate results
    produced by the group-by operator become visible to the projection and to HAVING
    without a second evaluation path.
    """
    existing = batch.get(str(expr))
    if existing is not None:
        return existing

    if isinstance(expr, Literal):
        return Column.constant(expr.value, batch.num_rows)
    if isinstance(expr, ColumnRef):
        return _resolve_column(expr, batch)
    if isinstance(expr, BinaryOp):
        return _binary(expr, batch)
    if isinstance(expr, UnaryOp):
        return _unary(expr, batch)
    if isinstance(expr, InList):
        return _in_list(expr, batch)
    if isinstance(expr, Between):
        return _between(expr, batch)
    if isinstance(expr, Like):
        return _like(expr, batch)
    if isinstance(expr, IsNull):
        return _is_null(expr, batch)
    if isinstance(expr, FunctionCall):
        raise ExecutionError(
            f"aggregate {expr} was not materialised by the group-by operator; "
            "this is an internal planning error"
        )
    if isinstance(expr, Star):
        raise ExecutionError("* cannot be evaluated as a value expression")
    raise ExecutionError(f"cannot evaluate expression node {type(expr).__name__}")


def _resolve_column(ref: ColumnRef, batch: Batch) -> Column:
    if ref.table is not None:
        column = batch.get(f"{ref.table}.{ref.name}")
        if column is not None:
            return column
    column = batch.get(ref.name)
    if column is not None:
        return column
    raise ExecutionError(
        f"column {ref} is not present in the current batch; "
        f"available: {', '.join(sorted(batch.columns))}"
    )


# --------------------------------------------------------------------------------------
# Boolean helpers
# --------------------------------------------------------------------------------------


def truth_mask(column: Column) -> np.ndarray:
    """Rows a WHERE/HAVING clause keeps: TRUE only, never NULL.

    SQL discards both FALSE and UNKNOWN, which is why this is ``value AND valid`` rather
    than just ``value``.
    """
    values = column.values
    if values.dtype != np.bool_:
        values = _to_boolean(values)
    return values & column.valid


def _to_boolean(values: np.ndarray) -> np.ndarray:
    """Coerce a non-boolean column to truthiness, as SQLite does for ``WHERE 1``."""
    if np.issubdtype(values.dtype, np.number):
        return values != 0
    return np.array([bool(v) for v in values], dtype=bool)


def _and(left: Column, right: Column) -> Column:
    """``AND`` under three-valued logic.

    FALSE dominates: ``FALSE AND NULL`` is FALSE, so the result is *valid* whenever
    either side is a known FALSE, even if the other is NULL.
    """
    left_values = left.values if left.values.dtype == np.bool_ else _to_boolean(left.values)
    right_values = right.values if right.values.dtype == np.bool_ else _to_boolean(right.values)
    known_false = (left.valid & ~left_values) | (right.valid & ~right_values)
    valid = (left.valid & right.valid) | known_false
    # Where a side is NULL the other side must be a known FALSE for the result to be
    # valid at all, and FALSE is what ``&`` produces there -- so no special-casing.
    return Column(left_values & right_values, valid)


def _or(left: Column, right: Column) -> Column:
    """``OR`` under three-valued logic: TRUE dominates."""
    left_values = left.values if left.values.dtype == np.bool_ else _to_boolean(left.values)
    right_values = right.values if right.values.dtype == np.bool_ else _to_boolean(right.values)
    known_true = (left.valid & left_values) | (right.valid & right_values)
    valid = (left.valid & right.valid) | known_true
    return Column(left_values | right_values, valid)


def _not(operand: Column) -> Column:
    values = operand.values if operand.values.dtype == np.bool_ else _to_boolean(operand.values)
    return Column(~values, operand.valid.copy())


# --------------------------------------------------------------------------------------
# Binary operators
# --------------------------------------------------------------------------------------

_COMPARISON_KERNELS: dict[BinaryOperator, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    BinaryOperator.EQ: np.equal,
    BinaryOperator.NEQ: np.not_equal,
    BinaryOperator.LT: np.less,
    BinaryOperator.LTE: np.less_equal,
    BinaryOperator.GT: np.greater,
    BinaryOperator.GTE: np.greater_equal,
}


def _binary(expr: BinaryOp, batch: Batch) -> Column:
    if expr.op is BinaryOperator.AND:
        return _and(evaluate(expr.left, batch), evaluate(expr.right, batch))
    if expr.op is BinaryOperator.OR:
        return _or(evaluate(expr.left, batch), evaluate(expr.right, batch))

    left = evaluate(expr.left, batch)
    right = evaluate(expr.right, batch)

    if expr.op in _COMPARISON_KERNELS:
        return _compare(expr.op, left, right)
    return _arithmetic(expr.op, left, right)


def _compare(op: BinaryOperator, left: Column, right: Column) -> Column:
    """Comparison. NULL on either side yields NULL, never TRUE or FALSE."""
    left_values, right_values = _align(left.values, right.values)
    valid = left.valid & right.valid

    if not valid.any():
        # Every row has a NULL on at least one side, so every answer is UNKNOWN
        # regardless of the values. Returning here also handles `col = NULL`, where the
        # NULL literal carries no usable dtype and would otherwise trip the TEXT/numeric
        # guard below. See FAILURES.md #6.
        return Column(np.zeros(valid.shape[0], dtype=bool), valid)

    kernel = _COMPARISON_KERNELS[op]
    text_sides = (left_values.dtype == object, right_values.dtype == object)
    if text_sides[0] != text_sides[1]:
        # SQLite would apply type affinity here and quietly return FALSE. Raising
        # instead surfaces the mistake (usually a quoted number, or an LLM comparing a
        # date column to an integer) rather than producing a plausible empty result.
        raise ExecutionError(
            "cannot compare a TEXT column with a numeric value; "
            "quote the literal or compare against a numeric column"
        )
    if left_values.dtype == object or right_values.dtype == object:
        # Object arrays: NumPy's comparison ufuncs fall back to elementwise Python
        # comparison, which raises on a None/str mix. Substituting a safe placeholder
        # under the invalid positions keeps the kernel total.
        left_values = np.where(valid, left_values, "")
        right_values = np.where(valid, right_values, "")
        result = _object_compare(op, left_values, right_values)
    else:
        with np.errstate(invalid="ignore"):
            result = kernel(left_values, right_values)
    return Column(np.asarray(result, dtype=bool), valid)


def _object_compare(op: BinaryOperator, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Comparison over TEXT columns.

    pandas' array comparison is used because it dispatches to a C loop; NumPy's ``>``
    on object arrays would still work but goes through the slower generic path.
    """
    left_series = pd.Series(left, dtype=object)
    right_series = pd.Series(right, dtype=object)
    if op is BinaryOperator.EQ:
        return (left_series == right_series).to_numpy(dtype=bool)
    if op is BinaryOperator.NEQ:
        return (left_series != right_series).to_numpy(dtype=bool)
    if op is BinaryOperator.LT:
        return (left_series < right_series).to_numpy(dtype=bool)
    if op is BinaryOperator.LTE:
        return (left_series <= right_series).to_numpy(dtype=bool)
    if op is BinaryOperator.GT:
        return (left_series > right_series).to_numpy(dtype=bool)
    return (left_series >= right_series).to_numpy(dtype=bool)


def _align(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bring two arrays to a common comparable dtype.

    Only the numeric/boolean side is promoted. A TEXT-vs-INTEGER comparison is left
    alone and will raise, rather than silently applying SQLite's type-affinity rules --
    those rules are surprising enough that quietly imitating them would hide real bugs.
    """
    if left.dtype == right.dtype:
        return left, right
    if left.dtype == object or right.dtype == object:
        return left, right
    if np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number):
        common = np.promote_types(left.dtype, right.dtype)
        return left.astype(common), right.astype(common)
    return left, right


def _arithmetic(op: BinaryOperator, left: Column, right: Column) -> Column:
    left_values, right_values = _align(left.values, right.values)
    if left_values.dtype == object or right_values.dtype == object:
        raise ExecutionError(f"arithmetic operator {op.value} is not defined for TEXT columns")

    valid = left.valid & right.valid
    both_integer = np.issubdtype(left_values.dtype, np.integer) and np.issubdtype(
        right_values.dtype, np.integer
    )

    if op is BinaryOperator.ADD:
        return Column(left_values + right_values, valid)
    if op is BinaryOperator.SUB:
        return Column(left_values - right_values, valid)
    if op is BinaryOperator.MUL:
        return Column(left_values * right_values, valid)
    if op is BinaryOperator.DIV:
        return _divide(left_values, right_values, valid, both_integer)
    if op is BinaryOperator.MOD:
        return _modulo(left_values, right_values, valid)
    raise ExecutionError(f"unsupported arithmetic operator {op.value}")


def _divide(
    left: np.ndarray, right: np.ndarray, valid: np.ndarray, both_integer: bool
) -> Column:
    """Division matching SQLite: integer/integer truncates toward zero, x/0 is NULL.

    Python and NumPy both floor, so ``-5 / 2`` is -3 for them and -2 for SQLite. The
    difference only shows up on negative operands, which is precisely why it survives
    casual testing and gets caught by the differential suite.
    """
    zero = right == 0
    safe_right = np.where(zero, 1, right)
    if both_integer:
        quotient = np.trunc(left.astype(np.float64) / safe_right.astype(np.float64))
        result = quotient.astype(np.int64)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = left.astype(np.float64) / safe_right.astype(np.float64)
    return Column(result, valid & ~zero)


def _modulo(left: np.ndarray, right: np.ndarray, valid: np.ndarray) -> Column:
    """Remainder matching SQLite/C: the sign follows the dividend, and ``x % 0`` is NULL."""
    zero = right == 0
    safe_right = np.where(zero, 1, right)
    result = np.fmod(left, safe_right)
    if np.issubdtype(left.dtype, np.integer) and np.issubdtype(right.dtype, np.integer):
        result = result.astype(np.int64)
    return Column(result, valid & ~zero)


def _unary(expr: UnaryOp, batch: Batch) -> Column:
    operand = evaluate(expr.operand, batch)
    if expr.op == "NOT":
        return _not(operand)
    if expr.op == "-":
        if operand.values.dtype == object:
            raise ExecutionError("unary minus is not defined for TEXT columns")
        return Column(-operand.values, operand.valid.copy())
    raise ExecutionError(f"unsupported unary operator {expr.op}")


# --------------------------------------------------------------------------------------
# Predicates
# --------------------------------------------------------------------------------------


def _in_list(expr: InList, batch: Batch) -> Column:
    """``x IN (...)`` with SQL's NULL rule.

    If x matches nothing but the list contains a NULL, the answer is UNKNOWN rather than
    FALSE -- because the NULL *might* have been the match. This is the single most
    commonly mis-implemented corner of IN.
    """
    target = evaluate(expr.expr, batch)
    literals = [evaluate(value, batch) for value in expr.values]

    matched = np.zeros(batch.num_rows, dtype=bool)
    saw_null_operand = np.zeros(batch.num_rows, dtype=bool)
    for candidate in literals:
        comparison = _compare(BinaryOperator.EQ, target, candidate)
        matched |= comparison.values & comparison.valid
        saw_null_operand |= ~candidate.valid

    valid = target.valid & (matched | ~saw_null_operand)
    if expr.negated:
        return Column(~matched, valid)
    return Column(matched, valid)


def _between(expr: Between, batch: Batch) -> Column:
    lower = _compare(BinaryOperator.GTE, evaluate(expr.expr, batch), evaluate(expr.low, batch))
    upper = _compare(BinaryOperator.LTE, evaluate(expr.expr, batch), evaluate(expr.high, batch))
    result = _and(lower, upper)
    return _not(result) if expr.negated else result


def _is_null(expr: IsNull, batch: Batch) -> Column:
    operand = evaluate(expr.expr, batch)
    is_null = ~operand.valid
    values = ~is_null if expr.negated else is_null
    # IS NULL is the one predicate that is never itself NULL.
    return Column(values, np.ones(batch.num_rows, dtype=bool))


_LIKE_SPECIALS = re.compile(r"[.^$*+?{}\[\]\\|()]")


def sql_like_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a SQL LIKE pattern to an anchored regex.

    ``%`` is any run of characters, ``_`` is exactly one, everything else is literal.
    The result is compiled case-insensitively because SQLite's LIKE is case-insensitive
    for ASCII by default -- matching that is required for the differential tests to mean
    anything.
    """
    out: list[str] = []
    for char in pattern:
        if char == "%":
            out.append(".*")
        elif char == "_":
            out.append(".")
        elif _LIKE_SPECIALS.match(char):
            out.append("\\" + char)
        else:
            out.append(char)
    return re.compile("".join(out), re.DOTALL | re.IGNORECASE)


def _like(expr: Like, batch: Batch) -> Column:
    target = evaluate(expr.expr, batch)
    pattern_column = evaluate(expr.pattern, batch)

    if not isinstance(expr.pattern, Literal) or not isinstance(expr.pattern.value, str):
        raise ExecutionError("LIKE requires a constant string pattern")

    series = pd.Series(target.as_nullable_object(), dtype=object)
    pattern = expr.pattern.value
    body = pattern.strip("%")

    # Fast paths avoid the regex engine for the three shapes that cover almost every
    # real LIKE. `case=False` keeps them consistent with SQLite's ASCII-insensitive LIKE.
    if "%" not in body and "_" not in body:
        if pattern.startswith("%") and pattern.endswith("%") and len(pattern) >= 2:
            matched = series.str.contains(body, case=False, regex=False, na=False)
        elif pattern.endswith("%") and not pattern.startswith("%"):
            matched = series.str.lower().str.startswith(body.lower(), na=False)
        elif pattern.startswith("%") and not pattern.endswith("%"):
            matched = series.str.lower().str.endswith(body.lower(), na=False)
        else:
            matched = series.str.lower() == pattern.lower()
            matched = matched.fillna(False)
    else:
        matched = series.str.fullmatch(
            sql_like_to_regex(pattern).pattern, case=False, na=False
        )

    values = np.asarray(matched, dtype=bool)
    if expr.negated:
        values = ~values
    return Column(values, target.valid & pattern_column.valid)
