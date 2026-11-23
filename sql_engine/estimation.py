"""Cardinality estimation.

This is a textbook selectivity model, not a good one. It exists for one reason: EXPLAIN
has to show row-count estimates that actually *move* when the optimiser pushes a
predicate into a scan, otherwise "the optimiser works" is a claim rather than something
visible. Every constant below is a standard System-R style default and is documented as
such in the README's Limitations section.

The model has no histograms, no correlation handling and no join-order search, so it
will be wrong on correlated predicates. It is used to *display* estimates and to gate
the verifier's scan-size check -- never to choose between plans, because sql-engine has
only one plan shape per query.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ast_nodes import (
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
    UnaryOp,
)
from .binder import qualified
from .catalog import ColumnSchema

#: System-R defaults, used whenever statistics are missing.
DEFAULT_EQUALITY_SELECTIVITY = 0.1
DEFAULT_RANGE_SELECTIVITY = 1.0 / 3.0
DEFAULT_LIKE_SELECTIVITY = 0.25
DEFAULT_NULL_SELECTIVITY = 0.1

_RANGE_OPS = frozenset(
    {BinaryOperator.LT, BinaryOperator.LTE, BinaryOperator.GT, BinaryOperator.GTE}
)


@dataclass(slots=True)
class Estimator:
    """Estimates selectivities and distinct counts from the bound column statistics."""

    #: qualified column name -> schema (with optional stats attached)
    columns: dict[str, ColumnSchema]

    def _schema(self, expr: Expression) -> ColumnSchema | None:
        if isinstance(expr, ColumnRef) and expr.table is not None:
            return self.columns.get(qualified(expr.table, expr.name))
        return None

    def distinct_count(self, expr: Expression, input_rows: int) -> int:
        """Estimated number of distinct values of ``expr`` over ``input_rows`` rows.

        Capped at ``input_rows``: a column cannot have more distinct values than the
        number of rows that survived the filters below it.
        """
        schema = self._schema(expr)
        if schema is not None and schema.stats is not None:
            return max(1, min(input_rows, schema.stats.distinct_count))
        # Fall back to the common heuristic: a derived expression is assumed to have
        # roughly n^0.75 distinct values.
        return max(1, min(input_rows, int(input_rows**0.75) or 1))

    def selectivity(self, predicate: Expression, input_rows: int) -> float:
        """Fraction of ``input_rows`` expected to satisfy ``predicate``, in [0, 1]."""
        value = self._selectivity(predicate, input_rows)
        return min(1.0, max(0.0, value))

    def _selectivity(self, predicate: Expression, rows: int) -> float:
        if isinstance(predicate, Literal):
            if predicate.value is True:
                return 1.0
            if predicate.value in (False, None):
                return 0.0
            return DEFAULT_EQUALITY_SELECTIVITY

        if isinstance(predicate, UnaryOp) and predicate.op == "NOT":
            return 1.0 - self._selectivity(predicate.operand, rows)

        if isinstance(predicate, BinaryOp):
            return self._binary_selectivity(predicate, rows)

        if isinstance(predicate, InList):
            base = min(
                1.0,
                len(predicate.values) / self.distinct_count(predicate.expr, rows),
            )
            return 1.0 - base if predicate.negated else base

        if isinstance(predicate, Between):
            base = self._range_fraction(predicate.expr, predicate.low, predicate.high)
            return 1.0 - base if predicate.negated else base

        if isinstance(predicate, Like):
            base = DEFAULT_LIKE_SELECTIVITY
            if isinstance(predicate.pattern, Literal) and isinstance(predicate.pattern.value, str):
                pattern = predicate.pattern.value
                if not pattern.startswith("%") and not pattern.startswith("_"):
                    # An anchored prefix is far more selective than a free-floating one.
                    base = 0.05
            return 1.0 - base if predicate.negated else base

        if isinstance(predicate, IsNull):
            schema = self._schema(predicate.expr)
            fraction = DEFAULT_NULL_SELECTIVITY
            if schema is not None and schema.stats is not None and rows > 0:
                total = schema.stats.null_count + max(0, rows - schema.stats.null_count)
                if total > 0:
                    fraction = min(1.0, schema.stats.null_count / total)
            return 1.0 - fraction if predicate.negated else fraction

        if isinstance(predicate, (ColumnRef, FunctionCall)):
            return DEFAULT_EQUALITY_SELECTIVITY

        return DEFAULT_EQUALITY_SELECTIVITY

    def _binary_selectivity(self, predicate: BinaryOp, rows: int) -> float:
        op = predicate.op
        if op is BinaryOperator.AND:
            # Independence assumption. Wrong for correlated predicates; documented.
            return self._selectivity(predicate.left, rows) * self._selectivity(
                predicate.right, rows
            )
        if op is BinaryOperator.OR:
            left = self._selectivity(predicate.left, rows)
            right = self._selectivity(predicate.right, rows)
            return left + right - left * right
        if op is BinaryOperator.EQ:
            return self._equality_selectivity(predicate, rows)
        if op is BinaryOperator.NEQ:
            return 1.0 - self._equality_selectivity(predicate, rows)
        if op in _RANGE_OPS:
            return self._comparison_fraction(predicate, rows)
        # Arithmetic used as a predicate: no information, assume the default.
        return DEFAULT_EQUALITY_SELECTIVITY

    def _equality_selectivity(self, predicate: BinaryOp, rows: int) -> float:
        for side, other in ((predicate.left, predicate.right), (predicate.right, predicate.left)):
            schema = self._schema(side)
            if schema is not None and isinstance(other, Literal):
                if schema.stats is not None:
                    return 1.0 / max(1, min(rows or 1, schema.stats.distinct_count))
                return DEFAULT_EQUALITY_SELECTIVITY
        # column = column (a join predicate evaluated as a filter)
        left_ndv = self.distinct_count(predicate.left, rows)
        right_ndv = self.distinct_count(predicate.right, rows)
        return 1.0 / max(1, max(left_ndv, right_ndv))

    def _comparison_fraction(self, predicate: BinaryOp, rows: int) -> float:
        """Fraction for ``col <op> literal`` using the column's min/max when known."""
        column_side, literal_side, flipped = predicate.left, predicate.right, False
        if self._schema(predicate.left) is None and self._schema(predicate.right) is not None:
            column_side, literal_side, flipped = predicate.right, predicate.left, True

        schema = self._schema(column_side)
        if (
            schema is None
            or schema.stats is None
            or schema.stats.minimum is None
            or schema.stats.maximum is None
            or not isinstance(literal_side, Literal)
            or not isinstance(literal_side.value, (int, float))
        ):
            return DEFAULT_RANGE_SELECTIVITY

        low, high = schema.stats.minimum, schema.stats.maximum
        span = high - low
        if span <= 0:
            return DEFAULT_RANGE_SELECTIVITY
        value = float(literal_side.value)
        below = min(1.0, max(0.0, (value - low) / span))

        op = predicate.op
        if flipped:  # `5 > col` means `col < 5`
            op = {
                BinaryOperator.LT: BinaryOperator.GT,
                BinaryOperator.LTE: BinaryOperator.GTE,
                BinaryOperator.GT: BinaryOperator.LT,
                BinaryOperator.GTE: BinaryOperator.LTE,
            }[op]
        if op in (BinaryOperator.LT, BinaryOperator.LTE):
            return below
        return 1.0 - below

    def _range_fraction(self, expr: Expression, low: Expression, high: Expression) -> float:
        schema = self._schema(expr)
        if (
            schema is None
            or schema.stats is None
            or schema.stats.minimum is None
            or schema.stats.maximum is None
            or not isinstance(low, Literal)
            or not isinstance(high, Literal)
            or not isinstance(low.value, (int, float))
            or not isinstance(high.value, (int, float))
        ):
            return DEFAULT_RANGE_SELECTIVITY
        span = schema.stats.maximum - schema.stats.minimum
        if span <= 0:
            return DEFAULT_RANGE_SELECTIVITY
        width = float(high.value) - float(low.value)
        return min(1.0, max(0.0, width / span))
