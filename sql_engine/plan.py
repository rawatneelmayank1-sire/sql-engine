"""Logical plan nodes.

A logical plan says *what* to compute, not how. The executor in ``sql_engine.executor``
turns each node into a physical operator; keeping the two separate is what lets EXPLAIN
print a plan without touching a single byte of data, and what lets the optimiser rewrite
plans structurally.

Every node is immutable. The optimiser returns a new tree rather than mutating the old
one, so ``EXPLAIN`` can print the before and after side by side and the "before" is
genuinely the unoptimised plan rather than a mutated shadow of the "after".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

from .ast_nodes import (
    Expression,
    FunctionCall,
    JoinType,
    OrderByItem,
    SelectItem,
)
from .binder import qualified
from .catalog import TableSchema
from .estimation import Estimator


class LogicalPlan(ABC):
    """Base class for logical plan nodes."""

    __slots__ = ()

    @property
    @abstractmethod
    def children(self) -> tuple[LogicalPlan, ...]:
        """Child plans, left to right."""

    @abstractmethod
    def estimated_rows(self, estimator: Estimator) -> int:
        """Estimated output cardinality. Always >= 0."""

    @abstractmethod
    def describe(self, estimator: Estimator) -> str:
        """One-line description of this node, without its children."""

    def explain(self, estimator: Estimator, indent: int = 0) -> str:
        """Render this node and its subtree as an indented tree with row estimates."""
        pad = "  " * indent
        rows = self.estimated_rows(estimator)
        lines = [f"{pad}{self.describe(estimator)}  (est. rows: {rows:,})"]
        for child in self.children:
            lines.append(child.explain(estimator, indent + 1))
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Scan(LogicalPlan):
    """A base-table read.

    ``projection`` and ``pushed_predicates`` start empty and are filled in by the
    optimiser. They are fields on the *scan* rather than separate nodes because that is
    exactly what pushdown means physically: the reader asks the file for fewer columns
    and drops rows before anything else sees them.
    """

    table: TableSchema
    alias: str
    #: Columns the reader must produce. ``None`` means "all of them" (unoptimised).
    projection: tuple[str, ...] | None = None
    #: Conjuncts evaluated inside the scan, immediately after each batch is read.
    pushed_predicates: tuple[Expression, ...] = ()

    @property
    def children(self) -> tuple[LogicalPlan, ...]:
        return ()

    @property
    def output_columns(self) -> tuple[str, ...]:
        names = self.projection if self.projection is not None else self.table.column_names
        return tuple(qualified(self.alias, name) for name in names)

    def estimated_rows(self, estimator: Estimator) -> int:
        rows = float(self.table.row_count)
        for predicate in self.pushed_predicates:
            rows *= estimator.selectivity(predicate, int(rows))
        return max(0, int(round(rows)))

    def describe(self, estimator: Estimator) -> str:
        name = self.table.name
        if self.alias.lower() != name.lower():
            name = f"{name} AS {self.alias}"
        parts = [f"Scan[{self.table.file_format}] {name}"]
        if self.projection is None:
            parts.append(f"columns=ALL({len(self.table.columns)})")
        else:
            parts.append(f"columns=[{', '.join(self.projection)}]")
        if self.pushed_predicates:
            joined = " AND ".join(str(p) for p in self.pushed_predicates)
            parts.append(f"pushed={joined}")
        return "  ".join(parts)


@dataclass(frozen=True, slots=True)
class Filter(LogicalPlan):
    """Row filter. Holds a conjunct list rather than one expression so the optimiser can
    push individual conjuncts to different places."""

    child: LogicalPlan
    predicates: tuple[Expression, ...]

    @property
    def children(self) -> tuple[LogicalPlan, ...]:
        return (self.child,)

    def estimated_rows(self, estimator: Estimator) -> int:
        rows = float(self.child.estimated_rows(estimator))
        for predicate in self.predicates:
            rows *= estimator.selectivity(predicate, int(rows))
        return max(0, int(round(rows)))

    def describe(self, estimator: Estimator) -> str:
        return "Filter  " + " AND ".join(str(p) for p in self.predicates)


@dataclass(frozen=True, slots=True)
class Project(LogicalPlan):
    """Evaluates the select list."""

    child: LogicalPlan
    items: tuple[SelectItem, ...]

    @property
    def children(self) -> tuple[LogicalPlan, ...]:
        return (self.child,)

    def estimated_rows(self, estimator: Estimator) -> int:
        return self.child.estimated_rows(estimator)

    def describe(self, estimator: Estimator) -> str:
        return "Project  " + ", ".join(str(item) for item in self.items)


@dataclass(frozen=True, slots=True)
class HashJoin(LogicalPlan):
    """Equi-join with an optional residual predicate.

    ``equi_keys`` are the ``left = right`` pairs that become the hash key. Anything else
    in the ON clause lands in ``residual`` and is evaluated after the match. An empty
    ``equi_keys`` is a cartesian product -- representable here so the verifier has
    something concrete to detect, but the parser makes it hard to reach by accident.
    """

    left: LogicalPlan
    right: LogicalPlan
    join_type: JoinType
    equi_keys: tuple[tuple[Expression, Expression], ...]
    residual: tuple[Expression, ...] = ()

    @property
    def children(self) -> tuple[LogicalPlan, ...]:
        return (self.left, self.right)

    @property
    def is_cartesian(self) -> bool:
        return not self.equi_keys

    def estimated_rows(self, estimator: Estimator) -> int:
        left_rows = self.left.estimated_rows(estimator)
        right_rows = self.right.estimated_rows(estimator)
        if self.is_cartesian:
            estimate = float(left_rows) * float(right_rows)
        else:
            # Textbook estimate: |L| * |R| / max(ndv(l_key), ndv(r_key)). With a
            # primary-key/foreign-key pair this collapses to the fact-table cardinality,
            # which is the common case here.
            estimate = float(left_rows) * float(right_rows)
            for left_key, right_key in self.equi_keys:
                ndv = max(
                    estimator.distinct_count(left_key, left_rows),
                    estimator.distinct_count(right_key, right_rows),
                )
                estimate /= max(1, ndv)
        if self.join_type is JoinType.LEFT:
            # Every left row survives, matched or not.
            estimate = max(estimate, float(left_rows))
        for predicate in self.residual:
            estimate *= estimator.selectivity(predicate, int(estimate))
        return max(0, int(round(estimate)))

    def describe(self, estimator: Estimator) -> str:
        if self.is_cartesian:
            keys = "CARTESIAN (no equi-predicate)"
        else:
            keys = " AND ".join(f"{left} = {right}" for left, right in self.equi_keys)
        text = f"HashJoin[{self.join_type.value}]  on {keys}"
        if self.residual:
            text += "  residual=" + " AND ".join(str(p) for p in self.residual)
        return text


@dataclass(frozen=True, slots=True)
class Aggregate(LogicalPlan):
    """GROUP BY with aggregate functions. Blocking: consumes its whole input."""

    child: LogicalPlan
    group_keys: tuple[Expression, ...]
    aggregates: tuple[FunctionCall, ...]

    @property
    def children(self) -> tuple[LogicalPlan, ...]:
        return (self.child,)

    def estimated_rows(self, estimator: Estimator) -> int:
        child_rows = self.child.estimated_rows(estimator)
        if not self.group_keys:
            return 1  # a bare aggregate always produces exactly one row
        estimate = 1.0
        for key in self.group_keys:
            estimate *= estimator.distinct_count(key, child_rows)
        return max(1, min(child_rows, int(round(estimate))))

    def describe(self, estimator: Estimator) -> str:
        keys = ", ".join(str(k) for k in self.group_keys) or "(none)"
        aggs = ", ".join(str(a) for a in self.aggregates) or "(none)"
        return f"Aggregate  keys=[{keys}]  aggs=[{aggs}]"


@dataclass(frozen=True, slots=True)
class Distinct(LogicalPlan):
    """SELECT DISTINCT. Blocking."""

    child: LogicalPlan
    keys: tuple[Expression, ...]

    @property
    def children(self) -> tuple[LogicalPlan, ...]:
        return (self.child,)

    def estimated_rows(self, estimator: Estimator) -> int:
        child_rows = self.child.estimated_rows(estimator)
        estimate = 1.0
        for key in self.keys:
            estimate *= estimator.distinct_count(key, child_rows)
        return max(1, min(child_rows, int(round(estimate)))) if child_rows else 0

    def describe(self, estimator: Estimator) -> str:
        return "Distinct  " + ", ".join(str(k) for k in self.keys)


@dataclass(frozen=True, slots=True)
class Sort(LogicalPlan):
    """ORDER BY. Blocking."""

    child: LogicalPlan
    keys: tuple[OrderByItem, ...]

    @property
    def children(self) -> tuple[LogicalPlan, ...]:
        return (self.child,)

    def estimated_rows(self, estimator: Estimator) -> int:
        return self.child.estimated_rows(estimator)

    def describe(self, estimator: Estimator) -> str:
        return "Sort  " + ", ".join(str(k) for k in self.keys)


@dataclass(frozen=True, slots=True)
class Limit(LogicalPlan):
    """LIMIT/OFFSET. Streaming: stops pulling from its child once satisfied."""

    child: LogicalPlan
    limit: int | None
    offset: int = 0

    @property
    def children(self) -> tuple[LogicalPlan, ...]:
        return (self.child,)

    def estimated_rows(self, estimator: Estimator) -> int:
        rows = max(0, self.child.estimated_rows(estimator) - self.offset)
        return rows if self.limit is None else min(rows, self.limit)

    def describe(self, estimator: Estimator) -> str:
        parts = []
        if self.limit is not None:
            parts.append(f"limit={self.limit}")
        if self.offset:
            parts.append(f"offset={self.offset}")
        return "Limit  " + " ".join(parts)


def iter_plan(plan: LogicalPlan):
    """Yield every node in ``plan``, parents before children."""
    yield plan
    for child in plan.children:
        yield from iter_plan(child)


def scans(plan: LogicalPlan) -> list[Scan]:
    """Every :class:`Scan` in ``plan``, in tree order."""
    return [node for node in iter_plan(plan) if isinstance(node, Scan)]
