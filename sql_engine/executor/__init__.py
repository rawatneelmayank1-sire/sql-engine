"""Physical planning: logical plan -> operator tree -> rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from ..errors import ExecutionError
from ..plan import (
    Aggregate,
    Distinct,
    Filter,
    HashJoin,
    Limit,
    LogicalPlan,
    Project,
    Scan,
    Sort,
)
from ..planner import project_output_keys
from .batch import DEFAULT_BATCH_SIZE, Batch, Column
from .operators import (
    AggregateOperator,
    DistinctOperator,
    FilterOperator,
    HashJoinOperator,
    LimitOperator,
    Operator,
    ProjectOperator,
    ScanOperator,
    SortOperator,
)

__all__ = [
    "Batch",
    "Column",
    "DEFAULT_BATCH_SIZE",
    "ExecutionStats",
    "Operator",
    "ResultSet",
    "build_operator",
    "run",
]


@dataclass(slots=True)
class ExecutionStats:
    """Counters collected while a plan runs.

    ``rows_scanned`` versus ``rows_emitted`` on the scans is the direct measurement of
    what predicate pushdown bought on this particular query -- not an estimate.
    """

    rows_scanned: int = 0
    rows_after_scan_filter: int = 0
    rows_returned: int = 0
    columns_read: int = 0
    batches: int = 0

    @property
    def scan_filter_ratio(self) -> float:
        if not self.rows_scanned:
            return 1.0
        return self.rows_after_scan_filter / self.rows_scanned


@dataclass(slots=True)
class ResultSet:
    """A materialised query result."""

    column_names: tuple[str, ...]
    rows: list[tuple[Any, ...]]
    stats: ExecutionStats = field(default_factory=ExecutionStats)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [dict(zip(self.column_names, row)) for row in self.rows]


def build_operator(plan: LogicalPlan, batch_size: int = DEFAULT_BATCH_SIZE) -> Operator:
    """Translate a logical plan into an operator tree.

    One physical operator per logical node: sql-engine does not choose between, say, a hash
    and a sort-merge join, so there is no physical planning decision to make. Keeping the
    mapping trivial is deliberate -- it means EXPLAIN's logical output is a faithful
    description of what actually runs.
    """
    if isinstance(plan, Scan):
        return ScanOperator(
            table=plan.table,
            alias=plan.alias,
            columns=plan.projection,
            predicates=plan.pushed_predicates,
            batch_size=batch_size,
        )
    if isinstance(plan, Filter):
        return FilterOperator(build_operator(plan.child, batch_size), plan.predicates)
    if isinstance(plan, Project):
        keys = project_output_keys(plan.items)
        return ProjectOperator(
            build_operator(plan.child, batch_size),
            [item.expr for item in plan.items],
            keys,
        )
    if isinstance(plan, HashJoin):
        return HashJoinOperator(
            left=build_operator(plan.left, batch_size),
            right=build_operator(plan.right, batch_size),
            join_type=plan.join_type,
            equi_keys=plan.equi_keys,
            residual=plan.residual,
            batch_size=batch_size,
        )
    if isinstance(plan, Aggregate):
        return AggregateOperator(
            build_operator(plan.child, batch_size), plan.group_keys, plan.aggregates
        )
    if isinstance(plan, Distinct):
        return DistinctOperator(build_operator(plan.child, batch_size), plan.keys)
    if isinstance(plan, Sort):
        return SortOperator(build_operator(plan.child, batch_size), plan.keys)
    if isinstance(plan, Limit):
        return LimitOperator(build_operator(plan.child, batch_size), plan.limit, plan.offset)
    raise ExecutionError(f"no physical operator for logical node {type(plan).__name__}")


def _iter_operators(operator: Operator) -> Iterator[Operator]:
    yield operator
    for attribute in ("child", "left", "right"):
        child = getattr(operator, attribute, None)
        if isinstance(child, Operator):
            yield from _iter_operators(child)


def run(
    plan: LogicalPlan,
    output_keys: Sequence[str],
    output_names: Sequence[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> ResultSet:
    """Execute ``plan`` and materialise the rows named by ``output_keys``.

    ``output_keys`` are the internal batch keys the projection wrote; ``output_names``
    are the headers the user sees. Keeping them separate means two select items can
    share a display name (SQL allows it) without one silently overwriting the other.
    """
    root = build_operator(plan, batch_size)
    stats = ExecutionStats()
    rows: list[tuple[Any, ...]] = []

    for batch in root.execute():
        stats.batches += 1
        try:
            columns = [batch.columns[key] for key in output_keys]
        except KeyError as error:  # pragma: no cover - indicates a planning bug
            raise ExecutionError(
                f"projection produced no column {error.args[0]!r}; "
                f"batch has: {', '.join(sorted(batch.columns))}"
            ) from None
        materialised = [column.to_pylist() for column in columns]
        rows.extend(zip(*materialised) if materialised else [])

    for operator in _iter_operators(root):
        if isinstance(operator, ScanOperator):
            stats.rows_scanned += operator.rows_scanned
            stats.rows_after_scan_filter += operator.rows_emitted
            stats.columns_read += len(operator.columns)
    stats.rows_returned = len(rows)
    return ResultSet(tuple(output_names), rows, stats)
