"""Rule-based logical optimiser: predicate pushdown and projection pruning.

Two rules, both classical, both chosen because their effect is *visible*: after
optimisation the EXPLAIN output shows fewer columns and a predicate attached to a scan,
and the estimated row counts drop. A rule whose benefit you cannot see in the plan is a
rule you cannot defend in an interview.

Correctness notes that matter more than the rules themselves:

* A predicate is never pushed through ``Limit``. ``LIMIT 10`` then filter is not the
  same query as filter then ``LIMIT 10``.
* For a LEFT JOIN, a WHERE predicate on the *preserved* (left) side may be pushed, but
  one on the null-supplying (right) side may not: it would discard rows that the outer
  join is required to keep with NULLs. Conversely an ON predicate on the right side
  *may* be pushed, because ON filters what counts as a match rather than what survives.
  This asymmetry is the classic outer-join pushdown trap and is tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .ast_nodes import (
    ColumnRef,
    Expression,
    FunctionCall,
    JoinType,
    contains_aggregate,
)
from .binder import qualified
from .plan import (
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
from .planner import expression_aliases, plan_aliases


@dataclass(frozen=True, slots=True)
class OptimiserReport:
    """What the optimiser actually did, for EXPLAIN and for the benchmark harness."""

    predicates_pushed_into_scans: int
    columns_pruned: int
    columns_read_before: int
    columns_read_after: int

    def summary(self) -> str:
        return (
            f"predicate pushdown: {self.predicates_pushed_into_scans} conjunct(s) "
            f"pushed into scans; projection pruning: "
            f"{self.columns_read_after}/{self.columns_read_before} columns read "
            f"({self.columns_pruned} pruned)"
        )


class Optimiser:
    """Applies the rewrite rules. ``enabled=False`` makes it a no-op.

    The switch exists so the benchmark can measure the same query with the same code
    path on both sides; a separate "unoptimised engine" would not be a fair comparison.
    """

    def __init__(self, *, predicate_pushdown: bool = True, projection_pruning: bool = True) -> None:
        self.predicate_pushdown = predicate_pushdown
        self.projection_pruning = projection_pruning

    def optimise(self, plan: LogicalPlan) -> tuple[LogicalPlan, OptimiserReport]:
        columns_before = _columns_read(plan)
        result = plan
        if self.predicate_pushdown:
            result = _push_predicates(result, ())
        if self.projection_pruning:
            result = _prune_columns(result, frozenset())
        columns_after = _columns_read(result)
        pushed = sum(len(scan.pushed_predicates) for scan in _iter_scans(result))
        return result, OptimiserReport(
            predicates_pushed_into_scans=pushed,
            columns_pruned=max(0, columns_before - columns_after),
            columns_read_before=columns_before,
            columns_read_after=columns_after,
        )


# --------------------------------------------------------------------------------------
# Rule 1: predicate pushdown
# --------------------------------------------------------------------------------------


def _push_predicates(node: LogicalPlan, pending: tuple[Expression, ...]) -> LogicalPlan:
    """Rewrite ``node`` so that ``pending`` (predicates from above) is applied as deep
    as it can legally go."""

    if isinstance(node, Scan):
        pushable, blocked = _partition(pending, lambda p: _refs_only(p, {node.alias.lower()}))
        new_scan = replace(node, pushed_predicates=node.pushed_predicates + pushable)
        return _wrap_filter(new_scan, blocked)

    if isinstance(node, Filter):
        # The Filter node dissolves; its conjuncts join the pending set and are
        # re-materialised wherever they come to rest.
        return _push_predicates(node.child, pending + node.predicates)

    if isinstance(node, HashJoin):
        return _push_through_join(node, pending)

    if isinstance(node, Aggregate):
        group_texts = {str(key) for key in node.group_keys}
        pushable, blocked = _partition(
            pending, lambda p: _depends_only_on_group_keys(p, group_texts)
        )
        child = _push_predicates(node.child, pushable)
        return _wrap_filter(replace(node, child=child), blocked)

    if isinstance(node, Limit):
        # Never push through LIMIT: it would change which rows the limit selects.
        child = _push_predicates(node.child, ())
        return _wrap_filter(replace(node, child=child), pending)

    if isinstance(node, (Project, Sort, Distinct)):
        child = _push_predicates(node.child, pending)
        return replace(node, child=child)

    return node  # pragma: no cover - every node type is handled above


def _push_through_join(node: HashJoin, pending: tuple[Expression, ...]) -> LogicalPlan:
    left_aliases = plan_aliases(node.left)
    right_aliases = plan_aliases(node.right)
    is_inner = node.join_type is JoinType.INNER

    to_left: list[Expression] = []
    to_right: list[Expression] = []
    stay_above: list[Expression] = []
    stay_residual: list[Expression] = []

    # Predicates arriving from above behave like WHERE: they filter the join *output*.
    for predicate in pending:
        aliases = expression_aliases(predicate)
        if aliases and aliases <= left_aliases:
            to_left.append(predicate)
        elif aliases and aliases <= right_aliases and is_inner:
            to_right.append(predicate)
        else:
            stay_above.append(predicate)

    # ON-clause residuals filter what counts as a match, which is why the legal
    # direction is the mirror image of the WHERE case for a LEFT join.
    for predicate in node.residual:
        aliases = expression_aliases(predicate)
        if aliases and aliases <= right_aliases:
            to_right.append(predicate)
        elif aliases and aliases <= left_aliases and is_inner:
            to_left.append(predicate)
        else:
            stay_residual.append(predicate)

    new_join = replace(
        node,
        left=_push_predicates(node.left, tuple(to_left)),
        right=_push_predicates(node.right, tuple(to_right)),
        residual=tuple(stay_residual),
    )
    return _wrap_filter(new_join, tuple(stay_above))


def _partition(items: tuple[Expression, ...], predicate) -> tuple[
    tuple[Expression, ...], tuple[Expression, ...]
]:
    yes = tuple(item for item in items if predicate(item))
    no = tuple(item for item in items if not predicate(item))
    return yes, no


def _refs_only(expr: Expression, aliases: set[str]) -> bool:
    """True when ``expr`` references at most the given aliases and at least one of them.

    The "at least one" clause keeps constant predicates (``WHERE 1 = 0``) out of scans;
    a constant belongs wherever it was written, and pushing it buys nothing.
    """
    referenced = expression_aliases(expr)
    return bool(referenced) and referenced <= aliases and not contains_aggregate(expr)


def _depends_only_on_group_keys(expr: Expression, group_texts: set[str]) -> bool:
    """True when a HAVING conjunct can be evaluated before grouping.

    ``HAVING region = 'EU'`` with ``GROUP BY region`` is really a WHERE clause, and
    filtering before the hash aggregate is strictly cheaper.
    """
    if contains_aggregate(expr):
        return False
    refs = [node for node in _walk(expr) if isinstance(node, ColumnRef)]
    if not refs:
        return False
    return all(str(ref) in group_texts for ref in refs)


def _wrap_filter(node: LogicalPlan, predicates: tuple[Expression, ...]) -> LogicalPlan:
    return Filter(node, predicates) if predicates else node


# --------------------------------------------------------------------------------------
# Rule 2: projection pruning
# --------------------------------------------------------------------------------------


def _prune_columns(node: LogicalPlan, needed: frozenset[str]) -> LogicalPlan:
    """Annotate every scan with the minimal set of columns its subtree requires.

    ``needed`` is the set of qualified base-column names the parent will read out of
    this node's output.
    """
    if isinstance(node, Scan):
        wanted = {
            name
            for name in node.table.column_names
            if qualified(node.alias, name) in needed
        }
        for predicate in node.pushed_predicates:
            wanted.update(_local_column_names(predicate, node.alias))
        if not wanted:
            # `SELECT COUNT(*) FROM t` needs no column values, only a row count. The
            # reader still has to read something to know how many rows there are, so
            # keep exactly one -- the narrowest available.
            wanted = {node.table.column_names[0]}
        ordered = tuple(name for name in node.table.column_names if name in wanted)
        return replace(node, projection=ordered)

    if isinstance(node, Filter):
        below = needed | _refs(node.predicates)
        return replace(node, child=_prune_columns(node.child, below))

    if isinstance(node, Project):
        below = needed | _refs(item.expr for item in node.items)
        return replace(node, child=_prune_columns(node.child, below))

    if isinstance(node, Distinct):
        below = needed | _refs(node.keys)
        return replace(node, child=_prune_columns(node.child, below))

    if isinstance(node, Sort):
        below = needed | _refs(key.expr for key in node.keys)
        return replace(node, child=_prune_columns(node.child, below))

    if isinstance(node, Limit):
        return replace(node, child=_prune_columns(node.child, needed))

    if isinstance(node, Aggregate):
        # An aggregate is a barrier: nothing above it can reach a base column that is
        # not a grouping key, so the requirement set is replaced rather than extended.
        below = _refs(node.group_keys) | _refs(node.aggregates)
        return replace(node, child=_prune_columns(node.child, below))

    if isinstance(node, HashJoin):
        below = needed | _refs(node.residual)
        for left_key, right_key in node.equi_keys:
            below = below | _refs((left_key, right_key))
        return replace(
            node,
            left=_prune_columns(node.left, below),
            right=_prune_columns(node.right, below),
        )

    return node  # pragma: no cover


def _refs(expressions) -> frozenset[str]:
    """Qualified names of every column referenced by ``expressions``."""
    names: set[str] = set()
    for expr in expressions:
        for node in _walk(expr):
            if isinstance(node, ColumnRef) and node.table is not None:
                names.add(qualified(node.table, node.name))
    return frozenset(names)


def _local_column_names(expr: Expression, alias: str) -> set[str]:
    return {
        node.name
        for node in _walk(expr)
        if isinstance(node, ColumnRef) and node.table is not None
        and node.table.lower() == alias.lower()
    }


def _walk(expr: Expression):
    from .ast_nodes import walk

    if isinstance(expr, FunctionCall):
        yield expr
        for arg in expr.args:
            yield from walk(arg)
        return
    yield from walk(expr)


def _iter_scans(plan: LogicalPlan):
    from .plan import iter_plan

    return (node for node in iter_plan(plan) if isinstance(node, Scan))


def _columns_read(plan: LogicalPlan) -> int:
    """Total number of columns every scan in ``plan`` will read."""
    total = 0
    for scan in _iter_scans(plan):
        total += (
            len(scan.projection)
            if scan.projection is not None
            else len(scan.table.columns)
        )
    return total
