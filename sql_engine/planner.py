"""Bound statement -> logical plan.

The plan shape is fixed: SQL's evaluation order is a total order, and sql-engine has no
subqueries or set operations, so there is nothing to search over. That is a deliberate
scope decision, not an oversight -- a join-order enumerator without histograms would
just be a slower way of guessing.

Plan shape, bottom to top:

    Scan(s) -> HashJoin(s) -> Filter(WHERE) -> Aggregate -> Filter(HAVING)
            -> Project -> Distinct -> Sort -> Limit
"""

from __future__ import annotations

from .ast_nodes import (
    BinaryOp,
    BinaryOperator,
    ColumnRef,
    Expression,
    FromItem,
    FunctionCall,
    Join,
    SelectItem,
    SelectStatement,
    Star,
    TableRef,
    column_refs,
    contains_aggregate,
    walk,
)
from .binder import BoundStatement
from .catalog import Catalog
from .errors import BindError
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
    iter_plan,
)


def split_conjuncts(expr: Expression | None) -> tuple[Expression, ...]:
    """Flatten ``a AND b AND c`` into ``(a, b, c)``.

    Predicate pushdown operates on individual conjuncts; keeping them as a nested
    BinaryOp tree would force every rule to re-walk it, and would prevent pushing one
    half of an AND when the other half spans two tables.
    """
    if expr is None:
        return ()
    if isinstance(expr, BinaryOp) and expr.op is BinaryOperator.AND:
        return split_conjuncts(expr.left) + split_conjuncts(expr.right)
    return (expr,)


def plan_aliases(plan: LogicalPlan) -> frozenset[str]:
    """The set of table aliases a subtree can produce columns for."""
    return frozenset(node.alias.lower() for node in iter_plan(plan) if isinstance(node, Scan))


def expression_aliases(expr: Expression) -> frozenset[str]:
    """Table aliases referenced by ``expr``. Empty for a constant expression."""
    return frozenset(
        ref.table.lower() for ref in column_refs(expr) if ref.table is not None
    )


class Planner:
    """Builds a logical plan from a :class:`~sql_engine.binder.BoundStatement`."""

    def __init__(self, catalog: Catalog) -> None:
        self._catalog = catalog

    def plan(self, bound: BoundStatement) -> LogicalPlan:
        statement = bound.statement

        if statement.from_item is None:
            raise BindError("sql-engine requires a FROM clause")

        node: LogicalPlan = self._plan_from(statement.from_item)

        where_conjuncts = split_conjuncts(statement.where)
        if where_conjuncts:
            node = Filter(node, where_conjuncts)

        aggregates = self._collect_aggregates(statement)
        if aggregates or statement.group_by:
            node = Aggregate(node, tuple(statement.group_by), aggregates)

        having_conjuncts = split_conjuncts(statement.having)
        if having_conjuncts:
            node = Filter(node, having_conjuncts)

        node = Project(node, statement.select_list)

        if statement.distinct:
            node = Distinct(node, tuple(item.expr for item in statement.select_list))

        if statement.order_by:
            node = Sort(node, statement.order_by)

        if statement.limit is not None or statement.offset:
            node = Limit(node, statement.limit, statement.offset or 0)

        return node

    # -- FROM --------------------------------------------------------------------

    def _plan_from(self, item: FromItem) -> LogicalPlan:
        if isinstance(item, TableRef):
            return Scan(table=self._catalog.table(item.name), alias=item.bound_name)
        assert isinstance(item, Join)

        left = self._plan_from(item.left)
        right = self._plan_from(item.right)
        equi_keys, residual = self._split_join_condition(item.condition, left, right)
        return HashJoin(
            left=left,
            right=right,
            join_type=item.join_type,
            equi_keys=tuple(equi_keys),
            residual=tuple(residual),
        )

    @staticmethod
    def _split_join_condition(
        condition: Expression, left: LogicalPlan, right: LogicalPlan
    ) -> tuple[list[tuple[Expression, Expression]], list[Expression]]:
        """Separate ON into hash keys and a residual predicate.

        A conjunct qualifies as a hash key only when it is ``=`` and each side draws its
        columns from exactly one input. ``ON a.x = b.y AND a.z > 3`` therefore gives one
        hash key and one residual, which is the right split: the inequality cannot be
        hashed but can be checked cheaply on matched rows.
        """
        left_aliases = plan_aliases(left)
        right_aliases = plan_aliases(right)
        equi_keys: list[tuple[Expression, Expression]] = []
        residual: list[Expression] = []

        for conjunct in split_conjuncts(condition):
            if isinstance(conjunct, BinaryOp) and conjunct.op is BinaryOperator.EQ:
                lhs = expression_aliases(conjunct.left)
                rhs = expression_aliases(conjunct.right)
                if lhs <= left_aliases and rhs <= right_aliases and lhs and rhs:
                    equi_keys.append((conjunct.left, conjunct.right))
                    continue
                if lhs <= right_aliases and rhs <= left_aliases and lhs and rhs:
                    equi_keys.append((conjunct.right, conjunct.left))
                    continue
            residual.append(conjunct)
        return equi_keys, residual

    # -- aggregation -------------------------------------------------------------

    @staticmethod
    def _collect_aggregates(statement: SelectStatement) -> tuple[FunctionCall, ...]:
        """Every distinct aggregate call anywhere in the statement, in first-seen order.

        De-duplicated by rendered text so ``SELECT SUM(x), SUM(x)/2`` computes SUM(x)
        once. Order is stable so EXPLAIN output is deterministic between runs.
        """
        found: dict[str, FunctionCall] = {}

        def scan(expr: Expression | None) -> None:
            if expr is None:
                return
            for node in walk(expr):
                if isinstance(node, FunctionCall) and node.is_aggregate:
                    found.setdefault(str(node), node)

        for item in statement.select_list:
            scan(item.expr)
        scan(statement.having)
        for order_item in statement.order_by:
            scan(order_item.expr)
        return tuple(found.values())


def project_output_keys(items: tuple[SelectItem, ...]) -> tuple[str, ...]:
    """The batch key each select item is materialised under.

    Two items that render identically share a key (so the work is done once); two items
    with the same alias but different expressions are disambiguated with a suffix so the
    second does not silently overwrite the first.
    """
    keys: list[str] = []
    claimed: dict[str, str] = {}  # key -> rendered expression that owns it
    for item in items:
        base = item.alias if item.alias else str(item.expr)
        rendered = str(item.expr)
        key = base
        counter = 1
        while key in claimed and claimed[key] != rendered:
            key = f"{base}#{counter}"
            counter += 1
        claimed[key] = rendered
        keys.append(key)
    return tuple(keys)


def build_plan(bound: BoundStatement, catalog: Catalog) -> LogicalPlan:
    """Convenience wrapper around :class:`Planner`."""
    return Planner(catalog).plan(bound)


__all__ = [
    "Planner",
    "build_plan",
    "split_conjuncts",
    "plan_aliases",
    "expression_aliases",
    "project_output_keys",
]
