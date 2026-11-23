"""Name resolution: turn a parsed statement into one where every column is qualified.

This pass is why the verifier can promise "every column in this query exists". It walks
the whole statement, resolves each ``ColumnRef`` against the tables in scope, and either
rewrites it into a fully-qualified reference or raises. There is no fall-through case
where an unresolved name survives into the executor.

The pass also collects the per-column statistics the optimiser uses for estimation, so
the catalog is consulted exactly once per query rather than once per estimation call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ast_nodes import (
    Between,
    BinaryOp,
    ColumnRef,
    Expression,
    FromItem,
    FunctionCall,
    InList,
    IsNull,
    Join,
    Like,
    Literal,
    OrderByItem,
    SelectItem,
    SelectStatement,
    Star,
    TableRef,
    UnaryOp,
    contains_aggregate,
    transform,
)
from .catalog import Catalog, ColumnSchema, TableSchema
from .errors import (
    AmbiguousColumnError,
    BindError,
    InvalidAggregationError,
    UnknownColumnError,
)


def qualified(table: str, column: str) -> str:
    """The canonical key a bound column is known by everywhere downstream."""
    return f"{table}.{column}"


@dataclass(slots=True)
class Relation:
    """One table instance in the FROM clause, under the name the query uses for it."""

    alias: str
    schema: TableSchema
    #: True when the relation sits on the null-supplying side of a LEFT JOIN, so its
    #: columns become nullable even if the underlying file has no nulls.
    nullable: bool = False


@dataclass(slots=True)
class Scope:
    """The set of relations visible to a query, plus name-resolution helpers."""

    relations: list[Relation] = field(default_factory=list)

    def add(self, relation: Relation) -> None:
        if any(r.alias.lower() == relation.alias.lower() for r in self.relations):
            raise BindError(
                f"table name or alias {relation.alias!r} appears twice in FROM; "
                "give one of them a distinct alias"
            )
        self.relations.append(relation)

    def relation(self, alias: str) -> Relation:
        for rel in self.relations:
            if rel.alias.lower() == alias.lower():
                return rel
        known = ", ".join(r.alias for r in self.relations)
        raise BindError(
            f"unknown table or alias {alias!r} in this query; FROM provides: {known}"
        )

    def resolve(self, ref: ColumnRef) -> tuple[str, ColumnSchema]:
        """Resolve ``ref`` to (alias, column schema).

        Raises when the column does not exist anywhere, or exists on more than one
        relation without a qualifier. Both are common LLM failures and both need to be
        distinguishable in the error text so the repair prompt can be specific.
        """
        if ref.table is not None:
            relation = self.relation(ref.table)
            if not relation.schema.has_column(ref.name):
                raise UnknownColumnError(
                    f"column {ref.name!r} does not exist on table "
                    f"{relation.schema.name!r} (aliased {relation.alias!r}); "
                    f"its columns are: {', '.join(relation.schema.column_names)}"
                )
            return relation.alias, relation.schema.column(ref.name)

        matches = [r for r in self.relations if r.schema.has_column(ref.name)]
        if not matches:
            raise UnknownColumnError(
                f"column {ref.name!r} does not exist in any table in this query. "
                f"Available columns: {self.describe_columns()}"
            )
        if len(matches) > 1:
            owners = ", ".join(f"{r.alias}.{ref.name}" for r in matches)
            raise AmbiguousColumnError(
                f"column {ref.name!r} is ambiguous; it exists on several tables. "
                f"Qualify it as one of: {owners}"
            )
        relation = matches[0]
        return relation.alias, relation.schema.column(ref.name)

    def describe_columns(self) -> str:
        parts = []
        for rel in self.relations:
            parts.append(f"{rel.alias}({', '.join(rel.schema.column_names)})")
        return "; ".join(parts)

    def expand_star(self, star: Star) -> list[ColumnRef]:
        """Expand ``*`` / ``t.*`` into explicit qualified column references."""
        if star.table is not None:
            relation = self.relation(star.table)
            return [ColumnRef(c, relation.alias) for c in relation.schema.column_names]
        expanded: list[ColumnRef] = []
        for rel in self.relations:
            expanded.extend(ColumnRef(c, rel.alias) for c in rel.schema.column_names)
        return expanded


@dataclass(slots=True)
class BoundStatement:
    """A statement in which every column reference is qualified and known to exist."""

    statement: SelectStatement
    scope: Scope
    #: qualified name -> schema, for the optimiser's estimation and the executor's dtypes
    column_stats: dict[str, ColumnSchema]
    #: The select list with stars expanded, so the output shape is fixed at bind time.
    output_names: tuple[str, ...]


class Binder:
    """Binds one :class:`SelectStatement` against a :class:`Catalog`."""

    def __init__(self, catalog: Catalog) -> None:
        self._catalog = catalog

    def bind(self, statement: SelectStatement) -> BoundStatement:
        scope = Scope()
        from_item = statement.from_item
        if from_item is not None:
            self._collect_relations(from_item, scope, nullable=False)
        elif any(not isinstance(item.expr, Literal) for item in statement.select_list):
            raise BindError("a query with column references requires a FROM clause")

        select_list = self._bind_select_list(statement.select_list, scope)

        # ORDER BY and HAVING may name a select-list alias or (for ORDER BY) an output
        # ordinal. Both are resolved by substitution *before* scope resolution, because
        # an alias is not a column of any table and would otherwise fail to bind.
        aliases = {
            item.alias: item.expr for item in select_list if item.alias is not None
        }
        where = self._bind_optional(statement.where, scope)
        group_by = tuple(
            self._bind_expression(self._substitute_alias(g, aliases), scope)
            for g in statement.group_by
        )
        having = self._bind_optional(
            self._substitute_alias(statement.having, aliases) if statement.having else None,
            scope,
        )
        order_by = tuple(
            OrderByItem(
                self._bind_expression(
                    self._substitute_alias(
                        self._substitute_ordinal(o.expr, select_list), aliases
                    ),
                    scope,
                ),
                o.direction,
            )
            for o in statement.order_by
        )
        from_bound = self._bind_from(from_item, scope) if from_item is not None else None

        bound_statement = SelectStatement(
            select_list=select_list,
            from_item=from_bound,
            where=where,
            group_by=group_by,
            having=having,
            order_by=order_by,
            limit=statement.limit,
            offset=statement.offset,
            distinct=statement.distinct,
        )
        self._validate_aggregation(bound_statement)

        column_stats: dict[str, ColumnSchema] = {}
        for rel in scope.relations:
            for column in rel.schema.columns:
                column_stats[qualified(rel.alias, column.name)] = column

        return BoundStatement(
            statement=bound_statement,
            scope=scope,
            column_stats=column_stats,
            output_names=tuple(item.output_name for item in select_list),
        )

    # -- FROM --------------------------------------------------------------------

    def _collect_relations(self, item: FromItem, scope: Scope, nullable: bool) -> None:
        if isinstance(item, TableRef):
            schema = self._catalog.table(item.name)
            scope.add(Relation(alias=item.bound_name, schema=schema, nullable=nullable))
        elif isinstance(item, Join):
            self._collect_relations(item.left, scope, nullable)
            # The right side of a LEFT JOIN can produce NULLs for rows with no match.
            self._collect_relations(
                item.right, scope, nullable or item.join_type.value == "LEFT"
            )
        else:  # pragma: no cover - the grammar produces nothing else
            raise BindError(f"unsupported FROM item {type(item).__name__}")

    def _bind_from(self, item: FromItem, scope: Scope) -> FromItem:
        if isinstance(item, TableRef):
            # Normalise so the planner never has to ask "alias or name?".
            schema = self._catalog.table(item.name)
            return TableRef(name=schema.name, alias=item.bound_name)
        assert isinstance(item, Join)
        return Join(
            left=self._bind_from(item.left, scope),
            right=self._bind_from(item.right, scope),
            join_type=item.join_type,
            condition=self._bind_expression(item.condition, scope),
        )

    # -- expressions -------------------------------------------------------------

    def _bind_select_list(
        self, items: tuple[SelectItem, ...], scope: Scope
    ) -> tuple[SelectItem, ...]:
        bound: list[SelectItem] = []
        for item in items:
            if isinstance(item.expr, Star):
                if item.alias is not None:
                    raise BindError("* cannot be given an alias")
                for ref in scope.expand_star(item.expr):
                    bound.append(SelectItem(ref, alias=None))
                continue
            bound.append(SelectItem(self._bind_expression(item.expr, scope), item.alias))
        if not bound:
            raise BindError("the select list is empty")
        return tuple(bound)

    def _bind_optional(self, expr: Expression | None, scope: Scope) -> Expression | None:
        return None if expr is None else self._bind_expression(expr, scope)

    @staticmethod
    def _substitute_alias(
        expr: Expression | None, aliases: dict[str, Expression]
    ) -> Expression | None:
        """Replace every bare reference to a select-list alias with what it names.

        Must recurse. ``HAVING n > 1`` hides the alias inside a comparison, so a
        root-only substitution silently leaves it unresolved and the bind then fails
        with "column 'n' does not exist" -- an error about a name the user never wrote.
        See FAILURES.md #7.

        Substituted expressions are already bound (they come from the bound select
        list), and binding is idempotent for a qualified reference, so re-binding the
        result is safe.
        """
        if expr is None:
            return None

        def rewrite(node: Expression) -> Expression:
            if isinstance(node, ColumnRef) and node.table is None and node.name in aliases:
                return aliases[node.name]
            return node

        return transform(expr, rewrite)

    @staticmethod
    def _substitute_ordinal(expr: Expression, select_list: tuple[SelectItem, ...]) -> Expression:
        """Resolve ``ORDER BY 2`` to the second select-list expression, as SQLite does."""
        if isinstance(expr, Literal) and isinstance(expr.value, int) and not isinstance(
            expr.value, bool
        ):
            index = expr.value
            if not 1 <= index <= len(select_list):
                raise BindError(
                    f"ORDER BY {index} is out of range; the select list has "
                    f"{len(select_list)} column(s)"
                )
            return select_list[index - 1].expr
        return expr

    def _bind_expression(self, expr: Expression, scope: Scope) -> Expression:
        """Rewrite ``expr`` with every ColumnRef qualified. Structure is preserved."""
        if isinstance(expr, ColumnRef):
            alias, _schema = scope.resolve(expr)
            return ColumnRef(name=expr.name, table=alias)
        if isinstance(expr, Literal):
            return expr
        if isinstance(expr, Star):
            raise BindError("* is only valid in a select list or as COUNT(*)")
        if isinstance(expr, BinaryOp):
            return BinaryOp(
                expr.op,
                self._bind_expression(expr.left, scope),
                self._bind_expression(expr.right, scope),
            )
        if isinstance(expr, UnaryOp):
            return UnaryOp(expr.op, self._bind_expression(expr.operand, scope))
        if isinstance(expr, InList):
            return InList(
                self._bind_expression(expr.expr, scope),
                tuple(self._bind_expression(v, scope) for v in expr.values),
                expr.negated,
            )
        if isinstance(expr, Between):
            return Between(
                self._bind_expression(expr.expr, scope),
                self._bind_expression(expr.low, scope),
                self._bind_expression(expr.high, scope),
                expr.negated,
            )
        if isinstance(expr, Like):
            return Like(
                self._bind_expression(expr.expr, scope),
                self._bind_expression(expr.pattern, scope),
                expr.negated,
            )
        if isinstance(expr, IsNull):
            return IsNull(self._bind_expression(expr.expr, scope), expr.negated)
        if isinstance(expr, FunctionCall):
            if len(expr.args) == 1 and isinstance(expr.args[0], Star):
                return expr  # COUNT(*) has nothing to bind
            return FunctionCall(
                expr.name,
                tuple(self._bind_expression(a, scope) for a in expr.args),
                distinct=expr.distinct,
            )
        raise BindError(f"cannot bind expression node {type(expr).__name__}")

    # -- aggregation rules -------------------------------------------------------

    def _validate_aggregation(self, statement: SelectStatement) -> None:
        """Reject select lists that mix bare columns with aggregates illegally.

        SQLite permits ``SELECT name, COUNT(*) FROM t`` and picks an arbitrary row for
        ``name``. sql-engine refuses it. Accepting it would make the differential tests
        non-deterministic on exactly the queries an LLM most often gets wrong, and a
        clear error is more useful to the repair loop than an arbitrary answer.
        """
        has_aggregate = any(contains_aggregate(item.expr) for item in statement.select_list)
        if statement.having is not None:
            has_aggregate = True
        if not has_aggregate and not statement.group_by:
            return

        group_keys = {str(g) for g in statement.group_by}
        for item in statement.select_list:
            if contains_aggregate(item.expr):
                continue
            if str(item.expr) in group_keys:
                continue
            if isinstance(item.expr, Literal):
                continue
            raise InvalidAggregationError(
                f"{item.expr} appears in the select list but is neither aggregated nor "
                "listed in GROUP BY"
            )

        for order_item in statement.order_by:
            if contains_aggregate(order_item.expr):
                continue
            if str(order_item.expr) in group_keys:
                continue
            if isinstance(order_item.expr, Literal):
                continue
            # ORDER BY may also name an output alias of the select list.
            aliases = {item.alias for item in statement.select_list if item.alias}
            if isinstance(order_item.expr, ColumnRef) and order_item.expr.name in aliases:
                continue
            raise InvalidAggregationError(
                f"{order_item.expr} appears in ORDER BY but is neither aggregated nor "
                "listed in GROUP BY"
            )
