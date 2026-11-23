"""Abstract syntax tree for the sql-engine SQL dialect.

The security argument of this project lives in this module. There is no `Insert`,
`Update`, `Delete`, `CreateTable` or `Drop` node -- so no sequence of tokens the parser
accepts can produce one. A destructive statement is not rejected by a rule that could be
edited or bypassed; it is *unrepresentable* in the type system of the tree.

Nodes are frozen dataclasses. The optimiser rewrites plans by constructing new nodes
rather than mutating shared ones, which means a plan can be printed before and after
optimisation without the "before" having been silently changed underneath.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Union


class Node:
    """Marker base class for every AST node."""

    __slots__ = ()


# --------------------------------------------------------------------------------------
# Expressions
# --------------------------------------------------------------------------------------


class Expression(Node):
    """Base class for anything that evaluates to a column of values."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class Literal(Expression):
    """A constant. ``value`` is already a Python object (int/float/str/bool/None)."""

    value: int | float | str | bool | None

    def __str__(self) -> str:
        if self.value is None:
            return "NULL"
        if isinstance(self.value, bool):
            return "TRUE" if self.value else "FALSE"
        if isinstance(self.value, str):
            escaped = self.value.replace(chr(39), chr(39) * 2)
            return f"{chr(39)}{escaped}{chr(39)}"
        return str(self.value)


@dataclass(frozen=True, slots=True)
class ColumnRef(Expression):
    """A column reference, optionally qualified by a table name or alias."""

    name: str
    table: str | None = None

    def __str__(self) -> str:
        return f"{self.table}.{self.name}" if self.table else self.name


@dataclass(frozen=True, slots=True)
class Star(Expression):
    """``*`` or ``t.*`` in a select list. Only legal directly in a select list."""

    table: str | None = None

    def __str__(self) -> str:
        return f"{self.table}.*" if self.table else "*"


class BinaryOperator(str, Enum):
    """Every infix operator the grammar accepts."""

    EQ = "="
    NEQ = "<>"
    LT = "<"
    LTE = "<="
    GT = ">"
    GTE = ">="
    AND = "AND"
    OR = "OR"
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"

    @property
    def is_comparison(self) -> bool:
        return self in _COMPARISONS

    @property
    def is_logical(self) -> bool:
        return self in (BinaryOperator.AND, BinaryOperator.OR)


_COMPARISONS = frozenset(
    {
        BinaryOperator.EQ,
        BinaryOperator.NEQ,
        BinaryOperator.LT,
        BinaryOperator.LTE,
        BinaryOperator.GT,
        BinaryOperator.GTE,
    }
)


@dataclass(frozen=True, slots=True)
class BinaryOp(Expression):
    """``left <op> right``."""

    op: BinaryOperator
    left: Expression
    right: Expression

    def __str__(self) -> str:
        return f"({self.left} {self.op.value} {self.right})"


@dataclass(frozen=True, slots=True)
class UnaryOp(Expression):
    """``NOT x`` or unary ``-x``."""

    op: str  # "NOT" or "-"
    operand: Expression

    def __str__(self) -> str:
        if self.op == "NOT":
            return f"(NOT {self.operand})"
        return f"(-{self.operand})"


@dataclass(frozen=True, slots=True)
class InList(Expression):
    """``expr IN (a, b, c)``. The right-hand side is a literal list only.

    A subquery on the right would require a correlated executor; the grammar simply has
    no production for it, which is why IN is safe to evaluate as a set membership test.
    """

    expr: Expression
    values: tuple[Expression, ...]
    negated: bool = False

    def __str__(self) -> str:
        items = ", ".join(str(v) for v in self.values)
        return f"({self.expr}{' NOT' if self.negated else ''} IN ({items}))"


@dataclass(frozen=True, slots=True)
class Between(Expression):
    """``expr BETWEEN low AND high`` (inclusive on both ends, per the SQL standard)."""

    expr: Expression
    low: Expression
    high: Expression
    negated: bool = False

    def __str__(self) -> str:
        return (
            f"({self.expr}{' NOT' if self.negated else ''} "
            f"BETWEEN {self.low} AND {self.high})"
        )


@dataclass(frozen=True, slots=True)
class Like(Expression):
    """``expr LIKE pattern`` with SQL wildcards ``%`` and ``_``."""

    expr: Expression
    pattern: Expression
    negated: bool = False

    def __str__(self) -> str:
        return f"({self.expr}{' NOT' if self.negated else ''} LIKE {self.pattern})"


@dataclass(frozen=True, slots=True)
class IsNull(Expression):
    """``expr IS NULL`` / ``expr IS NOT NULL``."""

    expr: Expression
    negated: bool = False

    def __str__(self) -> str:
        return f"({self.expr} IS{' NOT' if self.negated else ''} NULL)"


#: Aggregate functions the executor implements. Anything else is a parse-time error, so
#: an LLM cannot smuggle in a function name that reaches Python evaluation.
AGGREGATE_FUNCTIONS: frozenset[str] = frozenset({"COUNT", "SUM", "AVG", "MIN", "MAX"})


@dataclass(frozen=True, slots=True)
class FunctionCall(Expression):
    """An aggregate call. ``COUNT(*)`` is encoded as ``args=(Star(),)``."""

    name: str  # upper-cased at parse time
    args: tuple[Expression, ...]
    distinct: bool = False

    @property
    def is_aggregate(self) -> bool:
        return self.name in AGGREGATE_FUNCTIONS

    def __str__(self) -> str:
        inner = ", ".join(str(a) for a in self.args)
        prefix = "DISTINCT " if self.distinct else ""
        return f"{self.name}({prefix}{inner})"


# --------------------------------------------------------------------------------------
# FROM clause
# --------------------------------------------------------------------------------------


class FromItem(Node):
    """Base class for things that can appear in a FROM clause."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class TableRef(FromItem):
    """A base table with an optional alias."""

    name: str
    alias: str | None = None

    @property
    def bound_name(self) -> str:
        """The name columns are qualified by inside the query."""
        return self.alias or self.name

    def __str__(self) -> str:
        return f"{self.name} AS {self.alias}" if self.alias else self.name


class JoinType(str, Enum):
    """Only two join types exist. There is no CROSS JOIN production in the grammar."""

    INNER = "INNER"
    LEFT = "LEFT"


@dataclass(frozen=True, slots=True)
class Join(FromItem):
    """``left <type> JOIN right ON condition``.

    ``ON`` is mandatory in the grammar. Combined with the verifier's equi-predicate
    check this is what makes an accidental cartesian product hard to express and easy
    to detect.
    """

    left: FromItem
    right: FromItem
    join_type: JoinType
    condition: Expression

    def __str__(self) -> str:
        return f"{self.left} {self.join_type.value} JOIN {self.right} ON {self.condition}"


# --------------------------------------------------------------------------------------
# Statement
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SelectItem(Node):
    """One entry in the select list."""

    expr: Expression
    alias: str | None = None

    @property
    def output_name(self) -> str:
        """Column name this item produces, following SQLite's naming rules closely enough
        for differential testing to compare by position rather than by name."""
        if self.alias:
            return self.alias
        if isinstance(self.expr, ColumnRef):
            return self.expr.name
        return str(self.expr)

    def __str__(self) -> str:
        return f"{self.expr} AS {self.alias}" if self.alias else str(self.expr)


class SortDirection(str, Enum):
    ASC = "ASC"
    DESC = "DESC"


@dataclass(frozen=True, slots=True)
class OrderByItem(Node):
    """One ORDER BY key."""

    expr: Expression
    direction: SortDirection = SortDirection.ASC

    def __str__(self) -> str:
        return f"{self.expr} {self.direction.value}"


@dataclass(frozen=True, slots=True)
class SelectStatement(Node):
    """The only statement type in the language.

    ``parse()`` returns this or raises. There is deliberately no ``Statement`` union to
    dispatch on: a caller that gets a value back is holding a SELECT.
    """

    select_list: tuple[SelectItem, ...]
    from_item: FromItem | None = None
    where: Expression | None = None
    group_by: tuple[Expression, ...] = field(default_factory=tuple)
    having: Expression | None = None
    order_by: tuple[OrderByItem, ...] = field(default_factory=tuple)
    limit: int | None = None
    offset: int | None = None
    distinct: bool = False

    def __str__(self) -> str:
        parts = ["SELECT"]
        if self.distinct:
            parts.append("DISTINCT")
        parts.append(", ".join(str(item) for item in self.select_list))
        if self.from_item is not None:
            parts.append(f"FROM {self.from_item}")
        if self.where is not None:
            parts.append(f"WHERE {self.where}")
        if self.group_by:
            parts.append("GROUP BY " + ", ".join(str(g) for g in self.group_by))
        if self.having is not None:
            parts.append(f"HAVING {self.having}")
        if self.order_by:
            parts.append("ORDER BY " + ", ".join(str(o) for o in self.order_by))
        if self.limit is not None:
            parts.append(f"LIMIT {self.limit}")
        if self.offset is not None:
            parts.append(f"OFFSET {self.offset}")
        return " ".join(parts)


AnyExpression = Union[
    Literal, ColumnRef, Star, BinaryOp, UnaryOp, InList, Between, Like, IsNull, FunctionCall
]


def walk(expr: Expression):
    """Yield ``expr`` and every sub-expression, depth first.

    Used by the binder, the optimiser's column-requirement analysis and the verifier.
    Centralising the traversal means adding an expression node only requires editing
    this one function, not three separate visitors that could drift apart.
    """
    yield expr
    if isinstance(expr, BinaryOp):
        yield from walk(expr.left)
        yield from walk(expr.right)
    elif isinstance(expr, UnaryOp):
        yield from walk(expr.operand)
    elif isinstance(expr, InList):
        yield from walk(expr.expr)
        for value in expr.values:
            yield from walk(value)
    elif isinstance(expr, Between):
        yield from walk(expr.expr)
        yield from walk(expr.low)
        yield from walk(expr.high)
    elif isinstance(expr, Like):
        yield from walk(expr.expr)
        yield from walk(expr.pattern)
    elif isinstance(expr, IsNull):
        yield from walk(expr.expr)
    elif isinstance(expr, FunctionCall):
        for arg in expr.args:
            yield from walk(arg)


def transform(expr: Expression, rewrite) -> Expression:
    """Rebuild ``expr`` bottom-up, applying ``rewrite`` to every node.

    Companion to :func:`walk`: ``walk`` reads the tree, this one rewrites it. Both live
    here for the same reason -- a new expression node needs editing in exactly two
    places, not in every visitor scattered across the package. The bug that motivated
    this (see FAILURES.md #7) was a rewrite that only looked at the root node.
    """
    if isinstance(expr, BinaryOp):
        expr = BinaryOp(expr.op, transform(expr.left, rewrite), transform(expr.right, rewrite))
    elif isinstance(expr, UnaryOp):
        expr = UnaryOp(expr.op, transform(expr.operand, rewrite))
    elif isinstance(expr, InList):
        expr = InList(
            transform(expr.expr, rewrite),
            tuple(transform(value, rewrite) for value in expr.values),
            expr.negated,
        )
    elif isinstance(expr, Between):
        expr = Between(
            transform(expr.expr, rewrite),
            transform(expr.low, rewrite),
            transform(expr.high, rewrite),
            expr.negated,
        )
    elif isinstance(expr, Like):
        expr = Like(
            transform(expr.expr, rewrite), transform(expr.pattern, rewrite), expr.negated
        )
    elif isinstance(expr, IsNull):
        expr = IsNull(transform(expr.expr, rewrite), expr.negated)
    elif isinstance(expr, FunctionCall):
        expr = FunctionCall(
            expr.name,
            tuple(transform(arg, rewrite) for arg in expr.args),
            distinct=expr.distinct,
        )
    return rewrite(expr)


def column_refs(expr: Expression) -> set[ColumnRef]:
    """Every :class:`ColumnRef` reachable from ``expr``."""
    return {node for node in walk(expr) if isinstance(node, ColumnRef)}


def contains_aggregate(expr: Expression) -> bool:
    """True if ``expr`` contains an aggregate call anywhere inside it."""
    return any(isinstance(n, FunctionCall) and n.is_aggregate for n in walk(expr))
