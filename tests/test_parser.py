"""Parser unit tests.

Half of these assert what the grammar *cannot* express. That half is the load-bearing
half: the project's central claim is that destructive SQL is unrepresentable, and an
unrepresentability claim is only worth what its tests are worth.
"""

from __future__ import annotations

import pytest

from sql_engine.ast_nodes import (
    Between,
    BinaryOp,
    BinaryOperator,
    ColumnRef,
    FunctionCall,
    InList,
    IsNull,
    Join,
    JoinType,
    Like,
    Literal,
    SelectStatement,
    SortDirection,
    Star,
    TableRef,
    UnaryOp,
)
from sql_engine.errors import NotSelectError, ParseError
from sql_engine.parser import parse


# --------------------------------------------------------------------------------------
# What cannot be expressed
# --------------------------------------------------------------------------------------

DESTRUCTIVE = [
    "DROP TABLE trades",
    "DROP DATABASE warehouse",
    "DELETE FROM trades",
    "DELETE FROM trades WHERE 1 = 1",
    "INSERT INTO trades VALUES (1)",
    "UPDATE trades SET price = 0",
    "TRUNCATE TABLE trades",
    "CREATE TABLE evil (a INT)",
    "ALTER TABLE trades ADD COLUMN x INT",
    "GRANT ALL ON trades TO PUBLIC",
    "ATTACH DATABASE 'x.db' AS x",
    "PRAGMA writable_schema = 1",
    "VACUUM",
    "BEGIN TRANSACTION",
]


@pytest.mark.parametrize("sql", DESTRUCTIVE)
def test_destructive_statements_do_not_parse(sql: str):
    """There is no AST node for any of these, so there is no way to build one."""
    with pytest.raises(NotSelectError):
        parse(sql)


def test_statement_stacking_is_rejected():
    """`SELECT 1; DROP TABLE t` must fail, not silently execute the first half."""
    with pytest.raises(ParseError):
        parse("SELECT 1 FROM t; DROP TABLE t;")


def test_comment_terminated_injection_is_rejected():
    with pytest.raises(ParseError):
        parse("SELECT a FROM t WHERE b = 1; DROP TABLE t -- ")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM a WHERE x IN (SELECT y FROM b)",
        "SELECT (SELECT 1) FROM a",
        "WITH c AS (SELECT 1) SELECT * FROM c",
        "SELECT * FROM a UNION SELECT * FROM b",
        "SELECT ROW_NUMBER() OVER (ORDER BY x) FROM a",
        "SELECT CASE WHEN x THEN 1 ELSE 2 END FROM a",
    ],
)
def test_unsupported_constructs_are_rejected(sql: str):
    with pytest.raises(ParseError):
        parse(sql)


def test_comma_join_is_rejected_with_a_useful_message():
    """Removing the implicit cross join removes the commonest accidental cartesian."""
    with pytest.raises(ParseError) as error:
        parse("SELECT * FROM a, b")
    assert "INNER JOIN" in str(error.value)


def test_unknown_function_is_rejected_and_names_what_exists():
    with pytest.raises(ParseError) as error:
        parse("SELECT ROUND(x, 2) FROM t")
    assert "AVG" in str(error.value)


def test_aggregate_in_where_is_rejected():
    with pytest.raises(ParseError):
        parse("SELECT a FROM t WHERE COUNT(*) > 1")


def test_negative_limit_is_rejected():
    with pytest.raises(ParseError):
        parse("SELECT a FROM t LIMIT -1")


# --------------------------------------------------------------------------------------
# What can be expressed
# --------------------------------------------------------------------------------------


def test_minimal_select():
    statement = parse("SELECT a FROM t")
    assert isinstance(statement, SelectStatement)
    assert statement.select_list[0].expr == ColumnRef("a")
    assert statement.from_item == TableRef("t")


def test_star_and_qualified_star():
    assert parse("SELECT * FROM t").select_list[0].expr == Star()
    assert parse("SELECT t.* FROM t").select_list[0].expr == Star("t")


def test_aliases_with_and_without_as():
    assert parse("SELECT a AS b FROM t").select_list[0].alias == "b"
    assert parse("SELECT a b FROM t").select_list[0].alias == "b"
    assert parse("SELECT a FROM t x").from_item.alias == "x"


def test_operator_precedence_multiplication_binds_tighter():
    expr = parse("SELECT 1 + 2 * 3 FROM t").select_list[0].expr
    assert expr.op is BinaryOperator.ADD
    assert expr.right.op is BinaryOperator.MUL


def test_and_binds_tighter_than_or():
    where = parse("SELECT a FROM t WHERE x OR y AND z").where
    assert where.op is BinaryOperator.OR
    assert where.right.op is BinaryOperator.AND


def test_not_binds_looser_than_comparison():
    where = parse("SELECT a FROM t WHERE NOT x = 1").where
    assert isinstance(where, UnaryOp)
    assert isinstance(where.operand, BinaryOp)


def test_parentheses_override_precedence():
    expr = parse("SELECT (1 + 2) * 3 FROM t").select_list[0].expr
    assert expr.op is BinaryOperator.MUL


def test_unary_minus_on_a_literal_is_folded():
    assert parse("SELECT -3 FROM t").select_list[0].expr == Literal(-3)


@pytest.mark.parametrize(
    "sql,node",
    [
        ("SELECT a FROM t WHERE x IN (1, 2)", InList),
        ("SELECT a FROM t WHERE x NOT IN (1, 2)", InList),
        ("SELECT a FROM t WHERE x BETWEEN 1 AND 2", Between),
        ("SELECT a FROM t WHERE x NOT BETWEEN 1 AND 2", Between),
        ("SELECT a FROM t WHERE x LIKE 'a%'", Like),
        ("SELECT a FROM t WHERE x NOT LIKE 'a%'", Like),
        ("SELECT a FROM t WHERE x IS NULL", IsNull),
        ("SELECT a FROM t WHERE x IS NOT NULL", IsNull),
    ],
)
def test_predicate_forms(sql: str, node: type):
    assert isinstance(parse(sql).where, node)


def test_negation_flags_are_recorded():
    assert parse("SELECT a FROM t WHERE x NOT IN (1)").where.negated is True
    assert parse("SELECT a FROM t WHERE x IN (1)").where.negated is False
    assert parse("SELECT a FROM t WHERE x IS NOT NULL").where.negated is True


def test_count_star_is_a_function_call_over_star():
    expr = parse("SELECT COUNT(*) FROM t").select_list[0].expr
    assert isinstance(expr, FunctionCall)
    assert expr.name == "COUNT"
    assert expr.args == (Star(),)


def test_count_distinct():
    expr = parse("SELECT COUNT(DISTINCT a) FROM t").select_list[0].expr
    assert expr.distinct is True


def test_join_types():
    assert parse("SELECT a FROM x JOIN y ON x.i = y.i").from_item.join_type is JoinType.INNER
    assert parse("SELECT a FROM x INNER JOIN y ON x.i = y.i").from_item.join_type is JoinType.INNER
    assert parse("SELECT a FROM x LEFT JOIN y ON x.i = y.i").from_item.join_type is JoinType.LEFT
    assert (
        parse("SELECT a FROM x LEFT OUTER JOIN y ON x.i = y.i").from_item.join_type
        is JoinType.LEFT
    )


def test_join_requires_on():
    with pytest.raises(ParseError):
        parse("SELECT a FROM x JOIN y")


def test_joins_are_left_deep():
    statement = parse("SELECT a FROM x JOIN y ON x.i = y.i JOIN z ON y.j = z.j")
    outer = statement.from_item
    assert isinstance(outer, Join)
    assert isinstance(outer.left, Join)
    assert outer.right == TableRef("z")


def test_full_clause_set():
    statement = parse(
        "SELECT DISTINCT a, COUNT(*) AS n FROM t WHERE x > 1 "
        "GROUP BY a HAVING COUNT(*) > 2 ORDER BY n DESC LIMIT 10 OFFSET 5"
    )
    assert statement.distinct is True
    assert statement.where is not None
    assert len(statement.group_by) == 1
    assert statement.having is not None
    assert statement.order_by[0].direction is SortDirection.DESC
    assert statement.limit == 10
    assert statement.offset == 5


def test_trailing_semicolon_is_allowed():
    assert parse("SELECT a FROM t;").limit is None


def test_ast_round_trips_through_str():
    """`str(statement)` must re-parse to the same tree.

    This is what makes LIMIT injection safe: the verifier rewrites the AST and prints it
    back out, so the printer has to be a genuine inverse of the parser.
    """
    original = (
        "SELECT DISTINCT a, COUNT(*) AS n FROM x INNER JOIN y ON x.i = y.i "
        "WHERE (a > 1 AND b LIKE 'z%') OR c IS NULL "
        "GROUP BY a HAVING COUNT(*) > 2 ORDER BY n DESC LIMIT 10 OFFSET 5"
    )
    once = parse(original)
    twice = parse(str(once))
    assert str(once) == str(twice)


def test_error_carries_a_position():
    with pytest.raises(ParseError) as error:
        parse("SELECT a FROM")
    assert error.value.line == 1
    assert error.value.column > 1
