"""Recursive-descent parser with precedence climbing for expressions.

Grammar (EBNF, complete -- there is nothing else):

    statement      = select_stmt [";"] EOF
    select_stmt    = "SELECT" ["DISTINCT"] select_list
                     ["FROM" from_clause]
                     ["WHERE" expression]
                     ["GROUP" "BY" expr_list ["HAVING" expression]]
                     ["ORDER" "BY" order_list]
                     ["LIMIT" integer] ["OFFSET" integer]
    from_clause    = table_ref { join_type "JOIN" table_ref "ON" expression }
    join_type      = ["INNER"] | "LEFT" ["OUTER"]
    expression     = or_expr
    or_expr        = and_expr { "OR" and_expr }
    and_expr       = not_expr { "AND" not_expr }
    not_expr       = "NOT" not_expr | predicate
    predicate      = additive [ comparison_tail ]
    comparison_tail= ("="|"<>"|"!="|"<"|"<="|">"|">=") additive
                   | ["NOT"] "IN" "(" expr_list ")"
                   | ["NOT"] "BETWEEN" additive "AND" additive
                   | ["NOT"] "LIKE" additive
                   | "IS" ["NOT"] "NULL"
    additive       = multiplicative { ("+"|"-") multiplicative }
    multiplicative = unary { ("*"|"/"|"%") unary }
    unary          = "-" unary | primary
    primary        = literal | function_call | column_ref | "(" expression ")"

Note what has no production: INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, ATTACH,
PRAGMA, UNION, subqueries, CTEs, window functions, CROSS JOIN. The parser cannot build
them because the tree has no nodes for them.
"""

from __future__ import annotations

from .ast_nodes import (
    AGGREGATE_FUNCTIONS,
    Between,
    BinaryOp,
    BinaryOperator,
    ColumnRef,
    Expression,
    FromItem,
    FunctionCall,
    InList,
    IsNull,
    Join,
    JoinType,
    Like,
    Literal,
    OrderByItem,
    SelectItem,
    SelectStatement,
    SortDirection,
    Star,
    TableRef,
    UnaryOp,
    contains_aggregate,
)
from .errors import NotSelectError, ParseError
from .tokeniser import Token, TokenType, tokenise

_COMPARISON_OPS: dict[str, BinaryOperator] = {
    "=": BinaryOperator.EQ,
    "<>": BinaryOperator.NEQ,
    "!=": BinaryOperator.NEQ,
    "<": BinaryOperator.LT,
    "<=": BinaryOperator.LTE,
    ">": BinaryOperator.GT,
    ">=": BinaryOperator.GTE,
}

#: Words that begin a statement in other SQL dialects but not in this one. Recognising
#: them buys a *better error message* only -- the parse fails either way. Keeping the
#: check here (rather than as a pre-parse blocklist) is deliberate: the rejection is
#: still structural, this list only makes the failure legible to a human or an LLM.
_FOREIGN_STATEMENT_WORDS: frozenset[str] = frozenset(
    {
        "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE", "REPLACE",
        "MERGE", "GRANT", "REVOKE", "ATTACH", "DETACH", "PRAGMA", "VACUUM", "BEGIN",
        "COMMIT", "ROLLBACK", "COPY", "CALL", "EXEC", "EXECUTE", "SET", "WITH", "VALUES",
    }
)


class Parser:
    """Parses a token stream into a :class:`SelectStatement`."""

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    # -- token helpers -----------------------------------------------------------

    @property
    def _current(self) -> Token:
        return self._tokens[self._pos]

    def _peek(self, offset: int = 0) -> Token:
        index = min(self._pos + offset, len(self._tokens) - 1)
        return self._tokens[index]

    def _at_keyword(self, *words: str) -> bool:
        token = self._current
        return token.type is TokenType.KEYWORD and token.value in words

    def _at_punct(self, char: str) -> bool:
        token = self._current
        return token.type is TokenType.PUNCTUATION and token.value == char

    def _at_operator(self, *ops: str) -> bool:
        token = self._current
        return token.type is TokenType.OPERATOR and token.value in ops

    def _advance(self) -> Token:
        token = self._current
        if token.type is not TokenType.EOF:
            self._pos += 1
        return token

    def _accept_keyword(self, *words: str) -> Token | None:
        if self._at_keyword(*words):
            return self._advance()
        return None

    def _expect_keyword(self, *words: str) -> Token:
        if self._at_keyword(*words):
            return self._advance()
        self._fail(f"expected {' or '.join(words)}")

    def _expect_punct(self, char: str) -> Token:
        if self._at_punct(char):
            return self._advance()
        self._fail(f"expected {char!r}")

    def _fail(self, message: str) -> None:
        token = self._current
        near = "end of input" if token.type is TokenType.EOF else token.value
        raise ParseError(message, token.line, token.column, near)

    # -- entry point -------------------------------------------------------------

    def parse(self) -> SelectStatement:
        """Parse exactly one SELECT statement and require the input to end there."""
        token = self._current
        if token.type is not TokenType.KEYWORD or token.value != "SELECT":
            if token.type is TokenType.IDENTIFIER and token.value.upper() in _FOREIGN_STATEMENT_WORDS:
                raise NotSelectError(
                    f"{token.value.upper()} is not part of this SQL dialect; "
                    "sql-engine implements SELECT only",
                    token.line,
                    token.column,
                    token.value,
                )
            self._fail("a query must begin with SELECT")

        statement = self._parse_select()

        if self._at_punct(";"):
            self._advance()
        if self._current.type is not TokenType.EOF:
            # Catches `SELECT 1; DROP TABLE t` -- the trailing statement is not silently
            # discarded, it is a parse error, so statement stacking cannot smuggle
            # anything past the front-end.
            self._fail("unexpected trailing input after the end of the query")
        return statement

    # -- clauses -----------------------------------------------------------------

    def _parse_select(self) -> SelectStatement:
        self._expect_keyword("SELECT")
        distinct = self._accept_keyword("DISTINCT") is not None
        select_list = self._parse_select_list()

        from_item: FromItem | None = None
        if self._accept_keyword("FROM"):
            from_item = self._parse_from_clause()

        where: Expression | None = None
        if self._accept_keyword("WHERE"):
            where = self._parse_expression()
            if contains_aggregate(where):
                self._fail("aggregate functions are not allowed in WHERE; use HAVING")

        group_by: tuple[Expression, ...] = ()
        having: Expression | None = None
        if self._accept_keyword("GROUP"):
            self._expect_keyword("BY")
            group_by = self._parse_expression_list()
            if self._accept_keyword("HAVING"):
                having = self._parse_expression()

        if having is None and self._accept_keyword("HAVING"):
            # HAVING without GROUP BY is legal SQL (one implicit group).
            having = self._parse_expression()

        order_by: tuple[OrderByItem, ...] = ()
        if self._accept_keyword("ORDER"):
            self._expect_keyword("BY")
            order_by = self._parse_order_by_list()

        limit: int | None = None
        offset: int | None = None
        if self._accept_keyword("LIMIT"):
            limit = self._parse_non_negative_integer("LIMIT")
        if self._accept_keyword("OFFSET"):
            offset = self._parse_non_negative_integer("OFFSET")

        return SelectStatement(
            select_list=select_list,
            from_item=from_item,
            where=where,
            group_by=group_by,
            having=having,
            order_by=order_by,
            limit=limit,
            offset=offset,
            distinct=distinct,
        )

    def _parse_non_negative_integer(self, clause: str) -> int:
        token = self._current
        if token.type is not TokenType.NUMBER or not token.value.isdigit():
            self._fail(f"{clause} requires a non-negative integer literal")
        self._advance()
        return int(token.value)

    def _parse_select_list(self) -> tuple[SelectItem, ...]:
        items = [self._parse_select_item()]
        while self._at_punct(","):
            self._advance()
            items.append(self._parse_select_item())
        return tuple(items)

    def _parse_select_item(self) -> SelectItem:
        if self._at_operator("*"):
            self._advance()
            return SelectItem(Star())
        # `t.*` -- identifier, dot, star.
        if (
            self._current.type is TokenType.IDENTIFIER
            and self._peek(1).type is TokenType.PUNCTUATION
            and self._peek(1).value == "."
            and self._peek(2).type is TokenType.OPERATOR
            and self._peek(2).value == "*"
        ):
            table = self._advance().value
            self._advance()  # dot
            self._advance()  # star
            return SelectItem(Star(table=table))

        expr = self._parse_expression()
        alias = self._parse_optional_alias()
        return SelectItem(expr, alias)

    def _parse_optional_alias(self) -> str | None:
        """Accept ``AS name`` or a bare ``name``.

        A bare alias must not swallow a following keyword (FROM, WHERE, ...), which is
        why only IDENTIFIER tokens qualify -- the tokeniser has already classified
        keywords, so this needs no extra look-ahead.
        """
        if self._accept_keyword("AS"):
            token = self._current
            if token.type is not TokenType.IDENTIFIER:
                self._fail("expected an alias name after AS")
            return self._advance().value
        if self._current.type is TokenType.IDENTIFIER:
            return self._advance().value
        return None

    def _parse_expression_list(self) -> tuple[Expression, ...]:
        exprs = [self._parse_expression()]
        while self._at_punct(","):
            self._advance()
            exprs.append(self._parse_expression())
        return tuple(exprs)

    def _parse_order_by_list(self) -> tuple[OrderByItem, ...]:
        items = [self._parse_order_by_item()]
        while self._at_punct(","):
            self._advance()
            items.append(self._parse_order_by_item())
        return tuple(items)

    def _parse_order_by_item(self) -> OrderByItem:
        expr = self._parse_expression()
        direction = SortDirection.ASC
        if self._accept_keyword("DESC"):
            direction = SortDirection.DESC
        else:
            self._accept_keyword("ASC")
        return OrderByItem(expr, direction)

    # -- FROM --------------------------------------------------------------------

    def _parse_from_clause(self) -> FromItem:
        """Parse a left-deep chain of joins.

        Left-deep only: the grammar has no parenthesised join expression, so the join
        tree shape is fixed and the planner never has to reassociate.
        """
        left: FromItem = self._parse_table_ref()
        while True:
            join_type = self._parse_optional_join_type()
            if join_type is None:
                return left
            right = self._parse_table_ref()
            self._expect_keyword("ON")
            condition = self._parse_expression()
            left = Join(left=left, right=right, join_type=join_type, condition=condition)

    def _parse_optional_join_type(self) -> JoinType | None:
        if self._at_keyword("JOIN"):
            self._advance()
            return JoinType.INNER
        if self._at_keyword("INNER"):
            self._advance()
            self._expect_keyword("JOIN")
            return JoinType.INNER
        if self._at_keyword("LEFT"):
            self._advance()
            self._accept_keyword("OUTER")
            self._expect_keyword("JOIN")
            return JoinType.LEFT
        if self._at_punct(","):
            # `FROM a, b` is the classic implicit cross join. Refusing it in the grammar
            # removes the single most common way an LLM writes an accidental cartesian
            # product; the verifier's check is then a second line of defence, not the only one.
            self._fail(
                "comma-separated tables are not supported; write an explicit "
                "INNER JOIN ... ON ... instead"
            )
        return None

    def _parse_table_ref(self) -> TableRef:
        token = self._current
        if token.type is not TokenType.IDENTIFIER:
            self._fail("expected a table name")
        name = self._advance().value
        alias: str | None = None
        if self._accept_keyword("AS"):
            if self._current.type is not TokenType.IDENTIFIER:
                self._fail("expected an alias name after AS")
            alias = self._advance().value
        elif self._current.type is TokenType.IDENTIFIER:
            alias = self._advance().value
        return TableRef(name=name, alias=alias)

    # -- expressions -------------------------------------------------------------

    def _parse_expression(self) -> Expression:
        return self._parse_or()

    def _parse_or(self) -> Expression:
        left = self._parse_and()
        while self._accept_keyword("OR"):
            right = self._parse_and()
            left = BinaryOp(BinaryOperator.OR, left, right)
        return left

    def _parse_and(self) -> Expression:
        left = self._parse_not()
        while self._accept_keyword("AND"):
            right = self._parse_not()
            left = BinaryOp(BinaryOperator.AND, left, right)
        return left

    def _parse_not(self) -> Expression:
        if self._accept_keyword("NOT"):
            return UnaryOp("NOT", self._parse_not())
        return self._parse_predicate()

    def _parse_predicate(self) -> Expression:
        left = self._parse_additive()

        if self._current.type is TokenType.OPERATOR and self._current.value in _COMPARISON_OPS:
            op = _COMPARISON_OPS[self._advance().value]
            right = self._parse_additive()
            return BinaryOp(op, left, right)

        negated = False
        if self._at_keyword("NOT") and self._peek(1).type is TokenType.KEYWORD and self._peek(
            1
        ).value in ("IN", "BETWEEN", "LIKE"):
            self._advance()
            negated = True

        if self._accept_keyword("IN"):
            self._expect_punct("(")
            if self._at_keyword("SELECT"):
                self._fail("subqueries are not supported")
            values = self._parse_expression_list()
            self._expect_punct(")")
            return InList(left, values, negated)

        if self._accept_keyword("BETWEEN"):
            low = self._parse_additive()
            self._expect_keyword("AND")
            high = self._parse_additive()
            return Between(left, low, high, negated)

        if self._accept_keyword("LIKE"):
            pattern = self._parse_additive()
            return Like(left, pattern, negated)

        if self._accept_keyword("IS"):
            is_negated = self._accept_keyword("NOT") is not None
            self._expect_keyword("NULL")
            return IsNull(left, is_negated)

        if negated:  # consumed a NOT that nothing claimed
            self._fail("expected IN, BETWEEN or LIKE after NOT")
        return left

    def _parse_additive(self) -> Expression:
        left = self._parse_multiplicative()
        while self._at_operator("+", "-"):
            op = BinaryOperator.ADD if self._advance().value == "+" else BinaryOperator.SUB
            left = BinaryOp(op, left, self._parse_multiplicative())
        return left

    def _parse_multiplicative(self) -> Expression:
        left = self._parse_unary()
        while self._at_operator("*", "/", "%"):
            symbol = self._advance().value
            op = {
                "*": BinaryOperator.MUL,
                "/": BinaryOperator.DIV,
                "%": BinaryOperator.MOD,
            }[symbol]
            left = BinaryOp(op, left, self._parse_unary())
        return left

    def _parse_unary(self) -> Expression:
        if self._at_operator("-"):
            self._advance()
            operand = self._parse_unary()
            # Fold `-3` into a literal so the optimiser sees a constant, not a tree.
            if isinstance(operand, Literal) and isinstance(operand.value, (int, float)):
                return Literal(-operand.value)
            return UnaryOp("-", operand)
        return self._parse_primary()

    def _parse_primary(self) -> Expression:
        token = self._current

        if token.type is TokenType.NUMBER:
            self._advance()
            text = token.value
            is_float = "." in text or "e" in text.lower()
            return Literal(float(text) if is_float else int(text))

        if token.type is TokenType.STRING:
            self._advance()
            return Literal(token.value)

        if token.type is TokenType.KEYWORD and token.value in ("TRUE", "FALSE"):
            self._advance()
            return Literal(token.value == "TRUE")

        if token.type is TokenType.KEYWORD and token.value == "NULL":
            self._advance()
            return Literal(None)

        if self._at_punct("("):
            self._advance()
            if self._at_keyword("SELECT"):
                self._fail("subqueries are not supported")
            inner = self._parse_expression()
            self._expect_punct(")")
            return inner

        if token.type is TokenType.IDENTIFIER:
            return self._parse_identifier_expression()

        self._fail("expected an expression")

    def _parse_identifier_expression(self) -> Expression:
        name_token = self._advance()
        name = name_token.value

        # Qualified column: `alias.column`
        if self._at_punct("."):
            self._advance()
            if self._current.type is not TokenType.IDENTIFIER:
                self._fail("expected a column name after '.'")
            column = self._advance().value
            return ColumnRef(name=column, table=name)

        # Function call
        if self._at_punct("("):
            upper = name.upper()
            if upper not in AGGREGATE_FUNCTIONS:
                raise ParseError(
                    f"unknown function {name!r}; sql-engine implements only "
                    + ", ".join(sorted(AGGREGATE_FUNCTIONS)),
                    name_token.line,
                    name_token.column,
                    name,
                )
            self._advance()  # "("
            distinct = self._accept_keyword("DISTINCT") is not None
            if self._at_operator("*"):
                self._advance()
                self._expect_punct(")")
                if upper != "COUNT":
                    self._fail("only COUNT accepts *")
                if distinct:
                    self._fail("COUNT(DISTINCT *) is not valid")
                return FunctionCall(upper, (Star(),))
            args = self._parse_expression_list()
            self._expect_punct(")")
            if len(args) != 1:
                raise ParseError(
                    f"{upper} takes exactly one argument, got {len(args)}",
                    name_token.line,
                    name_token.column,
                    name,
                )
            return FunctionCall(upper, args, distinct=distinct)

        return ColumnRef(name=name)


def parse(sql: str) -> SelectStatement:
    """Tokenise and parse ``sql``.

    Returns a :class:`SelectStatement` or raises :class:`~sql_engine.errors.ParseError` /
    :class:`~sql_engine.errors.TokeniseError`. There is no third outcome: if this returns,
    the caller is holding a read-only query.
    """
    return Parser(tokenise(sql)).parse()
