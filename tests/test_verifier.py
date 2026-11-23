"""Verification-layer tests: deliberately malicious and deliberately broken SQL.

The project's headline claim is:

    "The LLM cannot write a destructive query against my engine, because my engine
    cannot execute one."

This file is where that claim is cashed. The first section throws real attack strings at
the front door. Note that none of them are *blocked* -- they simply fail to parse,
because the grammar has no production for them. There is no denylist in this codebase to
audit, bypass or forget to update.
"""

from __future__ import annotations

import pytest

from sql_engine import Engine
from sql_engine.llm import FailureCategory, Verifier

# --------------------------------------------------------------------------------------
# Destructive input
# --------------------------------------------------------------------------------------

MALICIOUS = [
    ("plain drop", "DROP TABLE trades"),
    ("drop database", "DROP DATABASE warehouse"),
    ("delete all", "DELETE FROM trades"),
    ("delete filtered", "DELETE FROM trades WHERE trade_id > 0"),
    ("insert", "INSERT INTO trades (trade_id) VALUES (1)"),
    ("update", "UPDATE traders SET desk = 'x'"),
    ("truncate", "TRUNCATE TABLE trades"),
    ("create", "CREATE TABLE backdoor (a INT)"),
    ("alter", "ALTER TABLE trades DROP COLUMN price"),
    ("grant", "GRANT ALL PRIVILEGES ON trades TO PUBLIC"),
    ("attach another database", "ATTACH DATABASE '/etc/passwd' AS leak"),
    ("pragma", "PRAGMA writable_schema = 1"),
    ("vacuum", "VACUUM"),
    ("transaction", "BEGIN; DELETE FROM trades; COMMIT;"),
]

INJECTION = [
    ("stacked statement", "SELECT trade_id FROM trades; DROP TABLE trades;"),
    ("stacked with comment", "SELECT trade_id FROM trades; DROP TABLE trades -- "),
    ("tautology and stack", "SELECT * FROM trades WHERE 1 = 1; DELETE FROM trades;"),
    ("union exfiltration", "SELECT symbol FROM instruments UNION SELECT desk FROM traders"),
    ("quoted drop as a value", "SELECT * FROM trades WHERE side = 'DROP TABLE trades'"),
]

#: Strings that *look* like injection but are ordinary queries once tokenised. These
#: must be allowed. A verifier that refused them would be pattern-matching on scary
#: text, which is the failure mode this design exists to avoid.
HARMLESS_LOOKALIKES = [
    # A destructive-looking string is still just a string.
    ("drop as a value", "SELECT trade_id FROM trades WHERE side = 'DROP TABLE trades'"),
    # Reserved-ish words as quoted identifiers.
    ("quoted identifier", 'SELECT "trade_id" FROM trades'),
    # `--` INSIDE a string literal is data, not a comment. This is the case the
    # comment check must not break, and the reason it tracks quote state instead of
    # doing `"--" in sql`.
    ("dashes inside a literal", "SELECT trade_id FROM trades WHERE side = 'a--b'"),
]


# Queries carrying a SQL comment. Refused by policy since 2026-09-09 -- see
# test_comments_are_refused_by_policy for the full argument.
COMMENTED = [
    ("payload inside a comment", "SELECT trade_id FROM trades WHERE trader_id = 1 -- ' OR '1'='1"),
    ("comment deletes a filter", "SELECT trade_id FROM trades -- AND trader_id = 1"),
    ("block comment", "SELECT trade_id FROM trades /* note */"),
]


@pytest.mark.parametrize("name,sql", MALICIOUS, ids=[n for n, _ in MALICIOUS])
def test_destructive_sql_is_refused(name: str, sql: str, engine: Engine):
    verdict = Verifier(engine).verify(sql)
    assert not verdict.ok
    assert verdict.categories[0] in (
        FailureCategory.NOT_SELECT,
        FailureCategory.UNPARSEABLE,
    )
    assert verdict.compiled is None, "a refused query must never produce a runnable plan"


@pytest.mark.parametrize("name,sql", INJECTION, ids=[n for n, _ in INJECTION])
def test_injection_shapes_are_refused_or_harmless(name: str, sql: str, engine: Engine):
    """Stacked statements are refused; a destructive string used as a *value* is fine.

    The last case matters: `WHERE side = 'DROP TABLE trades'` is a perfectly ordinary
    query about a string. An engine that pattern-matched on the text would reject it,
    which is how denylists end up blocking legitimate work while missing real attacks.
    """
    verdict = Verifier(engine).verify(sql)
    if name == "quoted drop as a value":
        assert verdict.ok, "a destructive-looking string literal is still just a string"
        assert engine.execute(verdict.compiled).rows == []
    else:
        assert not verdict.ok


@pytest.mark.parametrize(
    "name,sql", HARMLESS_LOOKALIKES, ids=[n for n, _ in HARMLESS_LOOKALIKES]
)
def test_scary_looking_but_legitimate_queries_are_allowed(
    name: str, sql: str, engine: Engine
):
    """False positives are a real cost, and this design does not pay it.

    Because rejection is structural rather than textual, a query containing the word
    DROP inside a string literal, or an injection payload inside a comment, is judged on
    what it *is* rather than on what it looks like.
    """
    verdict = Verifier(engine).verify(sql)
    assert verdict.ok, f"{name}: {[str(f) for f in verdict.failures]}"
    engine.execute(verdict.compiled)  # and it actually runs


@pytest.mark.parametrize("name,sql", COMMENTED, ids=[n for n, _ in COMMENTED])
def test_comments_are_refused_by_policy(name: str, sql: str, engine: Engine):
    """A SQL comment is a silent deletion, so generated SQL may not contain one.

    This deliberately reverses an earlier decision, and the reasoning is worth keeping
    because both positions are defensible.

    The original view: a payload inside a `--` comment is *neutralised*. The tokeniser
    discards it, the remaining query is an ordinary filter, and refusing it would be a
    false positive of exactly the textual-denylist kind this design avoids. That is
    correct as far as **safety** goes -- nothing dangerous can execute.

    What it misses is **correctness**. The adversarial corpus contains

        SELECT ticker FROM instruments -- AND is_active = 1

    which is not neutralised at all: the comment deletes a filter, so the query returns
    *more* rows than its own text says it does. It parses, it binds, it passes every
    structural check, and it is silently wrong. No amount of syntax analysis catches
    that, because there is nothing syntactically wrong with it.

    So the rule is about provenance rather than syntax: a model asked for a query has
    no legitimate reason to emit a comment, and the cost of refusing one is a single
    regeneration. A `--` inside a string literal is data and stays allowed -- see
    "dashes inside a literal" above, which is why the check tracks quote state.
    """
    verdict = Verifier(engine).verify(sql)
    assert not verdict.ok, f"{name} should be refused"
    assert verdict.categories[0] is FailureCategory.COMMENT_TOKEN
    assert verdict.compiled is None, "a refused query must never produce a runnable plan"


def test_no_write_path_exists_in_the_package():
    """The floor beneath every check above: there is no code that could write.

    Greps the whole package for the AST node names a destructive statement would need.
    A future refactor that adds one has to delete this test to pass, which is the point.
    """
    import sql_engine.ast_nodes as ast_nodes

    forbidden = {"Insert", "Update", "Delete", "Drop", "CreateTable", "Alter", "Truncate"}
    assert not forbidden & set(dir(ast_nodes))


# --------------------------------------------------------------------------------------
# Broken but non-malicious input -- the everyday case
# --------------------------------------------------------------------------------------

BROKEN = [
    ("invented column", "SELECT lifetime_value FROM traders", FailureCategory.UNKNOWN_COLUMN),
    (
        "invented column qualified",
        "SELECT t.pnl FROM trades t",
        FailureCategory.UNKNOWN_COLUMN,
    ),
    ("invented table", "SELECT * FROM customers", FailureCategory.UNKNOWN_TABLE),
    (
        "ambiguous column",
        "SELECT instrument_id FROM trades t INNER JOIN instruments i "
        "ON t.instrument_id = i.instrument_id",
        FailureCategory.AMBIGUOUS_COLUMN,
    ),
    (
        "bare column with aggregate",
        "SELECT side, price, COUNT(*) FROM trades GROUP BY side",
        FailureCategory.INVALID_AGGREGATION,
    ),
    (
        "explicit cartesian",
        "SELECT t.trade_id FROM trades t INNER JOIN instruments i ON 1 = 1",
        FailureCategory.CARTESIAN_PRODUCT,
    ),
    (
        "comma cartesian",
        "SELECT * FROM trades, instruments",
        FailureCategory.UNPARSEABLE,
    ),
    (
        "unsupported function",
        "SELECT ROUND(AVG(price), 2) FROM trades",
        FailureCategory.UNPARSEABLE,
    ),
    (
        "subquery",
        "SELECT * FROM trades WHERE trader_id IN (SELECT trader_id FROM traders)",
        FailureCategory.UNPARSEABLE,
    ),
    ("prose, not sql", "I think you want all the trades", FailureCategory.UNPARSEABLE),
    ("empty string", "", FailureCategory.UNPARSEABLE),
    ("window function", "SELECT SUM(price) OVER () FROM trades", FailureCategory.UNPARSEABLE),
]


@pytest.mark.parametrize("name,sql,expected", BROKEN, ids=[n for n, _, _ in BROKEN])
def test_broken_sql_is_categorised(
    name: str, sql: str, expected: FailureCategory, engine: Engine
):
    verdict = Verifier(engine).verify(sql)
    assert not verdict.ok
    assert expected in verdict.categories, f"{name}: got {verdict.categories}"


def test_unknown_column_error_lists_the_real_ones(engine: Engine):
    """The repair loop is only as good as the feedback it forwards."""
    verdict = Verifier(engine).verify("SELECT desk_name FROM traders")
    message = verdict.failures[0].message
    assert "desk" in message and "region" in message


def test_feedback_names_the_category_and_the_detail(engine: Engine):
    verdict = Verifier(engine).verify("SELECT lifetime_value FROM traders")
    feedback = verdict.feedback()
    assert "column" in feedback.lower()
    assert "lifetime_value" in feedback


# --------------------------------------------------------------------------------------
# Repairs and budgets
# --------------------------------------------------------------------------------------


def test_missing_limit_is_injected_not_refused(engine: Engine):
    verdict = Verifier(engine, default_limit=25).verify("SELECT trade_id FROM trades")
    assert verdict.ok
    assert "LIMIT 25" in verdict.sql
    assert verdict.repairs[0].kind == "limit-injected"


def test_existing_limit_is_left_alone(engine: Engine):
    verdict = Verifier(engine, default_limit=25).verify("SELECT trade_id FROM trades LIMIT 3")
    assert verdict.ok
    assert not verdict.repairs
    assert engine.execute(verdict.compiled).rows.__len__() == 3


def test_injected_limit_produces_re_parseable_sql(engine: Engine):
    """LIMIT injection rewrites the AST and prints it back; the printer must be exact."""
    original = (
        "SELECT i.exchange, COUNT(*) AS n FROM trades t INNER JOIN instruments i "
        "ON t.instrument_id = i.instrument_id WHERE t.price BETWEEN 100 AND 200 "
        "GROUP BY i.exchange HAVING COUNT(*) > 1 ORDER BY n DESC"
    )
    verdict = Verifier(engine).verify(original)
    assert verdict.ok
    assert engine.compile(verdict.sql) is not None


def test_scan_budget_refuses_an_oversized_query(engine: Engine):
    verdict = Verifier(engine, max_scan_bytes=1_000).verify("SELECT * FROM trades LIMIT 10")
    assert not verdict.ok
    assert FailureCategory.SCAN_TOO_LARGE in verdict.categories


def test_projection_pruning_can_bring_a_query_under_budget(engine: Engine):
    """The budget is computed on the *optimised* plan, so reading less costs less.

    Same table, same row count; the only difference is how many columns the query
    touches. A budget applied before optimisation would refuse both.
    """
    tight = Verifier(engine, max_scan_bytes=2_000_000)
    assert not tight.verify("SELECT * FROM trades LIMIT 10").ok
    assert tight.verify("SELECT trade_id FROM trades LIMIT 10").ok


def test_row_budget_refuses_a_huge_estimate(engine: Engine):
    verdict = Verifier(engine, max_estimated_rows=10, require_limit=False).verify(
        "SELECT trade_id FROM trades"
    )
    assert not verdict.ok
    assert FailureCategory.SCAN_TOO_LARGE in verdict.categories


def test_verification_never_raises(engine: Engine):
    """Whatever a model produces, verification returns a verdict rather than exploding."""
    verifier = Verifier(engine)
    for text in [
        "",
        "   ",
        "SELECT",
        "SELECT FROM",
        "-- just a comment",
        "/* unterminated",
        "SELECT 'unterminated",
        "SELECT \x00 FROM t",
        "SELECT " + "a," * 200 + "b FROM trades",
        "SELECT * FROM " + "t" * 5000,
        "🙂",
    ]:
        verdict = verifier.verify(text)
        assert isinstance(verdict.ok, bool)
        if not verdict.ok:
            assert verdict.failures
