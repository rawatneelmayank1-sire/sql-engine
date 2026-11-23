"""Prompt construction.

Two design choices worth defending:

**The system prompt describes the dialect's limits, not its dangers.** It does not say
"do not write DROP TABLE" -- saying so would imply that DROP is something the system
could execute if only the model were naughty enough, which is false and which invites
exactly the adversarial framing that makes models less reliable. It says what the
grammar contains, in the same tone as any other API documentation.

**Repair feedback quotes the verifier verbatim.** The engine's own error messages
already name the offending column and list the valid ones. Paraphrasing them into
friendlier prose would throw away the specificity that makes the repair work.
"""

from __future__ import annotations

from .verifier import VerificationResult

SYSTEM_PROMPT = """You translate questions about a data warehouse into SQL for an engine \
called sql-engine.

sql-engine implements a deliberately small subset of SQL. These are the only constructs \
it has:

  SELECT [DISTINCT] <expressions>
  FROM <table> [alias]
  [INNER | LEFT] JOIN <table> [alias] ON <equality condition>
  WHERE <condition>
  GROUP BY <expressions> [HAVING <condition>]
  ORDER BY <expressions> [ASC | DESC]
  LIMIT <n> [OFFSET <n>]

Conditions may use = <> < <= > >= AND OR NOT IN BETWEEN LIKE "IS NULL" "IS NOT NULL".
Aggregates are COUNT, SUM, AVG, MIN, MAX, and COUNT(DISTINCT x).

The engine has no subqueries, no CTEs, no window functions, no UNION, no CASE, no \
scalar functions (no DATE(), no ROUND(), no COALESCE(), no CAST), and no comma joins -- \
every join must be written as JOIN ... ON with an equality between the two tables.

Dates are stored as TEXT in ISO format ('2025-03-14'), so compare them as strings: \
trade_date >= '2025-03-01'.

Every column in a non-aggregated position must appear in GROUP BY.

Reply with one SQL statement and nothing else. No explanation, no markdown fences."""


def build_user_message(question: str, schema: str) -> str:
    """The first-attempt prompt."""
    return (
        f"Schema:\n{schema}\n\n"
        f"Question: {question}\n\n"
        "SQL:"
    )


def build_repair_message(
    question: str, schema: str, verdict: VerificationResult
) -> str:
    """The follow-up prompt after a rejection.

    The rejected SQL is included so the model can see what it wrote; models given only
    an error message frequently regenerate the same query.
    """
    return (
        f"Schema:\n{schema}\n\n"
        f"Question: {question}\n\n"
        f"You previously answered:\n{verdict.original_sql}\n\n"
        f"{verdict.feedback()}\n\n"
        "Corrected SQL:"
    )


__all__ = ["SYSTEM_PROMPT", "build_repair_message", "build_user_message"]
