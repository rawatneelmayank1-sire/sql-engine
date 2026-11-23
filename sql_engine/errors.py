"""Exception hierarchy for sql-engine.

Every failure mode gets its own class because the LLM verification layer routes on
*category*, not on message text. Matching on strings would silently break the repair
loop the moment a message is reworded.
"""

from __future__ import annotations


class SQLEngineError(Exception):
    """Root of every error raised by this package."""


class TokeniseError(SQLEngineError):
    """Raised when the input cannot be split into tokens at all."""

    def __init__(self, message: str, position: int, line: int, column: int) -> None:
        super().__init__(f"{message} (line {line}, column {column})")
        self.message = message
        self.position = position
        self.line = line
        self.column = column


class ParseError(SQLEngineError):
    """Raised when a token stream is not a sentence in the sql-engine grammar.

    This is the load-bearing error for the security argument: `DROP TABLE users` fails
    here because the grammar has no production that begins with DROP -- there is no
    blocklist to bypass.
    """

    def __init__(self, message: str, line: int, column: int, near: str = "") -> None:
        detail = f"{message} (line {line}, column {column})"
        if near:
            detail += f" near {near!r}"
        super().__init__(detail)
        self.message = message
        self.line = line
        self.column = column
        self.near = near


class NotSelectError(ParseError):
    """Raised when the input begins with a statement keyword from another SQL dialect.

    ``DROP TABLE users`` would fail as a plain ParseError anyway -- the grammar has no
    production for it. This subclass exists purely so the verifier can *report* the
    failure as "not a SELECT" rather than "syntax error at token 1", which is what the
    repair loop and the evaluation taxonomy need to distinguish.
    """


class CatalogError(SQLEngineError):
    """Raised when a query names a table the catalog does not contain."""


class BindError(SQLEngineError):
    """Raised when a syntactically valid query cannot be resolved against the schema."""


class UnknownColumnError(BindError):
    """A column that exists on no table in the query.

    The most common LLM failure by a wide margin: the model invents a plausible column
    name. It gets its own class so the evaluation can count it separately.
    """


class AmbiguousColumnError(BindError):
    """A column that exists on more than one relation and was not qualified."""


class InvalidAggregationError(BindError):
    """A select list that mixes bare columns with aggregates outside GROUP BY."""


class ExecutionError(SQLEngineError):
    """Raised when a bound plan fails at runtime (type mismatch, bad cast, ...)."""


class VerificationError(SQLEngineError):
    """Raised when generated SQL fails the safety checks and cannot be repaired."""
