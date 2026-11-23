"""Verification: decide whether generated SQL is allowed to run.

This module is the reason the project exists. Its central claim:

    The LLM cannot write a destructive query against this engine, because the engine
    cannot execute one.

That claim is *not* enforced here. It is enforced by ``sql_engine.ast_nodes``, which has no
node for INSERT, UPDATE, DELETE, DROP or CREATE, and by ``sql_engine.parser``, which has no
production that could build one. By the time a statement reaches this module it is a
``SelectStatement`` or it does not exist. There is no blocklist to bypass, no regex to
evade, no `; --` to append.

What this module adds is everything *above* that floor -- the checks that stop a query
that is safe but wrong or safe but ruinous:

1. it parses under the sql-engine grammar (so it is a SELECT by construction);
2. every table and column it names exists in the catalog;
3. no join is a cartesian product;
4. a LIMIT is present, or one is injected;
5. the estimated scan stays under a byte budget.

Failures are returned as structured categories, never as prose, because the repair loop
and the evaluation both route on category.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING

from ..errors import (
    AmbiguousColumnError,
    BindError,
    CatalogError,
    SQLEngineError,
    InvalidAggregationError,
    NotSelectError,
    ParseError,
    TokeniseError,
    UnknownColumnError,
)
from ..parser import parse
from ..plan import HashJoin, Scan, iter_plan

if TYPE_CHECKING:  # pragma: no cover
    from ..engine import CompiledQuery, Engine


class FailureCategory(str, Enum):
    """Why a candidate query was refused.

    The taxonomy is the evaluation's unit of measurement: "the verifier caught 11
    invented columns and 2 cartesian joins" is a statement you can check, whereas
    "the verifier caught 13 bad queries" is not.
    """

    NOT_SELECT = "not_select"
    COMMENT_TOKEN = "comment_token"
    UNPARSEABLE = "unparseable"
    UNKNOWN_TABLE = "unknown_table"
    UNKNOWN_COLUMN = "unknown_column"
    AMBIGUOUS_COLUMN = "ambiguous_column"
    INVALID_AGGREGATION = "invalid_aggregation"
    CARTESIAN_PRODUCT = "cartesian_product"
    SCAN_TOO_LARGE = "scan_too_large"
    INTERNAL = "internal"

    @property
    def description(self) -> str:
        return _CATEGORY_DESCRIPTIONS[self]


_CATEGORY_DESCRIPTIONS: dict[FailureCategory, str] = {
    FailureCategory.NOT_SELECT: "statement is not a SELECT",
    FailureCategory.COMMENT_TOKEN: "contains a SQL comment, which silently discards query text",
    FailureCategory.UNPARSEABLE: "does not parse under the sql-engine grammar",
    FailureCategory.UNKNOWN_TABLE: "references a table that does not exist",
    FailureCategory.UNKNOWN_COLUMN: "references a column that does not exist",
    FailureCategory.AMBIGUOUS_COLUMN: "column is ambiguous across joined tables",
    FailureCategory.INVALID_AGGREGATION: "mixes aggregated and non-aggregated columns",
    FailureCategory.CARTESIAN_PRODUCT: "join has no equality predicate",
    FailureCategory.SCAN_TOO_LARGE: "estimated scan exceeds the configured budget",
    FailureCategory.INTERNAL: "verification itself failed unexpectedly",
}


@dataclass(frozen=True, slots=True)
class Failure:
    """One reason a query was refused, in a form the repair prompt can quote."""

    category: FailureCategory
    message: str

    def __str__(self) -> str:
        return f"[{self.category.value}] {self.message}"


@dataclass(frozen=True, slots=True)
class Repair:
    """A change the verifier made to the candidate SQL rather than refusing it.

    Kept separate from Failure and surfaced in the UI: a query that ran because the
    verifier silently rewrote it is not the same thing as a query that was correct.
    """

    kind: str
    detail: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}"


@dataclass(slots=True)
class VerificationResult:
    """The verdict on one candidate query."""

    ok: bool
    #: The SQL that would actually run -- the candidate plus any repairs.
    sql: str
    original_sql: str
    failures: tuple[Failure, ...] = ()
    repairs: tuple[Repair, ...] = ()
    compiled: "CompiledQuery | None" = None
    estimated_scan_bytes: int = 0
    estimated_rows: int = 0

    @property
    def categories(self) -> tuple[FailureCategory, ...]:
        return tuple(failure.category for failure in self.failures)

    def feedback(self) -> str:
        """The text handed back to the model on a repair attempt.

        Deliberately specific and deliberately actionable: it names the failing
        construct and, where the catalog can, lists what would have been valid. A
        generic "invalid query" gives the model nothing to work with and turns the
        repair loop into three rolls of the same dice.
        """
        lines = ["The SQL you produced was rejected. Reasons:"]
        for failure in self.failures:
            lines.append(f"  - {failure.category.description}: {failure.message}")
        lines.append("Rewrite the query so that every reason above no longer applies.")
        return "\n".join(lines)


class Verifier:
    """Checks candidate SQL against the engine's own parser and catalog.

    Note what is *not* used: no sqlparse, no sqlglot, no regex over the query text. The
    grammar that decides whether the query is a SELECT is the same grammar that will
    execute it, so there is no gap between what the verifier accepts and what the engine
    runs. A separate parser would create exactly that gap, and the gap is where every
    SQL-injection bug in history lives.
    """

    def __init__(
        self,
        engine: "Engine",
        *,
        default_limit: int = 200,
        max_scan_bytes: int = 256 * 1024 * 1024,
        max_estimated_rows: int = 5_000_000,
        require_limit: bool = True,
    ) -> None:
        self.engine = engine
        self.default_limit = default_limit
        self.max_scan_bytes = max_scan_bytes
        self.max_estimated_rows = max_estimated_rows
        self.require_limit = require_limit

    def verify(self, sql: str) -> VerificationResult:
        """Run every check against ``sql``. Never raises; always returns a verdict."""
        original = sql.strip()

        # -- check 0: reject SQL comments outright --------------------------------
        # A comment is a *silent deletion*: everything after `--` vanishes, and the
        # remaining query still parses, still binds, and still runs -- it just does
        # something other than what the text appears to say. The adversarial corpus
        # has two cases that defeat every other check here for exactly this reason:
        #
        #   SELECT ticker FROM instruments -- AND is_active = 1
        #       -> the filter disappears and the query returns MORE rows
        #   SELECT ticker FROM instruments WHERE ticker = 'A' OR '1' = '1' --
        #       -> a tautology, with the terminator hiding whatever followed
        #
        # Neither is malformed, so parsing cannot catch them and neither can the
        # catalog. The defence is a policy rather than an analysis: model-generated
        # SQL has no legitimate reason to carry a comment, so its presence is itself
        # the signal. This is the one check that is about provenance, not syntax.
        if (offset := _comment_offset(original)) is not None:
            return _refused(
                original,
                FailureCategory.COMMENT_TOKEN,
                f"SQL comment at offset {offset}; comments silently discard the rest of "
                f"the statement and are never present in legitimate generated SQL",
            )

        # -- check 1: does it parse at all, under *this* grammar? -----------------
        try:
            statement = parse(original)
        except NotSelectError as error:
            return _refused(original, FailureCategory.NOT_SELECT, str(error))
        except (ParseError, TokeniseError) as error:
            return _refused(original, FailureCategory.UNPARSEABLE, str(error))

        # -- repair: inject a LIMIT if the model did not write one ----------------
        repairs: list[Repair] = []
        if self.require_limit and statement.limit is None:
            statement = replace(statement, limit=self.default_limit)
            repairs.append(
                Repair(
                    "limit-injected",
                    f"no LIMIT was present; LIMIT {self.default_limit} was added",
                )
            )
        effective_sql = str(statement) if repairs else original

        # -- checks 2 and 6: bind against the catalog and build the plan ----------
        try:
            compiled = self.engine.compile(effective_sql)
        except CatalogError as error:
            return _refused(original, FailureCategory.UNKNOWN_TABLE, str(error), repairs)
        except UnknownColumnError as error:
            return _refused(original, FailureCategory.UNKNOWN_COLUMN, str(error), repairs)
        except AmbiguousColumnError as error:
            return _refused(original, FailureCategory.AMBIGUOUS_COLUMN, str(error), repairs)
        except InvalidAggregationError as error:
            return _refused(
                original, FailureCategory.INVALID_AGGREGATION, str(error), repairs
            )
        except BindError as error:
            return _refused(original, FailureCategory.UNKNOWN_COLUMN, str(error), repairs)
        except (ParseError, TokeniseError) as error:
            # Only reachable if LIMIT injection produced something unparseable, which
            # would be a bug in the AST printer rather than in the model's output.
            return _refused(original, FailureCategory.INTERNAL, str(error), repairs)
        except SQLEngineError as error:  # pragma: no cover - defensive
            return _refused(original, FailureCategory.INTERNAL, str(error), repairs)

        failures: list[Failure] = []

        # -- check 3: no cartesian products ---------------------------------------
        for node in iter_plan(compiled.optimised_plan):
            if isinstance(node, HashJoin) and node.is_cartesian:
                failures.append(
                    Failure(
                        FailureCategory.CARTESIAN_PRODUCT,
                        "a join has no equality predicate, so it would pair every row "
                        "on the left with every row on the right. Add an ON clause "
                        "that equates a key column from each table.",
                    )
                )

        # -- check 5: scan budget ---------------------------------------------------
        scan_bytes = estimated_scan_bytes(compiled)
        estimated_rows = compiled.estimated_rows
        if scan_bytes > self.max_scan_bytes:
            failures.append(
                Failure(
                    FailureCategory.SCAN_TOO_LARGE,
                    f"the query would read about {scan_bytes / 1e6:.1f} MB, over the "
                    f"{self.max_scan_bytes / 1e6:.1f} MB budget. Add a WHERE clause "
                    "that narrows the scan, or select fewer columns.",
                )
            )
        if estimated_rows > self.max_estimated_rows:
            failures.append(
                Failure(
                    FailureCategory.SCAN_TOO_LARGE,
                    f"the query is estimated to produce {estimated_rows:,} rows, over "
                    f"the {self.max_estimated_rows:,} row budget.",
                )
            )

        if failures:
            return VerificationResult(
                ok=False,
                sql=effective_sql,
                original_sql=original,
                failures=tuple(failures),
                repairs=tuple(repairs),
                estimated_scan_bytes=scan_bytes,
                estimated_rows=estimated_rows,
            )

        return VerificationResult(
            ok=True,
            sql=effective_sql,
            original_sql=original,
            repairs=tuple(repairs),
            compiled=compiled,
            estimated_scan_bytes=scan_bytes,
            estimated_rows=estimated_rows,
        )


def estimated_scan_bytes(compiled: "CompiledQuery") -> int:
    """Bytes the optimised plan is expected to read off disk.

    Computed from the *optimised* plan on purpose: projection pruning changes this
    number, so the budget rewards a query that reads three columns rather than nine.
    The estimate is proportional -- table bytes scaled by the fraction of columns kept
    -- which ignores per-column width differences and is stated as approximate wherever
    it is shown.
    """
    total = 0.0
    for node in iter_plan(compiled.optimised_plan):
        if not isinstance(node, Scan):
            continue
        all_columns = len(node.table.columns) or 1
        kept = len(node.projection) if node.projection is not None else all_columns
        total += node.table.byte_size * (kept / all_columns)
    return int(total)


def _comment_offset(sql: str) -> int | None:
    """Offset of the first SQL comment outside a string literal, or ``None``.

    The subtlety is that ``--`` and ``/*`` are only comment introducers *outside* a
    quoted string. ``WHERE note = 'a--b'`` is a perfectly ordinary query and must not
    be refused, so a plain ``'--' in sql`` test is wrong. This walks the statement
    tracking quote state, including SQL's doubled-quote escape (``'it''s'``).

    Returns an offset rather than a bool so the failure message can point at the exact
    position -- a repair prompt that says "at offset 47" is far more actionable than
    one that says "somewhere".
    """
    in_string = False
    index = 0
    length = len(sql)
    while index < length:
        char = sql[index]
        if in_string:
            if char == "'":
                # A doubled quote is an escaped quote, not the end of the literal.
                if index + 1 < length and sql[index + 1] == "'":
                    index += 2
                    continue
                in_string = False
            index += 1
            continue
        if char == "'":
            in_string = True
            index += 1
            continue
        if char == "-" and sql.startswith("--", index):
            return index
        if char == "/" and sql.startswith("/*", index):
            return index
        index += 1
    return None


def _refused(
    sql: str,
    category: FailureCategory,
    message: str,
    repairs: list[Repair] | None = None,
) -> VerificationResult:
    return VerificationResult(
        ok=False,
        sql=sql,
        original_sql=sql,
        failures=(Failure(category, message),),
        repairs=tuple(repairs or ()),
    )


__all__ = [
    "Failure",
    "FailureCategory",
    "Repair",
    "VerificationResult",
    "Verifier",
    "estimated_scan_bytes",
]
