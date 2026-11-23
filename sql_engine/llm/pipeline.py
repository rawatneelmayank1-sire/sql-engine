"""Question -> SQL -> verification -> (maybe) execution, with a bounded repair loop.

The loop is the whole interface between an unreliable generator and a strict engine:

    for attempt in 1..N:
        sql = model(question, schema, previous_failure)
        verdict = verifier.verify(sql)
        if verdict.ok: execute and return
    fail loudly, with the reason

Two properties are non-negotiable and are asserted by tests:

* **Nothing unverified is ever executed.** ``execute`` is called only inside the
  ``verdict.ok`` branch, on the ``CompiledQuery`` the verifier itself produced -- not on
  the SQL string, which cannot drift from what was checked.
* **Failure is loud.** After N attempts the answer carries ``ok=False`` and the specific
  reason. There is no fallback query, no "best effort" result, and no empty result set
  masquerading as an answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..errors import SQLEngineError
from ..executor import ResultSet
from .client import LLMClient, LLMError, Prompt
from .prompts import SYSTEM_PROMPT, build_repair_message, build_user_message
from .verifier import FailureCategory, VerificationResult, Verifier

if TYPE_CHECKING:  # pragma: no cover
    from ..engine import CompiledQuery, Engine

#: Attempts before giving up. Three because the marginal value collapses: in the recorded
#: fixture set every repair that was going to succeed succeeded on attempt 2.
DEFAULT_MAX_ATTEMPTS = 3


@dataclass(slots=True)
class Attempt:
    """One trip round the loop."""

    index: int
    raw_response: str
    sql: str
    verification: VerificationResult

    @property
    def ok(self) -> bool:
        return self.verification.ok


@dataclass(slots=True)
class Answer:
    """The result of asking a question, including everything that happened on the way.

    The failed attempts are kept rather than discarded. The UI shows them, and "the
    model's first answer referenced a column that does not exist, here it is" is a far
    better demonstration than a clean final answer with no history.
    """

    question: str
    ok: bool
    attempts: list[Attempt] = field(default_factory=list)
    sql: str | None = None
    result: ResultSet | None = None
    compiled: "CompiledQuery | None" = None
    error: str | None = None

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def repair_count(self) -> int:
        """Number of *extra* model calls the verifier caused. Zero when it got it right."""
        return max(0, len(self.attempts) - 1)

    @property
    def first_failure_categories(self) -> tuple[FailureCategory, ...]:
        if not self.attempts:
            return ()
        return self.attempts[0].verification.categories

    @property
    def verification(self) -> VerificationResult | None:
        return self.attempts[-1].verification if self.attempts else None


class TextToSQL:
    """Turns a natural-language question into a verified, executed query."""

    def __init__(
        self,
        engine: "Engine",
        client: LLMClient,
        *,
        verifier: Verifier | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        self.engine = engine
        self.client = client
        self.verifier = verifier if verifier is not None else Verifier(engine)
        self.max_attempts = max_attempts

    def answer(self, question: str, *, key: str = "adhoc", execute: bool = True) -> Answer:
        """Ask ``question``. ``key`` identifies this turn for fixture replay."""
        answer = Answer(question=question, ok=False)
        schema = self.engine.schema_text()
        last: VerificationResult | None = None

        for index in range(self.max_attempts):
            user = (
                build_user_message(question, schema)
                if last is None
                else build_repair_message(question, schema, last)
            )
            prompt = Prompt(system=SYSTEM_PROMPT, user=user, key=f"{key}/attempt-{index}")
            try:
                raw = self.client.complete(prompt)
            except LLMError as error:
                answer.error = f"model call failed: {error}"
                return answer

            sql = extract_sql(raw)
            verdict = self.verifier.verify(sql)
            answer.attempts.append(Attempt(index, raw, sql, verdict))

            if verdict.ok:
                answer.sql = verdict.sql
                answer.compiled = verdict.compiled
                if execute and verdict.compiled is not None:
                    try:
                        # Executes the CompiledQuery the verifier produced, not the SQL
                        # text. Re-compiling would open a window for the two to differ.
                        answer.result = self.engine.execute(verdict.compiled)
                    except SQLEngineError as error:
                        # Verification is static; a type error can still surface at
                        # runtime. Reporting it as a failure rather than letting it
                        # escape keeps the promise that this method always returns an
                        # Answer, and keeps a runtime error out of the "verified"
                        # column of the evaluation.
                        answer.error = f"execution failed after verification: {error}"
                        return answer
                answer.ok = True
                return answer

            last = verdict

        answer.error = (
            f"gave up after {self.max_attempts} attempts. Last rejection: "
            + "; ".join(str(f) for f in (last.failures if last else ()))
        )
        return answer


_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_LEADING_PROSE = re.compile(r"\bSELECT\b", re.IGNORECASE)


def extract_sql(raw: str) -> str:
    """Pull the SQL out of a model response.

    Models wrap SQL in fences, prefix it with "Here is the query:", and append an
    explanation. Stripping all three is unglamorous but it is the difference between a
    verifier that rejects the model's prose and one that judges its SQL.

    Note the order of operations: the *first* fenced block wins, and the text is cut at
    the first ``SELECT`` only when no fence was found. A model that emits two statements
    still fails verification, because the parser rejects trailing input -- this function
    never has to decide whether a second statement is safe.
    """
    text = raw.strip()

    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    else:
        match = _LEADING_PROSE.search(text)
        if match:
            text = text[match.start() :].strip()

    # Drop a trailing explanation that follows a blank line after the statement.
    if ";" in text:
        text = text[: text.index(";") + 1]

    return text.strip()


__all__ = ["Answer", "Attempt", "TextToSQL", "extract_sql", "DEFAULT_MAX_ATTEMPTS"]
