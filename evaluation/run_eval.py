"""Evaluate the text-to-SQL front-end, with and without the verification layer.

The single number this project cares about is the *gap*: execution accuracy when the
model's first answer is executed directly, versus execution accuracy when every
candidate must pass verification and failures are fed back for repair.

Execution accuracy is measured the way the text-to-SQL literature measures it: run the
gold query and the predicted query, compare the result sets. Two queries that are
textually different but return the same rows both count as correct, which is the only
definition that survives contact with real SQL.

The gold query is executed through **SQLite**, not through sql-engine. Using sql-engine for
both sides would make the evaluation blind to a class of engine bug that happens to
affect gold and prediction identically.

Run:  python evaluation/run_eval.py
      python evaluation/run_eval.py --fixtures recorded   (after record_fixtures.py)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sql_engine import Engine  # noqa: E402
from sql_engine.errors import SQLEngineError  # noqa: E402
from sql_engine.llm import (  # noqa: E402
    FailureCategory,
    ReplayClient,
    TextToSQL,
    Verifier,
)
from sql_engine.sqlite_oracle import build_sqlite, results_match, run_sqlite  # noqa: E402

WAREHOUSE = ROOT / "data" / "warehouse"
GOLDEN = HERE / "golden_set.jsonl"
RESULTS = HERE / "results.md"


@dataclass(slots=True)
class Case:
    question_id: str
    question: str
    gold_sql: str


@dataclass(slots=True)
class Outcome:
    """What happened for one question, under both regimes."""

    case: Case
    #: Attempt-0 SQL executed directly, no verification, no repair.
    unverified_correct: bool
    unverified_note: str
    #: Full pipeline: verify, repair up to N times, execute only if verified.
    verified_correct: bool
    verified_note: str
    attempts: int
    first_failure: tuple[FailureCategory, ...] = ()
    ran_unsafe: bool = False


@dataclass(slots=True)
class Report:
    outcomes: list[Outcome] = field(default_factory=list)
    fixture_source: str = ""
    max_attempts: int = 3

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def unverified_correct(self) -> int:
        return sum(1 for o in self.outcomes if o.unverified_correct)

    @property
    def verified_correct(self) -> int:
        return sum(1 for o in self.outcomes if o.verified_correct)

    @property
    def rejected_first_attempt(self) -> int:
        return sum(1 for o in self.outcomes if o.first_failure)

    @property
    def category_counts(self) -> Counter:
        counter: Counter = Counter()
        for outcome in self.outcomes:
            for category in outcome.first_failure:
                counter[category.value] += 1
        return counter

    @property
    def mean_attempts(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(o.attempts for o in self.outcomes) / len(self.outcomes)

    @property
    def mean_repairs(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(max(0, o.attempts - 1) for o in self.outcomes) / len(self.outcomes)

    @property
    def unsafe_executions(self) -> int:
        return sum(1 for o in self.outcomes if o.ran_unsafe)


def load_cases() -> list[Case]:
    cases = []
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        cases.append(Case(payload["id"], payload["question"], payload["gold_sql"]))
    return cases


def _compare(predicted_rows: list[tuple[Any, ...]], gold_rows: list[tuple[Any, ...]],
             gold_sql: str) -> bool:
    ordered = " order by " in gold_sql.lower()
    matched, _ = results_match(predicted_rows, gold_rows, ordered=ordered)
    return matched


def evaluate(fixtures: Path, max_attempts: int = 3) -> Report:
    engine = Engine.from_directory(WAREHOUSE)
    connection = build_sqlite(engine.catalog)
    client = ReplayClient.from_file(fixtures)
    payload = json.loads(fixtures.read_text(encoding="utf-8"))
    pipeline = TextToSQL(
        engine, client, verifier=Verifier(engine), max_attempts=max_attempts
    )

    report = Report(
        fixture_source=payload.get("recorded_from", str(fixtures.name)),
        max_attempts=max_attempts,
    )

    for case in load_cases():
        gold_rows = run_sqlite(connection, case.gold_sql)

        # -- regime A: no verification. Execute attempt 0 exactly as written. --------
        raw = client.complete(
            _FakePrompt(f"{case.question_id}/attempt-0")
        )
        candidate = _extract(raw)
        unverified_correct = False
        ran_unsafe = False
        try:
            # Executing through sql-engine is the *only* option here, and it is worth
            # noticing why the unverified regime still cannot do damage: the engine has
            # no destructive operation to reach. What the verifier adds is protection
            # from queries that are merely wrong or ruinously expensive.
            result = engine.query(candidate)
            unverified_correct = _compare(result.rows, gold_rows, case.gold_sql)
            unverified_note = "executed" if unverified_correct else "executed, wrong rows"
            ran_unsafe = True
        except SQLEngineError as error:
            unverified_note = f"{type(error).__name__}: {str(error)[:90]}"

        # -- regime B: full pipeline --------------------------------------------------
        answer = pipeline.answer(case.question, key=case.question_id)
        if answer.ok and answer.result is not None:
            verified_correct = _compare(answer.result.rows, gold_rows, case.gold_sql)
            verified_note = "verified" if verified_correct else "verified, wrong rows"
        else:
            verified_correct = False
            verified_note = (answer.error or "no answer")[:110]

        report.outcomes.append(
            Outcome(
                case=case,
                unverified_correct=unverified_correct,
                unverified_note=unverified_note,
                verified_correct=verified_correct,
                verified_note=verified_note,
                attempts=answer.attempt_count,
                first_failure=answer.first_failure_categories,
                ran_unsafe=ran_unsafe and not answer.ok,
            )
        )
    return report


class _FakePrompt:
    """Minimal stand-in so regime A can pull attempt 0 without running the pipeline."""

    def __init__(self, key: str) -> None:
        self.key = key
        self.system = ""
        self.user = ""


def _extract(raw: str) -> str:
    from sql_engine.llm import extract_sql

    return extract_sql(raw)


def render(report: Report) -> str:
    total = report.total
    without = report.unverified_correct
    with_verification = report.verified_correct
    gap = with_verification - without

    lines: list[str] = []
    lines.append("# Evaluation results")
    lines.append("")
    lines.append(f"Generated by `python evaluation/run_eval.py` on {date.today().isoformat()}.")
    lines.append("Regenerate with that one command; nothing here is typed by hand.")
    lines.append("")
    lines.append("## Provenance of the candidate SQL")
    lines.append("")
    lines.append(f"Fixture source: **{report.fixture_source}**")
    lines.append("")
    if "hand-authored" in report.fixture_source:
        lines.append(
            "> These candidate queries were written by hand in "
            "`evaluation/build_reference_fixtures.py`, not recorded from a live model, "
            "because no LLM API key was available when this project was built. "
            "**The execution-accuracy gap below is therefore a property of this fixture "
            "set, not a measurement of any model.** The verification catch-rate *is* a "
            "real measurement: it is this repository's verifier judging these queries. "
            "Set `GROQ_API_KEY`, run `evaluation/record_fixtures.py`, then "
            "`run_eval.py --fixtures recorded` to regenerate every number below against "
            "a real model."
        )
        lines.append("")
    lines.append(f"Questions: {total}. Max repair attempts: {report.max_attempts}.")
    lines.append("")

    lines.append("## Execution accuracy")
    lines.append("")
    lines.append("| Regime | Correct | Accuracy |")
    lines.append("| --- | ---: | ---: |")
    lines.append(
        f"| First answer executed directly (no verification) | {without}/{total} | "
        f"{without / total:.1%} |"
    )
    lines.append(
        f"| Verified, with repair loop | {with_verification}/{total} | "
        f"{with_verification / total:.1%} |"
    )
    lines.append(
        f"| **Difference** | **{gap:+d}** | "
        f"**{(with_verification - without) / total:+.1%}** |"
    )
    lines.append("")

    lines.append("## What the verifier caught, by category")
    lines.append("")
    lines.append(
        f"{report.rejected_first_attempt} of {total} first answers were refused."
    )
    lines.append("")
    lines.append("| Category | First answers refused |")
    lines.append("| --- | ---: |")
    counts = report.category_counts
    for category in FailureCategory:
        count = counts.get(category.value, 0)
        if count:
            lines.append(f"| `{category.value}` — {category.description} | {count} |")
    lines.append("")

    lines.append("## Repair loop")
    lines.append("")
    lines.append(f"- Mean attempts per question: **{report.mean_attempts:.2f}**")
    lines.append(f"- Mean repair calls per question: **{report.mean_repairs:.2f}**")
    repaired = sum(
        1 for o in report.outcomes if o.first_failure and o.verified_correct
    )
    unrecovered = sum(
        1 for o in report.outcomes if o.first_failure and not o.verified_correct
    )
    lines.append(f"- Refused answers repaired into a correct one: **{repaired}**")
    lines.append(f"- Refused answers never recovered within the attempt budget: **{unrecovered}**")
    lines.append("")

    lines.append("## Queries the verifier stopped that would otherwise have run")
    lines.append("")
    lines.append(
        f"{report.unsafe_executions} question(s) produced a first answer that executed "
        "when run directly, but was refused by the verifier and never repaired into a "
        "correct query. Note what these are *not*: none of them could have damaged "
        "anything, because the engine has no destructive operation to reach. What the "
        "verifier adds on top of that floor is protection from queries that are merely "
        "wrong or ruinously expensive."
    )
    lines.append("")

    lines.append("## The verifier's blind spot")
    lines.append("")
    blind = [
        o
        for o in report.outcomes
        if not o.first_failure and not o.verified_correct
    ]
    lines.append(
        f"{len(blind)} question(s) were answered with SQL that is valid, parses, binds "
        "against the catalog, passes every safety check — and returns the wrong rows. "
        "Structural verification cannot catch a semantic error, and this table is here "
        "so that limitation is stated rather than buried."
    )
    lines.append("")
    if blind:
        lines.append("| Question | Why it is wrong |")
        lines.append("| --- | --- |")
        for outcome in blind:
            lines.append(f"| {outcome.case.question_id} | {outcome.case.question} |")
        lines.append("")

    lines.append("## Per-question detail")
    lines.append("")
    lines.append("| id | unverified | verified | attempts | first refusal |")
    lines.append("| --- | --- | --- | ---: | --- |")
    for outcome in report.outcomes:
        first = ", ".join(c.value for c in outcome.first_failure) or "—"
        lines.append(
            f"| {outcome.case.question_id} "
            f"| {'pass' if outcome.unverified_correct else 'fail'} "
            f"| {'pass' if outcome.verified_correct else 'fail'} "
            f"| {outcome.attempts} | {first} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sql-engine text-to-SQL.")
    parser.add_argument(
        "--fixtures",
        default="reference",
        help="fixture set name under evaluation/fixtures (default: reference)",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=None,
        help=(
            "exit non-zero if verified accuracy falls below this fraction. Used in CI "
            "as a regression guard; a date-stamped report cannot be diffed, but the "
            "number it contains can be asserted."
        ),
    )
    args = parser.parse_args()

    path = HERE / "fixtures" / f"{args.fixtures}.json"
    report = evaluate(path, max_attempts=args.max_attempts)
    text = render(report)
    args.output.write_text(text, encoding="utf-8")

    print(f"questions               {report.total}")
    print(
        f"accuracy (unverified)   {report.unverified_correct}/{report.total} "
        f"= {report.unverified_correct / report.total:.1%}"
    )
    print(
        f"accuracy (verified)     {report.verified_correct}/{report.total} "
        f"= {report.verified_correct / report.total:.1%}"
    )
    print(f"first answers refused   {report.rejected_first_attempt}")
    print(f"mean repair calls       {report.mean_repairs:.2f}")
    print(f"written to              {args.output}")

    if args.min_accuracy is not None:
        accuracy = report.verified_correct / report.total
        if accuracy < args.min_accuracy:
            raise SystemExit(
                f"verified accuracy {accuracy:.1%} is below the required "
                f"{args.min_accuracy:.1%}"
            )


if __name__ == "__main__":
    main()
