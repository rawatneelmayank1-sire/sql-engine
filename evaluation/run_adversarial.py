"""Score the verifier against the adversarial corpus.

Why this exists separately from ``run_eval.py``:

``run_eval.py`` measures *end-to-end accuracy* — can the pipeline turn a question into
SQL that returns the right rows. That is a measurement of the **model**, and on a strong
model the verifier barely moves it, because a strong model does not invent column names.

This script measures the **verifier**, which is a different thing entirely. It feeds SQL
that is deliberately wrong, dangerous, or malformed straight past the model and asks a
single question: does the verifier refuse it? That is where a safety layer earns its
place — not on well-formed output, but on the tail.

The corpus (``eval_data/adversarial/cases.jsonl``, 48 cases) is labelled with the
expected verdict, and deliberately includes cases that must be **allowed**. A verifier
that rejects everything would be useless and would score 100% on rejections alone, so
precision on the allow-cases is reported alongside recall on the reject-cases.

Run:  python evaluation/run_adversarial.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sql_engine import Engine  # noqa: E402
from sql_engine.llm import Verifier  # noqa: E402

CASES = ROOT / "eval_data" / "adversarial" / "cases.jsonl"
WAREHOUSE = ROOT / "eval_data" / "golden" / "db"
OUT = HERE / "adversarial_results.md"


@dataclass(frozen=True, slots=True)
class Outcome:
    case_id: str
    reason: str
    expected: str
    actual: str
    detail: str

    @property
    def correct(self) -> bool:
        return self.expected == self.actual


def load_cases() -> list[dict[str, object]]:
    if not CASES.exists():
        raise SystemExit(f"adversarial corpus not found at {CASES}")
    return [json.loads(line) for line in CASES.read_text("utf-8").splitlines() if line.strip()]


def build_engine() -> Engine:
    """Load the golden warehouse, preferring Parquet where both formats exist.

    ``eval_data/golden/db`` deliberately ships each table as *both* CSV and Parquet so
    the corpus is usable either way. ``Engine.from_directory`` scans a directory and
    registers what it finds, so pointing it straight at that folder tries to register
    ``analysts`` twice and raises. Rather than weaken the catalog's duplicate check --
    which is a genuinely useful guard -- mirror the Parquet files into a scratch
    directory and load that. Parquet over CSV because the column types are stored
    rather than inferred.
    """
    import shutil
    import tempfile

    staging = Path(tempfile.mkdtemp(prefix="sql-engine-adversarial-"))
    for parquet in sorted(WAREHOUSE.glob("*.parquet")):
        shutil.copy2(parquet, staging / parquet.name)
    if not any(staging.iterdir()):
        raise SystemExit(f"no parquet tables found in {WAREHOUSE}")
    return Engine.from_directory(staging)


def run() -> tuple[list[Outcome], dict[str, int]]:
    engine = build_engine()
    verifier = Verifier(engine)

    outcomes: list[Outcome] = []
    for case in load_cases():
        sql = str(case["sql"])
        try:
            result = verifier.verify(sql)
            passed = bool(getattr(result, "ok", False))
            # Different failure shapes across versions; take whichever is present.
            detail = str(
                getattr(result, "reason", None)
                or getattr(result, "category", None)
                or getattr(result, "message", "")
            )[:120]
        except Exception as error:  # a raising verifier is still a refusal
            passed, detail = False, f"{type(error).__name__}: {error}"[:120]

        outcomes.append(
            Outcome(
                case_id=str(case["id"]),
                reason=str(case.get("reason", "")),
                expected=str(case["expect"]),
                actual="allow" if passed else "reject",
                detail=detail,
            )
        )

    counts = {
        "total": len(outcomes),
        "correct": sum(o.correct for o in outcomes),
        "should_reject": sum(o.expected == "reject" for o in outcomes),
        "caught": sum(o.expected == "reject" and o.actual == "reject" for o in outcomes),
        "should_allow": sum(o.expected == "allow" for o in outcomes),
        "allowed": sum(o.expected == "allow" and o.actual == "allow" for o in outcomes),
    }
    return outcomes, counts


def report(outcomes: list[Outcome], counts: dict[str, int]) -> str:
    by_reason: dict[str, list[Outcome]] = defaultdict(list)
    for o in outcomes:
        by_reason[o.reason].append(o)

    recall = counts["caught"] / counts["should_reject"] if counts["should_reject"] else 0.0
    # Precision here means: of everything refused, how much genuinely deserved it.
    refused = sum(o.actual == "reject" for o in outcomes)
    precision = counts["caught"] / refused if refused else 0.0

    lines = [
        "# Adversarial verification results",
        "",
        "Regenerate with `python evaluation/run_adversarial.py`. Nothing here is typed by hand.",
        "",
        "## What this measures",
        "",
        "`run_eval.py` measures the **model**: can it turn a question into correct SQL.",
        "This measures the **verifier**: given SQL that is wrong, dangerous or malformed,",
        "does it refuse. Those are different questions and they deserve different tables.",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Cases | {counts['total']} |",
        f"| Overall correct verdicts | {counts['correct']}/{counts['total']} = "
        f"{100 * counts['correct'] / counts['total']:.1f}% |",
        f"| Malicious/broken caught (recall) | {counts['caught']}/{counts['should_reject']} = "
        f"{100 * recall:.1f}% |",
        f"| Legitimate queries preserved | {counts['allowed']}/{counts['should_allow']} = "
        f"{100 * counts['allowed'] / counts['should_allow']:.1f}% |"
        if counts["should_allow"]
        else "| Legitimate queries preserved | n/a |",
        f"| Precision of refusals | {100 * precision:.1f}% |",
        "",
        "A verifier that refused everything would score 100% recall and would be useless.",
        "The allow-cases exist so that failure mode is visible rather than hidden.",
        "",
        "## By category",
        "",
        "| Category | Cases | Correct | |",
        "| --- | ---: | ---: | --- |",
    ]
    for reason, group in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        ok = sum(o.correct for o in group)
        mark = "" if ok == len(group) else " **<- misses**"
        lines.append(f"| `{reason}` | {len(group)} | {ok}/{len(group)} |{mark} |")

    misses = [o for o in outcomes if not o.correct]
    lines += ["", "## Every miss, listed", ""]
    if not misses:
        lines.append("None. Every case produced the expected verdict.")
    else:
        lines += ["| Case | Category | Expected | Got | Verifier said |", "| --- | --- | --- | --- | --- |"]
        for o in misses:
            lines.append(
                f"| `{o.case_id}` | `{o.reason}` | {o.expected} | {o.actual} | {o.detail or '—'} |"
            )
        lines += [
            "",
            "These are reported rather than tuned away. A miss on an `allow` case is a",
            "false positive: the verifier blocked something legitimate. A miss on a",
            "`reject` case is a false negative, which is the more dangerous direction.",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    outcomes, counts = run()
    text = report(outcomes, counts)
    OUT.write_text(text, encoding="utf-8")

    if not args.quiet:
        recall = counts["caught"] / counts["should_reject"] if counts["should_reject"] else 0
        print(f"cases                 {counts['total']}")
        print(f"correct verdicts      {counts['correct']}/{counts['total']} = "
              f"{100 * counts['correct'] / counts['total']:.1f}%")
        print(f"caught (recall)       {counts['caught']}/{counts['should_reject']} = {100 * recall:.1f}%")
        print(f"legit preserved       {counts['allowed']}/{counts['should_allow']}")
        print(f"written to            {OUT}")
        by = Counter(o.reason for o in outcomes if not o.correct)
        if by:
            print("misses by category   ", dict(by))


if __name__ == "__main__":
    main()
