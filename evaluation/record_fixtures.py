"""Record real model responses so the evaluation can report real numbers.

    set GROQ_API_KEY=...
    python evaluation/record_fixtures.py
    python evaluation/run_eval.py --fixtures recorded

This drives the *same* :class:`~sql_engine.llm.pipeline.TextToSQL` the evaluation drives,
so the recorded fixtures include the repair turns as well as the first attempt -- the
repair prompt depends on the verifier's feedback, which depends on the first response,
so the two cannot be recorded independently.

Resumable by construction: :class:`~sql_engine.llm.client.RecordingClient` writes after
every call and never re-requests a key it already holds. A run that dies on question 30
of 45 costs one call when restarted, not thirty.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sql_engine import Engine  # noqa: E402
from sql_engine.llm import (  # noqa: E402
    DEFAULT_GROQ_MODEL,
    GroqClient,
    LLMError,
    RecordingClient,
    TextToSQL,
    Verifier,
)

WAREHOUSE = ROOT / "data" / "warehouse"
GOLDEN = HERE / "golden_set.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Record LLM fixtures from a live model.")
    parser.add_argument("--model", default=DEFAULT_GROQ_MODEL)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--output", type=Path, default=HERE / "fixtures" / "recorded.json")
    args = parser.parse_args()

    client = GroqClient(model=args.model)
    if not client.available:
        raise SystemExit(
            "GROQ_API_KEY is not set. The committed reference fixtures work without a "
            "key: run `python evaluation/run_eval.py`."
        )

    engine = Engine.from_directory(WAREHOUSE)
    recorder = RecordingClient(client, args.output)
    pipeline = TextToSQL(
        engine, recorder, verifier=Verifier(engine), max_attempts=args.max_attempts
    )

    cases = [
        json.loads(line)
        for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for index, case in enumerate(cases, start=1):
        try:
            answer = pipeline.answer(case["question"], key=case["id"], execute=False)
        except LLMError as error:
            print(f"[{index}/{len(cases)}] {case['id']}  ABORTED: {error}")
            raise SystemExit(1) from None
        status = "ok" if answer.ok else "unrecovered"
        print(
            f"[{index}/{len(cases)}] {case['id']}  {status}  "
            f"attempts={answer.attempt_count}"
        )

    print(f"\nwritten to {args.output}")
    print("now run: python evaluation/run_eval.py --fixtures recorded")


if __name__ == "__main__":
    main()
