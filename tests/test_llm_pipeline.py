"""Repair-loop tests. No network, no API key -- every model response is a fixture.

The two properties this file exists to hold:

1. nothing unverified is ever executed;
2. when the loop gives up, it says so loudly and returns no result.

Both are easy to break with a well-meaning "fall back to something" branch, and neither
failure would be visible from the outside until it mattered.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sql_engine import Engine
from sql_engine.llm import (
    FailureCategory,
    FixtureMissingError,
    Prompt,
    ReplayClient,
    StaticClient,
    TextToSQL,
    Verifier,
    extract_sql,
)

FIXTURES = Path(__file__).resolve().parent.parent / "evaluation" / "fixtures"


def pipeline(engine: Engine, responses: list[str], max_attempts: int = 3) -> TextToSQL:
    return TextToSQL(
        engine,
        StaticClient(responses),
        verifier=Verifier(engine),
        max_attempts=max_attempts,
    )


# --------------------------------------------------------------------------------------
# extract_sql
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("SELECT 1", "SELECT 1"),
        ("```sql\nSELECT 1\n```", "SELECT 1"),
        ("```\nSELECT 1\n```", "SELECT 1"),
        ("Here you go:\n```sql\nSELECT 1;\n```\nHope that helps!", "SELECT 1;"),
        ("Sure! SELECT a FROM b", "SELECT a FROM b"),
        ("SELECT a FROM b;\n\nThis counts the rows.", "SELECT a FROM b;"),
        ("  \n SELECT 1 \n ", "SELECT 1"),
    ],
)
def test_extract_sql(raw: str, expected: str):
    assert extract_sql(raw) == expected


def test_extract_sql_leaves_non_sql_alone_for_the_verifier_to_reject():
    """Extraction must not try to be clever. Judging is the verifier's job."""
    assert extract_sql("I cannot answer that.") == "I cannot answer that."


# --------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------


def test_valid_first_answer_needs_no_repair(engine: Engine):
    answer = pipeline(engine, ["SELECT COUNT(*) AS n FROM trades LIMIT 10"]).answer("how many?")
    assert answer.ok
    assert answer.attempt_count == 1
    assert answer.repair_count == 0
    assert answer.result is not None and answer.result.rows == [(200_000,)]


def test_invalid_answer_is_repaired(engine: Engine):
    answer = pipeline(
        engine,
        [
            "SELECT lifetime_value FROM traders LIMIT 10",
            "SELECT trader_name FROM traders LIMIT 10",
        ],
    ).answer("who are the traders?")
    assert answer.ok
    assert answer.attempt_count == 2
    assert answer.repair_count == 1
    assert answer.first_failure_categories == (FailureCategory.UNKNOWN_COLUMN,)


def test_destructive_answer_is_never_executed(engine: Engine):
    """The claim, in one test."""
    answer = pipeline(engine, ["DROP TABLE trades"], max_attempts=1).answer("delete it all")
    assert not answer.ok
    assert answer.result is None
    assert answer.compiled is None
    assert answer.attempts[0].verification.categories == (FailureCategory.NOT_SELECT,)


def test_giving_up_is_loud(engine: Engine):
    answer = pipeline(engine, ["SELECT nope FROM traders"], max_attempts=3).answer("?")
    assert not answer.ok
    assert answer.result is None
    assert answer.error is not None
    assert "gave up after 3 attempts" in answer.error
    assert "unknown_column" in answer.error


def test_attempt_budget_is_respected(engine: Engine):
    client = StaticClient(["SELECT nope FROM traders"])
    answer = TextToSQL(engine, client, max_attempts=2).answer("?")
    assert len(client.calls) == 2
    assert answer.attempt_count == 2


def test_failed_attempts_are_kept_for_display(engine: Engine):
    answer = pipeline(
        engine,
        ["SELECT ghost FROM traders", "SELECT trader_name FROM traders LIMIT 5"],
    ).answer("?")
    assert answer.attempts[0].sql == "SELECT ghost FROM traders"
    assert not answer.attempts[0].ok
    assert answer.attempts[1].ok


def test_repair_prompt_contains_the_rejected_sql_and_the_reason(engine: Engine):
    """The model needs both; giving it only the error makes it repeat itself."""
    from sql_engine.llm import build_repair_message

    verdict = Verifier(engine).verify("SELECT lifetime_value FROM traders")
    message = build_repair_message("who?", engine.schema_text(), verdict)
    assert "SELECT lifetime_value FROM traders" in message
    assert "lifetime_value" in message
    assert "trader_name" in message  # the schema is repeated so it can be re-read


def test_answer_executes_the_verified_plan_not_the_text(engine: Engine):
    """Re-compiling the SQL string would let the executed query drift from the checked one."""
    answer = pipeline(engine, ["SELECT COUNT(*) AS n FROM trades"]).answer("?")
    assert answer.ok
    assert answer.compiled is answer.attempts[-1].verification.compiled


def test_limit_is_injected_into_the_executed_query(engine: Engine):
    answer = TextToSQL(
        engine,
        StaticClient(["SELECT trade_id FROM trades"]),
        verifier=Verifier(engine, default_limit=5),
    ).answer("?")
    assert answer.ok
    assert len(answer.result.rows) == 5
    assert "LIMIT 5" in answer.sql


# --------------------------------------------------------------------------------------
# Replay client
# --------------------------------------------------------------------------------------


def test_replay_client_serves_recorded_responses():
    client = ReplayClient({"k/attempt-0": "SELECT 1"})
    assert client.complete(Prompt("s", "u", "k/attempt-0")) == "SELECT 1"
    assert client.calls == ["k/attempt-0"]


def test_replay_client_fails_loudly_on_a_missing_key():
    """A silent empty response would be scored as a model error, corrupting the eval."""
    client = ReplayClient({})
    with pytest.raises(FixtureMissingError):
        client.complete(Prompt("s", "u", "missing"))


def test_reference_fixtures_load_and_cover_the_golden_set():
    """The committed evaluation must be runnable with no API key, on any machine."""
    client = ReplayClient.from_file(FIXTURES / "reference.json")
    golden = (
        Path(__file__).resolve().parent.parent / "evaluation" / "golden_set.jsonl"
    ).read_text(encoding="utf-8")
    ids = [json.loads(line)["id"] for line in golden.splitlines() if line.strip()]
    assert len(ids) >= 40, "the golden set must have at least 40 pairs"
    for question_id in ids:
        assert f"{question_id}/attempt-0" in client


def test_reference_fixtures_declare_their_provenance():
    """The fixtures are hand-authored, and the artefact has to say so."""
    payload = json.loads((FIXTURES / "reference.json").read_text(encoding="utf-8"))
    assert "hand-authored" in payload["recorded_from"]
    assert "NOT recorded from a live model" in payload["provenance"]


def test_groq_client_reports_a_missing_key_without_leaking_anything():
    from sql_engine.llm import GroqClient, LLMError

    client = GroqClient(api_key=None)
    if client.available:  # a developer machine may legitimately have one set
        pytest.skip("GROQ_API_KEY is set in this environment")
    with pytest.raises(LLMError) as error:
        client.complete(Prompt("s", "u", "k"))
    assert "GROQ_API_KEY" in str(error.value)
    assert "Bearer" not in str(error.value)
