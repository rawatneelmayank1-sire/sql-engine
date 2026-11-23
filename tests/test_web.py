"""Web API tests.

The UI is a thin layer over the engine, so these do not re-test SQL semantics. They test
the two things the layer itself can get wrong: reporting a refusal as a success, and
letting a refused query reach the executor.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sql_engine.web.app import create_app


@pytest.fixture(scope="module")
def client(warehouse) -> TestClient:
    return TestClient(create_app(warehouse))


def test_index_serves_the_page(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "SQL Engine" in response.text


def test_schema_endpoint_describes_the_catalog(client: TestClient):
    payload = client.get("/api/schema").json()
    names = {table["name"] for table in payload["tables"]}
    assert names == {"trades", "instruments", "traders", "daily_prices"}
    assert payload["examples"]
    assert payload["attack_examples"]


def test_verify_accepts_a_good_query(client: TestClient):
    payload = client.post("/api/verify", json={"sql": "SELECT COUNT(*) FROM trades"}).json()
    assert payload["ok"] is True
    assert payload["result"]["rows"] == [[200_000]]
    assert [stage["status"] for stage in payload["stages"]] == ["passed"] * 6


@pytest.mark.parametrize(
    "sql,halt_stage",
    [
        ("DROP TABLE trades", "parse"),
        ("DELETE FROM trades", "parse"),
        ("SELECT * FROM trades; DROP TABLE trades;", "parse"),
        ("SELECT lifetime_value FROM traders", "bind"),
        ("SELECT * FROM customers", "bind"),
        ("SELECT t.trade_id FROM trades t INNER JOIN instruments i ON 1 = 1", "verify"),
        # The comment check runs on the raw text before parsing. Unmapped, it used to be
        # reported as six passed stages next to ok=False.
        ("SELECT trade_id FROM trades WHERE trader_id = 7 -- AND side = 'BUY'", "tokenise"),
    ],
)
def test_verify_reports_where_the_pipeline_stopped(
    client: TestClient, sql: str, halt_stage: str
):
    payload = client.post("/api/verify", json={"sql": sql}).json()
    assert payload["ok"] is False
    stages = {stage["name"]: stage["status"] for stage in payload["stages"]}
    assert stages[halt_stage] == "refused"
    # Everything downstream of the halt is skipped, not failed: the pipeline stopped.
    names = [stage["name"] for stage in payload["stages"]]
    for name in names[names.index(halt_stage) + 1 :]:
        assert stages[name] == "skipped"
    assert payload["result"]["rows"] == []


def test_every_failure_category_halts_the_rail_at_a_stage():
    """A category with no stage would draw a refused query as six passed stages."""
    from sql_engine.llm import FailureCategory
    from sql_engine.web.app import CATEGORY_STAGE, STAGES

    for category in FailureCategory:
        assert CATEGORY_STAGE.get(category) in STAGES, category


def test_ask_runs_the_full_pipeline_from_fixtures(client: TestClient):
    payload = client.post("/api/ask", json={"question": "How many trades are there in total?"}).json()
    assert payload["ok"] is True
    assert payload["result"]["rows"] == [[200_000]]
    assert payload["plan"]["before"] and payload["plan"]["after"]


def test_ask_exposes_the_repair_history(client: TestClient):
    """The refused attempt is the interesting one, so the API must not discard it."""
    payload = client.post("/api/ask", json={"question": "What is the average trade price?"}).json()
    assert payload["ok"] is True
    assert payload["repair_count"] == 1
    assert payload["attempts"][0]["ok"] is False
    assert "ROUND" in payload["attempts"][0]["sql"]


def test_ask_says_so_when_no_model_is_configured(client: TestClient, monkeypatch):
    """An unknown question must produce a plain refusal, never an invented answer."""
    payload = client.post(
        "/api/ask", json={"question": "something nobody recorded an answer for"}
    ).json()
    assert payload["ok"] is False
    assert payload["no_model"] is True


def test_limit_injection_is_reported_as_a_repair(client: TestClient):
    payload = client.post("/api/verify", json={"sql": "SELECT trade_id FROM trades"}).json()
    assert payload["ok"] is True
    assert payload["repairs"][0]["kind"] == "limit-injected"
    assert "LIMIT 200" in payload["sql"]


def test_oversized_input_is_rejected_by_validation(client: TestClient):
    response = client.post("/api/verify", json={"sql": "x" * 9000})
    assert response.status_code == 422
