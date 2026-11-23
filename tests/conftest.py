"""Shared fixtures.

The demo dataset is generated once per session and cached on disk, and the SQLite oracle
is built once per session too. Building either per test would dominate the run time and
would discourage adding cases to the differential corpus -- which is exactly the corpus
that needs to keep growing.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.generate import generate  # noqa: E402
from sql_engine import Catalog, Engine, Optimiser  # noqa: E402
from sql_engine.sqlite_oracle import build_sqlite  # noqa: E402

WAREHOUSE = ROOT / "data" / "warehouse"


@pytest.fixture(scope="session")
def warehouse() -> Path:
    """The demo dataset directory, generated if it is not already there."""
    generate(WAREHOUSE)
    return WAREHOUSE


@pytest.fixture(scope="session")
def catalog(warehouse: Path) -> Catalog:
    return Catalog.from_directory(warehouse)


@pytest.fixture(scope="session")
def engine(catalog: Catalog) -> Engine:
    return Engine(catalog)


@pytest.fixture(scope="session")
def unoptimised_engine(catalog: Catalog) -> Engine:
    """The same engine with both rewrite rules switched off.

    Used to prove the optimiser is semantics-preserving: every corpus query must return
    identical rows with the rules on and off. An optimiser that changes an answer is not
    an optimiser.
    """
    return Engine(
        catalog,
        optimiser=Optimiser(predicate_pushdown=False, projection_pruning=False),
    )


@pytest.fixture(scope="session")
def sqlite_connection(catalog: Catalog) -> sqlite3.Connection:
    connection = build_sqlite(catalog)
    yield connection
    connection.close()
