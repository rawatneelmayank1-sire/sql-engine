"""Spider coverage tests.

The Spider dev files are optional -- a clone without them must still pass -- so these
skip rather than fail when the data is absent. What they do assert, when it is present,
is that the measurement is sound: no query is counted twice, every parsed query is
accounted for, and the double-quote normalisation that the headline number depends on
actually does what it claims.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPIDER = ROOT / "eval_data" / "spider"

pytestmark = pytest.mark.skipif(
    not (SPIDER / "dev.jsonl").exists() or not (SPIDER / "dev_schemas.json").exists(),
    reason="Spider dev files are not present in this clone",
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "spider_coverage", ROOT / "evaluation" / "spider_coverage.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["spider_coverage"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def coverage():
    return _module().analyse()


def test_the_dev_split_is_the_published_size(coverage):
    """1,034 questions over 20 databases is the canonical Spider dev split."""
    assert coverage["total"] == 1034
    assert coverage["databases"] == 20


def test_every_query_is_accounted_for(coverage):
    """Parsed plus each parse-failure reason must sum to the whole split."""
    assert coverage["parsed"] + sum(coverage["parse_reasons"].values()) == coverage["total"]


def test_every_parsed_query_is_accounted_for(coverage):
    """Bound plus each bind-failure reason must sum to the parsed count."""
    assert coverage["bound"] + sum(coverage["bind_reasons"].values()) == coverage["parsed"]


def test_normalisation_only_ever_helps(coverage):
    """Rewriting double-quoted literals cannot make a query stop binding."""
    assert coverage["bound_normalised"] >= coverage["bound"]
    assert coverage["bound_normalised"] - coverage["bound"] == coverage["dqs_only"]


def test_double_quote_normalisation_does_what_it_says():
    normalise = _module().normalise_double_quoted_strings
    assert normalise('WHERE a = "x"') == "WHERE a = 'x'"
    assert normalise("WHERE a = 'x'") == "WHERE a = 'x'"
    # An apostrophe inside the rewritten literal must be escaped, not left to break it.
    assert normalise('WHERE a = "it\'s"') == "WHERE a = 'it''s'"


def test_coverage_has_not_silently_collapsed(coverage):
    """A guard against a parser change quietly shrinking the dialect.

    The threshold is well below the measured 84.6%, so ordinary variation will not trip
    it, but removing a feature would.
    """
    assert coverage["parsed"] / coverage["total"] > 0.75
