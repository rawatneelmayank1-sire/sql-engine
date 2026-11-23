"""Independent integrity check for the eval datasets. Run: ``python verify.py``

Re-executes every golden gold SQL against the golden database and compares the
result against the hash recorded in questions.jsonl. This is deliberately
separate from build_questions.py (which *wrote* those hashes) so that a passing
run means the recorded expectations are genuinely reproducible.

Exits non-zero if anything fails.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys

from loader import (
    golden_db_path,
    load_adversarial,
    load_golden,
    load_spider,
)


def canonical(value: object) -> str:
    if value is None:
        return "\x00NULL"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def hash_rows(rows: list[tuple], sort: bool) -> str:
    encoded = [[canonical(v) for v in row] for row in rows]
    if sort:
        encoded.sort()
    return hashlib.sha256(
        json.dumps(encoded, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> int:
    failures: list[str] = []

    # ---- golden -----------------------------------------------------------
    golden = load_golden()
    conn = sqlite3.connect(f"file:{golden_db_path().as_posix()}?mode=ro", uri=True)
    executed = 0
    for case in golden:
        try:
            cursor = conn.execute(case.sql)
            rows = cursor.fetchall()
            columns = tuple(d[0] for d in cursor.description)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{case.id}: did not execute - {type(exc).__name__}: {exc}")
            continue
        executed += 1
        if len(rows) != case.expected_row_count:
            failures.append(
                f"{case.id}: row count {len(rows)} != recorded {case.expected_row_count}"
            )
        if hash_rows(rows, sort=True) != case.expected_hash:
            failures.append(f"{case.id}: result-set hash mismatch")
        if hash_rows(rows, sort=False) != case.expected_ordered_hash:
            failures.append(f"{case.id}: ordered result hash mismatch")
        if columns != case.expected_columns:
            failures.append(f"{case.id}: columns {columns} != {case.expected_columns}")
    conn.close()
    print(f"golden:      {executed}/{len(golden)} executed and re-verified against SQLite")

    # ---- adversarial ------------------------------------------------------
    adversarial = load_adversarial()
    ids = [c.id for c in adversarial]
    if len(ids) != len(set(ids)):
        failures.append("adversarial: duplicate ids")
    for case in adversarial:
        if case.expect not in {"reject", "allow"}:
            failures.append(f"{case.id}: bad expect value {case.expect!r}")
        if case.reason == "valid" and case.expect != "allow":
            failures.append(f"{case.id}: reason 'valid' must expect 'allow'")
        if case.reason != "valid" and case.expect != "reject":
            failures.append(f"{case.id}: reason {case.reason!r} must expect 'reject'")
    print(f"adversarial: {len(adversarial)} cases, labels consistent"
          if not failures else f"adversarial: {len(adversarial)} cases")

    # ---- spider -----------------------------------------------------------
    spider = load_spider()
    print(f"spider:      {len(spider) if spider else 0} dev cases"
          f"{'' if spider else ' (not available)'}")

    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for line in failures:
            print("  ", line)
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
