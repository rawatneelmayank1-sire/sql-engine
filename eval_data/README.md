# SQL Engine evaluation datasets

Three datasets for evaluating the sql-engine text-to-SQL engine. Every count below was
counted from the files on disk, not estimated. Total size on disk: **4.4 MB**.

| Set | Count | Expected results? | Provenance |
|---|---|---|---|
| `golden/` | **75** question/SQL pairs | yes — all 75 executed and verified | hand-written for this project |
| `adversarial/` | **48** cases (42 reject / 6 allow) | n/a — labelled expectations | hand-written for this project |
| `spider/` | **1034** dev questions, **20** dev schemas | **no** | Spider benchmark, CC BY-SA 4.0 |

Quick start:

```python
from eval_data.loader import load_golden, load_adversarial, load_spider, golden_db_path

for case in load_golden():
    rows = my_engine.run(case.sql)
    assert len(rows) == case.expected_row_count
```

```bash
python loader.py    # print dataset statistics
python verify.py    # re-execute every gold SQL and check the recorded hashes (exit 0 = OK)
```

---

## 1. `golden/` — the primary set

### Database

A synthetic market / instrument reference database, generated deterministically by
`generate_db.py` with **fixed seed 20260909**. Re-running reproduces identical data;
re-running when the outputs already exist is a no-op.

| Table | Rows | Notes |
|---|---:|---|
| `exchanges` | 8 | |
| `sectors` | 11 | `description` nullable; *Communication Services* deliberately has zero instruments |
| `instruments` | 120 | `sector_id` nullable (9 unclassified), `shares_outstanding` nullable (8 unknown) |
| `daily_bars` | 14,400 | 120 instruments x 120 trading days (2024-01-02 .. 2024-06-17) |
| `analysts` | 40 | `country` nullable (4 unknown) |
| `ratings` | 620 | `price_target` nullable (80 unpublished) |
| **total** | **15,199** | |

Relationships: `exchanges 1-* instruments *-1 sectors`, `instruments 1-* daily_bars`,
`instruments 1-* ratings *-1 analysts`.

The nullable columns and the empty sector are load-bearing: they make `IS NULL` and
`LEFT JOIN ... WHERE right.key IS NULL` return non-empty, meaningful answers instead of
trivially empty ones.

Files in `golden/db/`:
- `<table>.csv` and `<table>.parquet` for each of the 6 tables
- `schema.sql` — the CREATE TABLE DDL, the schema of record
- `schema.json` — machine-readable tables / columns / types / nullability / PK / FK / row counts
- `market.sqlite` — the built database, used for verification and returned by `golden_db_path()`

### Questions — `golden/questions.jsonl`

One JSON object per line:

```json
{"id": "g001", "question": "...", "sql": "...", "difficulty": "easy",
 "features": ["select"], "expected_row_count": 8,
 "expected_columns": ["code"],
 "expected_hash": "<sha256 of the sorted result set>",
 "expected_ordered_hash": "<sha256 of the result set in returned order>"}
```

- `expected_hash` is order-**insensitive** (rows canonicalised, then sorted) — use it to
  compare correctness regardless of row order.
- `expected_ordered_hash` preserves the returned order — this is what `ORDER BY` /
  `LIMIT` / `OFFSET` cases are really asserting.
- Floats are canonicalised to 6 decimal places and `NULL` to a sentinel before hashing.

**75 authored, 75 verified.** Every gold SQL was executed against `market.sqlite`;
0 failed, and 0 returned an empty result set (an empty result would make a case
vacuously passable). `verify.py` re-executes all 75 independently of the script that
wrote the hashes.

Difficulty: **easy 21 / medium 33 / hard 21**.

Feature coverage (22 distinct tags, counts across the 75 cases):

| | | | |
|---|---|---|---|
| `select` 75 | `where` 32 | `group_by` 23 | `count` 16 |
| `join` 12 | `order_by` 12 | `having` 8 | `is_null` 7 |
| `avg` 6 | `left_join` 6 | `limit` 6 | `and` 5 |
| `like` 5 | `not` 5 | `sum` 5 | `in` 4 |
| `max` 4 | `between` 3 | `distinct` 3 | `min` 2 |
| `or` 2 | `offset` 1 | | |

All required constructs are covered: simple projection; `WHERE` with
AND / OR / NOT / IN / BETWEEN / LIKE / IS NULL; `GROUP BY` with each of
COUNT / SUM / AVG / MIN / MAX; `HAVING`; `ORDER BY`; `LIMIT` and `LIMIT ... OFFSET`;
and both INNER and LEFT equi-joins.

### Supported-subset constraint

Every gold SQL stays inside the subset sql-engine supports: **SELECT only, no subqueries,
no CTEs, no window functions, no set operations (UNION/INTERSECT/EXCEPT), no self-joins.**
`build_questions.py` enforces this with `check_subset()`, so an out-of-subset query cannot
quietly enter the file.

---

## 2. `adversarial/` — the verification-layer set

`adversarial/cases.jsonl`, one JSON object per line:

```json
{"id": "a001", "sql": "SELECT ticker, market_cap FROM instruments",
 "expect": "reject", "reason": "invented_column",
 "note": "...", "sqlite_parses": false, "sqlite_error": "OperationalError: ..."}
```

**48 cases: 42 `reject`, 6 `allow`.**

| `reason` | Count | What it tests |
|---|---:|---|
| `not_select` | 8 | DROP, DELETE, UPDATE, INSERT, ALTER, ATTACH, PRAGMA, CREATE |
| `unparseable` | 8 | malformed SQL, prose instead of SQL, markdown-fenced SQL, empty string |
| `invented_column` | 7 | plausible column names absent from the schema |
| `injection_attempt` | 6 | stacked statements, comment-terminated payloads, tautologies |
| `valid` | 6 | must **pass** — guards against a layer that just rejects everything |
| `invented_table` | 5 | plausible table names absent from the schema |
| `cartesian` | 4 | joins with no equi-predicate (comma join, `ON 1=1`, CROSS JOIN) |
| `unbounded` | 4 | full scans of the 14,400-row table with no LIMIT |

### The `sqlite_parses` field — the important one

`build_adversarial.py` runs `EXPLAIN <sql>` against a **read-only** connection to the
golden database. EXPLAIN parses and resolves every table and column name without
executing anything, so it is safe even for the DROP/ATTACH cases.

- **23 of 48** cases fail to parse — SQLite itself rejects them. These are catchable by
  parsing and name resolution alone.
- **25 of 48** parse cleanly. **19 of those are cases that must still be rejected**
  (`not_select`, `cartesian`, `unbounded`, and the comment-terminator injection `a037`).
  They are perfectly legal SQL and will not raise on execution, so **sql-engine's own
  static analysis and policy are the only thing that can catch them.** That is the
  point of this set. `AdversarialCase.needs_static_analysis` exposes this.

The parse check also validates the labels: every `invented_column` / `invented_table` /
`unparseable` case is asserted to genuinely fail against the real schema, and every
`valid` case is asserted to genuinely parse. The build script reports a mismatch rather
than writing a wrong label.

Case `a037` (`SELECT ticker FROM instruments -- AND is_active = 1`) is worth singling out:
it parses, executes, and returns *more* rows than intended. Nothing errors.

---

## 3. `spider/` — the public benchmark

**Obtained.** 1034 dev questions with gold SQL, plus 166 database schemas (20 of which
cover the dev split). Full sourcing, sha256 checksums, licence and citation are in
[`spider/PROVENANCE.md`](spider/PROVENANCE.md).

- Questions + gold SQL: HuggingFace `xlangai/spider`, `spider/validation-*.parquet`,
  repo sha `0c35091`.
- Schemas: the official Yale LILY repo `taoyds/spider`, `evaluation_examples/examples/tables.json`.
- Licence: **CC BY-SA 4.0** (attribution and share-alike apply to redistribution).

### Two caveats that must not be lost

1. **The Spider SQLite database files are NOT included.** The official release ships them
   only in a ~1 GB Google Drive archive, which is not retrievable non-interactively and
   would break the size budget for this directory. **Execution accuracy against Spider
   cannot be computed from here** — `load_spider()` returns cases with
   `expected_hash is None` and `is_verified == False` for exactly this reason. Do not
   report Spider execution numbers on the basis of these files.
2. **Spider gold SQL routinely falls outside the sql-engine supported subset** (subqueries,
   UNION/INTERSECT/EXCEPT, nested aggregates, self-joins). Spider is a coverage and
   robustness set, not a target the engine is expected to satisfy in full. Filter before use.

The hand-written `golden/` set is the one with verified, executable expected results.

---

## 4. `loader.py`

Standard library only — no third-party imports — so the engine can depend on it without
pulling pandas or pyarrow into its runtime. Fully type-hinted.

```python
def load_golden() -> list[GoldenCase]: ...
def load_adversarial() -> list[AdversarialCase]: ...
def load_spider() -> list[GoldenCase] | None: ...   # None if unavailable
def golden_db_path() -> Path: ...                   # raises FileNotFoundError with a fix hint
def golden_schema() -> dict: ...
def spider_schemas() -> list[dict] | None: ...
```

`GoldenCase` exposes `is_verified` and `has_feature(name)`; `AdversarialCase` exposes
`should_reject` and `needs_static_analysis`. Both are frozen dataclasses.

## Regenerating

Each step is resumable and skips work already on disk.

```bash
python spider/_fetch_spider.py        # re-download Spider (skips files already present)
python golden/generate_db.py          # regenerate the database (no-op if outputs exist)
python golden/build_questions.py      # re-verify all gold SQL and rewrite questions.jsonl
python adversarial/build_adversarial.py
python verify.py                      # independent integrity check
```

Note: `generate_db.py` coerces every value through `_to_sqlite()` before insertion. This
is not cosmetic — pandas nullable `Int64` values arrive as `numpy.int64`, which sqlite3
has no adapter for and silently stores as an 8-byte **BLOB**, making every join on the
column match zero rows while the INSERT still reports success. The generator asserts that
no column is stored as a BLOB and that no foreign key is left unresolvable.
