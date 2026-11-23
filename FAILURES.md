# FAILURES

Bugs hit while building sql-engine, written down as they happened rather than
reconstructed afterwards. Each entry is a symptom, a cause, and the number if there was
one. Most were caught by the SQLite differential oracle rather than by reading the code,
which is the strongest argument for having the oracle at all.

---

## 1. `WHERE desk IS NULL` returned zero rows over CSV

**Symptom.** `SELECT trader_name, desk FROM traders WHERE desk IS NULL` returned 0 rows
from sql-engine and 5 from SQLite over the same file.

**Cause.** `pyarrow.csv.ConvertOptions` has `strings_can_be_null=False` by default. An
empty field in a TEXT column was read as the empty string, not as NULL. The catalog's
own CSV inference *did* treat `""` as NULL, so the catalog reported a null count while
the reader produced none — the two halves of the system disagreed about the same file.

**Fix.** `strings_can_be_null=True` on the reader, in `sql_engine/executor/operators.py`.

**Lesson worth keeping.** The oracle caught this on the first run of the twelfth query I
tried. A unit test written against my own reader would have asserted whatever my reader
did and passed forever.

---

## 2. LEFT JOIN crashed when the build side was filtered to nothing

**Symptom.**

```
SELECT t.trader_name, tr.trade_id
FROM traders t LEFT JOIN trades tr
  ON t.trader_id = tr.trader_id AND tr.price > 5000
```

raised `ExecutionError: column tr.trader_id is not present in the current batch;
available: ` (an empty column list).

**Cause.** Two independent problems stacked.

1. `HashJoinOperator.execute` evaluated the build-side key expressions *before* checking
   whether the build side had any rows. With no rows there were no columns to evaluate
   against.
2. Deeper: operators dropped empty batches entirely (`if batch.num_rows: yield`), so a
   scan whose pushed predicate matched nothing emitted *no batches at all* — and a batch
   is where the column names live. Downstream operators lost the schema, not just the
   rows. A LEFT join specifically needs that schema, because it has to pad unmatched
   rows with a NULL for every right-hand column.

`tr.price > 5000` matches nothing because the generated prices top out near 4000, so
this is the exact case: an outer join whose inner side is empty.

**Fix.** Operators now propagate an empty-but-typed batch when they would otherwise
produce nothing, so column names and dtypes survive an empty result. The build-side
emptiness check moved ahead of key evaluation.

**Lesson worth keeping.** "No rows" and "no schema" are different things, and an engine
that conflates them works fine until the first empty intermediate result. The row count
is data; the schema is metadata and must survive independently.

---

## 3. `_and` under three-valued logic was written wrong on the first pass

**Symptom.** Caught by reading rather than by a test — the first version of the
three-valued `AND` computed

```python
Column(left & right & ~known_false | (left & right), valid)
```

which is `left & right` with extra steps, dressed up to look like it handled the NULL
case.

**Cause.** I was reasoning about the *value* when the interesting part is the *validity
mask*. In `FALSE AND NULL`, the value under the NULL position is arbitrary, but the mask
must be True and the value must be False — and `left & right` already produces False
there because the known-FALSE side is False. The special-casing was noise.

**Fix.** `value = left & right`; `valid = (left.valid & right.valid) | known_false`, with
a comment recording why no special case is needed. `sql_engine/executor/expressions.py`.

---

## 4. Integer division disagreed with SQLite on negative operands

**Symptom.** Anticipated rather than observed, and then confirmed: `-5 / 2` is `-2` in
SQLite (C truncation toward zero) and `-3` in Python and NumPy (floor division).
`-5 % 2` is `-1` in SQLite and `1` in Python.

**Cause.** Python's `//` and `%` follow the mathematician's convention (floor, sign
follows divisor); C and therefore SQLite follow truncation, sign follows dividend.

**Fix.** `np.trunc(a / b)` for integer division and `np.fmod` for the remainder, in
`sql_engine/executor/expressions.py`. Division by zero yields NULL rather than raising,
also matching SQLite.

**Why this one matters.** It only shows up on negative numbers. The demo dataset has
negative `quantity` values specifically so that an ordinary query hits it — a dataset of
all-positive integers would have let this ship.

---

## 5. A SUM over 200,000 floats disagreed with SQLite in the 14th significant digit

**Symptom.** `SELECT SUM(price) FROM trades` gave `401348455.3286044` from sql-engine and
`401348455.3286` from SQLite. The differential test reported a mismatch.

**Cause, part one — the numbers.** Neither engine is wrong. NumPy's `sum` uses pairwise
summation, which halves the depth of the accumulation tree and therefore the accumulated
rounding error; SQLite adds left to right in one accumulator. Over 200,000 float64
values the two answers differ by about 4e-6 absolute, which is 1.1e-14 relative — below
float64's ~2.2e-16 epsilon times the number of operations. Both are correct answers to a
question that does not have a single correct answer in floating point.

**Cause, part two — the comparator, which *was* wrong.** The tolerance check
(`math.isclose(rel_tol=1e-9)`) would have accepted this comfortably. It never ran,
because unordered results are compared as a multiset via a `Counter`, and the hashable
key rounded floats with `round(value, 9)` — nine *decimal places*. A value in the
hundreds of millions has no decimal places left to round; the two numbers went into
different buckets and the tolerant comparison was never consulted.

**Fix.** Two changes in `sql_engine/sqlite_oracle.py`:

1. The key now buckets by *significant* digits (`f"{value:.12g}"`), which is
   scale-invariant.
2. More importantly, a Counter mismatch is no longer a verdict. The rows the fast path
   could not pair are rechecked with the tolerant pairwise comparison. The exact-key
   path is now an optimisation that cannot produce a false failure, only a slower path.

**Lesson worth keeping.** The bug was in the *test harness*, not the engine — and a
harness bug that produces a false failure is the good kind. The dangerous version is the
same mistake in the direction that hides a difference. That is the argument for making
the fast path a filter over an exact check rather than a substitute for one.

---

## 6. `WHERE desk = NULL` crashed the engine instead of returning nothing

**Symptom.** The evaluation harness died on `golden-008` with
`ExecutionError: cannot compare a TEXT column with a numeric value`. The query was
`SELECT COUNT(*) FROM traders WHERE desk = NULL` — a classic beginner mistake that
SQLite answers with `0`, not an error.

**Cause.** A `NULL` literal has no type, so `Column.constant(None)` allocates a float64
array with an all-false validity mask; the dtype is a placeholder that nothing should
ever read. The comparison kernel checked dtypes *before* it checked validity, saw
`object` on the left and `float64` on the right, and hit the guard that refuses
TEXT-versus-numeric comparisons — a guard added for a completely different reason (see
`_compare` in `sql_engine/executor/expressions.py`).

**Fix.** Check the validity mask first. If no row has a valid value on both sides, the
answer is UNKNOWN everywhere and can be returned without looking at the values at all.
Cheap, correct, and it makes the dtype guard unreachable for the one case where the
dtype is meaningless.

**A second fix in the same place.** The pipeline now catches `SQLEngineError` around
execution of a *verified* query. Verification is static and a type error can still
surface at runtime; letting it propagate would have broken the pipeline's contract that
`answer()` always returns an `Answer`, and would have let a runtime crash be scored as a
verified success. This is exactly the "valid but wrong" boundary the evaluation reports.

**Why this one is a good bug to have found.** The fixture that triggered it was written
deliberately as a *semantic* error the verifier is supposed to miss. It found a
*structural* bug instead. Adversarial fixtures earn their keep.

---

## 7. `HAVING n > 1` could not see its own alias

**Symptom.** `SELECT side, COUNT(*) AS n FROM trades GROUP BY side HAVING n > 1` failed
with `UnknownColumnError: column 'n' does not exist in any table in this query` — an
error about a column the user never claimed existed. Found by a unit test written while
covering the binder, not by the differential suite, because SQLite accepts the query and
I had simply never written it down.

**Cause.** ORDER BY and HAVING may name a select-list alias, which is not a column of any
table, so aliases are substituted before name resolution. The substitution checked only
the *root* node of the expression:

```python
if isinstance(expr, ColumnRef) and expr.name in aliases:
    return aliases[expr.name]
return expr
```

`ORDER BY n` happens to put the alias at the root, so it worked and looked correct.
`HAVING n > 1` puts it one level down inside a `BinaryOp`, where nothing looked.

**Fix.** Added a general `transform(expr, rewrite)` to `sql_engine/ast_nodes.py` — the
write-side companion to the existing `walk` — and rebuilt the substitution on top of it.

**Lesson worth keeping.** There were already two tree traversals in the codebase
(`walk`, and the binder's own recursive `_bind_expression`) and I wrote a third by hand
for this one job, badly. The fix is not "be more careful", it is "have one traversal
primitive so the careless version is not even convenient to write".

---

## 8. A Spider coverage number I nearly published wrong

**Symptom.** Not a crash — a *measurement* that was about to go into the README as fact.
Measuring how much of the Spider dev set this dialect can express gave 84.6% parsing but
only 56.9% binding, with 186 queries failing as "unknown column". A 28-point gap between
parsing and binding did not make sense for gold SQL written against its own schema, so I
looked at the failures instead of writing them up.

**Cause.** Spider's gold SQL writes string literals in **double quotes**:

```sql
SELECT Country FROM AIRLINES WHERE Airline = "JetBlue Airways"
```

In standard SQL, and in sql-engine, double quotes delimit an *identifier*. So this query
asks for a column named `JetBlue Airways`, and the binder correctly reports that no such
column exists. SQLite accepts it through a documented compatibility fallback: if a
double-quoted token does not resolve to a column, it silently reinterprets it as a string
literal. SQLite's own documentation calls this a misfeature and ships a flag to disable
it.

So sql-engine was right and the measurement was wrong — it was counting "follows the SQL
standard" as "cannot express this query".

**Fix.** `evaluation/spider_coverage.py` now reports coverage both ways: as written
(56.9%) and after rewriting double-quoted literals into single quotes (74.9%). Both
numbers are in the report with the reason stated, rather than picking whichever is more
flattering.

**What it uncovered underneath.** With the quoting difference separated out, the
remaining 101 bind failures are a single real cause: sql-engine refuses
`SELECT name, COUNT(*) ... GROUP BY id` where SQLite picks an arbitrary row for `name`.
That is a deliberate strictness choice, and now I know exactly what it costs — 9.8% of
the Spider dev set.

**Lesson worth keeping.** The instinct that saved this was distrusting a number I liked
the shape of. A 28-point drop was implausible, so it got investigated instead of
published. Numbers that confirm what you already believe deserve the same scrutiny as
ones that do not; this one did not confirm anything flattering, which is probably the
only reason I checked.

---

## 9. A 403 that was not an auth failure

**Symptom.** With a valid `GROQ_API_KEY`, every request returned `HTTP 403: Forbidden`.
The obvious reading is a bad or revoked key.

**Cause.** The body said `error code: 1010`. That is a *Cloudflare* code, not a Groq one
— a browser-signature block triggered by Python's default `urllib`/`requests`
User-Agent. The key was fine the whole time. Sending an ordinary browser UA returned
`200` on the first try.

**Fix.** `_BROWSER_UA` in `sql_engine/llm/client.py`, with a comment explaining why a
seemingly cosmetic header is load-bearing.

**Lesson.** Read the body before concluding anything from a status code. A 403 from a
CDN and a 403 from an API mean completely different things, and only one of them is
about your credentials.

## 10. The rate limit was on tokens, and the failure was silent

**Symptom.** Recording fixtures for 45 questions produced 21 responses. Questions 19
onward mostly reported `unrecovered attempts=0`, which reads as "the model failed to
produce anything usable" — i.e. a *model* problem.

**Cause.** Rate limiting, but not the kind I assumed. The response headers showed
`x-ratelimit-remaining-requests: 972` of 1000 — plenty — while
`x-ratelimit-limit-tokens` was **8,000** per window. Each prompt carries the full schema,
so the token budget ran out roughly 18 questions in while the request budget was barely
touched. The client had no retry, so 429s surfaced as empty results.

**Why this one mattered more than it looks.** Had I not checked, the evaluation would
have reported accuracy over the 21 questions that happened to get through — a number
computed from a truncated sample, presented as if it covered 45. Silent partial failure
producing a plausible-looking number is the most dangerous shape a bug can have.

**Fix.** Retry with backoff honouring `Retry-After`, falling back to
`x-ratelimit-reset-tokens` (which is the one that actually matters here) and then to
exponential backoff. All 45 recorded on the next run.

## 11. The evaluation was measuring itself

**Symptom.** Nothing looked wrong. The reported figures were 57.8% unverified → 84.4%
verified, a clean +26.7 point lift attributed to the verification layer.

**Cause.** No API key existed when the harness was built, so the candidate SQL was
hand-authored — and, being written to exercise the verifier, it contained precisely the
errors the verifier detects. The measurement was circular: the fixtures had been
authored with the answer in mind.

**What the real number turned out to be.** Against recorded `openai/gpt-oss-120b`
responses: **42.2% unverified, 42.2% verified, +0.0 lift.** Zero of 45 answers refused.
The model makes no structural errors, so there was nothing to catch.

**Fix.** Real fixtures committed; the circular ones removed from every headline; both
README and DEFENCE rewritten to lead with the zero and explain it. The verifier is now
measured where it is actually meaningful — 48 labelled adversarial cases —
**44/48 correct verdicts, 90.5% recall, 6/6 legitimate queries preserved.**

**Lesson.** A number that flatters the thing you built deserves more scrutiny than one
that does not. This one survived until an API key made it checkable.

## 12. A comment is a silent deletion

**Symptom.** Two adversarial cases passed verification untouched:

```sql
SELECT ticker FROM instruments -- AND is_active = 1
SELECT ticker FROM instruments WHERE ticker = 'A' OR '1' = '1' --
```

**Cause.** Both are legal `SELECT` statements. They parse, they bind, every structural
check passes. The first is the dangerous one: the comment deletes a filter, so the query
returns *more* rows than its own text claims. Nothing syntactic is wrong with it.

**The disagreement worth recording.** An existing test asserted the opposite — that a
payload inside a comment is *neutralised* and refusing it would be a false positive. That
is correct about **safety**: nothing dangerous executes. It misses **correctness**: the
query no longer means what it says. Both positions are defensible; they are about
different threat models.

**Fix.** Reject SQL containing a comment, as a matter of provenance rather than syntax —
generated SQL has no legitimate reason to carry one, and the cost of refusing is a single
regeneration. The check tracks quote state, so `WHERE note = 'a--b'` stays allowed; a
naive `'--' in sql` would have broken that. Adversarial verdicts went 42/48 → **44/48**,
recall 85.7% → **90.5%**, with no new false positives. The superseded test was rewritten
rather than deleted, and carries the argument for both sides.
