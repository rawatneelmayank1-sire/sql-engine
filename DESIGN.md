# Design

The design decisions behind sql-engine, the alternatives rejected, and what each one cost.
Every number here comes from a command in this repository; where
I could not measure something I say so.

---

## 1. Why write an engine instead of using SQLite?

If the goal were "run SQL fast", using SQLite would be the correct engineering answer and
writing this would be malpractice. The goal is different: **make destructive SQL
unrepresentable rather than blocked.**

To do that I need to own the grammar. SQLite parses `DROP TABLE trades` perfectly well —
the only thing standing between an LLM and a dropped table is a check I remember to
write, and a check can be bypassed. Owning the parser means:

- **There is no AST node for a destructive statement.** `sql_engine/ast_nodes.py` has
  `SelectStatement` and nothing else. `parse()` returns that type or raises. If it
  returns, the caller is holding a read-only query, and that is a fact about the type
  system rather than about my discipline.
- **The verifier and the executor share one parser.** A verifier built on `sqlparse` or
  `sqlglot` accepts a slightly different language from the one that runs. That gap —
  between "what the checker understood" and "what the engine will do" — is where every
  SQL-injection bug in history lives. Here the gap has no room to exist.
- **The catalog is mine, so "this column exists" is provable.** The binder resolves every
  reference or raises with the list of real columns. That is what makes the repair loop
  effective rather than a retry.

There is a second, quieter reason. An engine forces you to make every decision explicit:
NULL semantics, integer division sign, ordering of NULLs, when a predicate may be moved.
Reading about three-valued logic and implementing it against SQLite as an oracle are
different activities, and only the second one finds out you were wrong.

**The honest limit of the argument:** this buys safety against *destructive* and
*malformed* queries. It buys nothing against a query that is valid and semantically
wrong, and the evaluation quantifies exactly that — 26 of 45.

---

## 2. What did predicate pushdown actually buy?

Measured, `python bench/run_bench.py`, median of 7 runs, same engine with one flag
flipped:

| query | speedup | what changed |
| --- | ---: | --- |
| `count_filtered` | **8.17x** | 9 columns → 1, plus the filter runs inside the scan |
| `join_filter` | **3.97x** | 16 → 6 columns, 2 conjuncts pushed to two different scans |
| `filter_project` | **3.59x** | 9 → 2 columns |
| `distinct` | **1.70x** | 9 → 2 columns; no predicate to push |

Range across all nine benchmark queries: **1.70x to 8.17x**.

The honest attribution: most of that is **projection pruning**, not predicate pushdown.
Pruning changes what the Parquet reader decodes off disk, which is real I/O and real CPU
saved. Pushdown moves a filter that was going to run anyway a few operators earlier — it
saves materialising intermediate batches, which matters, but less. `count_filtered` gets
8x because it benefits from both: one column instead of nine, and 4,956 rows leaving the
scan instead of 200,000.

**What is still on the table:** sql-engine does not use Parquet row-group statistics to
skip row groups entirely. A pushed predicate is applied *after* each batch is decoded.
Pushing the predicate into the reader as an Arrow filter expression would let whole row
groups be skipped without decoding. That is the single largest performance win I have not
taken, and it is in the README's Limitations.

---

## 3. Why vectorised rather than row-at-a-time?

Because in Python the interpreter overhead is not *a* cost, it is essentially *the* cost.

A classic Volcano executor pulls one row per `next()` call. Evaluating `price > 3900` over
200,000 rows that way means 200,000 Python-level function calls, 200,000 attribute
lookups, 200,000 boxed comparisons — roughly 50–100 ns each before any actual work.

sql-engine pulls one *batch* of 4096 rows per `next()`. The same predicate becomes one
NumPy call over a contiguous `float64` array: about 49 batches instead of 200,000 rows, so
Python's per-call overhead is paid roughly 4,000 times less often, and the comparison
itself runs in C at memory bandwidth.

**Why 4096 specifically?** Large enough that the fixed cost of a NumPy call amortises to
nothing; small enough that a batch and its intermediates stay in L2 cache rather than
being streamed through memory twice. It is the number DuckDB, Velox and ClickHouse all
converge on for the same reason. I did not tune it empirically — I tested that results are
*identical* at 1, 3, 7, 64, 4096 and 65536, which is the property that actually matters.

**The visible payoff:** on `group_by_wide` sql-engine is 0.51x of file-backed SQLite — it
beats a mature C engine, because a grouped aggregate over a columnar batch is exactly what
NumPy is good at, and because the scan reads 3 columns where SQLite's row store reads all
9.

**The cost of the choice:** batching makes every operator harder to write correctly.
Blocking operators materialise. Empty batches have to carry schema even with no rows —
that cost me a real bug (`FAILURES.md` #2). And off-by-one errors at chunk boundaries are
invisible at any single batch size, which is why the batch-size-independence test exists.

---

## 4. How do you know the engine is correct?

I do not test it against my own expectations. I test it against SQLite.

Every test query runs through both engines over the same files, and the results must
match. SQLite is the most-tested SQL implementation in existence and it ships in the
standard library, so the oracle costs nothing and runs in CI.

Four layers:

1. **108 hand-written differential queries** (`tests/corpus.py`), grouped by the feature
   they stress and weighted toward the awkward cases: NULL in `IN` lists, `NOT` over
   NULL, NULL group keys, NULL join keys, integer division on negatives, outer joins with
   an empty inner side.
2. **Property-based differential tests** (Hypothesis) that generate random `WHERE` clauses
   out of comparisons, `IN`, `BETWEEN`, `IS NULL`, `AND`, `OR`, `NOT` and check both
   engines agree. Three-valued logic in combination is where hand-written tests run out
   of imagination.
3. **Optimiser equivalence**: the entire corpus runs twice, with the rewrite rules on and
   off, and must produce identical rows. A separate test asserts the rules actually fire,
   so that comparison cannot pass vacuously.
4. **Batch-size independence**: identical results at six different batch sizes.

```
$ python -m pytest
483 passed in 53.76s
```

(Six tests skip on a clone without the optional Spider dev files.)

**What this does *not* prove.** The oracle only covers queries I thought to write plus the
shapes Hypothesis generates. It proves nothing about performance, and nothing about
queries outside the corpus. It also only proves agreement — if SQLite and I are both
wrong about the standard, the test passes.

**The best evidence it works:** `FAILURES.md` has twelve entries, and five of them were
found by the oracle rather than by reading code, including a Parquet/CSV reader
disagreeing with itself about what an empty field means. One more (#8) was a *number*
caught before it was published rather than a bug in the engine.

---

## 5. How was the LLM layer evaluated, and how honest is the number?

Execution accuracy, the way the text-to-SQL literature measures it: run the gold query
and the predicted query, compare result sets. The gold side runs through **SQLite**, not
through sql-engine, so an engine bug affecting both sides identically cannot hide.

Against real recorded responses from `groq:openai/gpt-oss-120b`:

| Regime | Accuracy |
| --- | ---: |
| First answer executed directly | 19/45 = **42.2%** |
| Verified, with the repair loop | 19/45 = **42.2%** |
| Difference | **+0.0 points** |

**Zero of 45 first answers were refused.** The repair loop never fired.

### Why the lift is zero

An earlier version of this document claimed +26.7 points. That number was measured
against fixtures I had written by hand, before I had an API key — and since I wrote them
to exercise the verifier, they contained exactly the errors the verifier catches. It was
circular. When I got a key and recorded real responses, the lift went to zero.

**A zero that holds up is worth more than a +26.7 that does not.** What the zero
means:

- A strong model does not make structural mistakes. Across 45 questions this one never
  invented a column, never wrote a non-`SELECT`, never produced a cartesian join. There
  was nothing for the layer to catch.
- **The errors it does make are semantic, and structural verification cannot reach
  them.** 26 of 45 answers parse, bind, pass every check, and return the wrong rows. The
  sharpest example: the model wrote `side = 'buy'` where the data holds `'BUY'`. A schema
  gives you column names and types but not the value domain, so casing is a guess. The
  fix is sampling distinct values of low-cardinality columns into the prompt — I know
  what it is, I have not built it.
- **A safety layer's value is on the tail, not the mean.** So I measure it separately, on
  48 labelled adversarial cases: **44/48 correct verdicts, 90.5% of malicious or broken
  SQL refused, all 6 legitimate queries preserved.** That is the number that says whether
  the verifier works. `run_eval.py` measures the model; `run_adversarial.py` measures the
  verifier; conflating them flatters one at the expense of the truth about the other.

**Caveats.**

- **This is not Spider.** It is a hand-built set of 45 pairs over my own warehouse. No
  Spider number appears anywhere in this repository.
- **42.2% is a floor, not a verdict on the model.** Scoring is strict result-set
  equality. Asked for the ten busiest traders the model returned an extra
  `trader_name` column — arguably a better answer, scored wrong. Loosening the metric
  would have produced a nicer number and a less honest one.
- **The verifier's catch-rate is measured separately from the model's accuracy**, on the
  adversarial corpus, because the verifier judges SQL strings and neither knows nor cares
  where they came from.
- `evaluation/record_fixtures.py` regenerates real fixtures from Groq with a key, and
  `run_eval.py --fixtures recorded` reproduces every number against them.

---

## 5b. "You didn't run Spider."

Correct, and I say so in the README rather than hoping nobody asks. Spider execution
accuracy needs the 20 dev SQLite databases and an engine that can execute Spider's gold
queries; this engine deliberately cannot execute most of them.

What I did run is the real Spider dev split — 1,034 gold queries over 20 databases — as a
**coverage** measurement:

| Stage | Queries | Share |
| --- | ---: | ---: |
| Parse under the sql-engine grammar | 875 | **84.6%** |
| Also bind against the Spider schema | 588 | 56.9% |
| Also bind, after normalising double-quoted literals | 774 | **74.9%** |

Every rejection has one of four causes and there is no long tail: 83 subqueries, 76 set
operations, 186 double-quoted string literals, 101 stricter-than-SQLite aggregations.

The double-quoted literals are the interesting one. Spider writes
`WHERE Airline = "JetBlue Airways"`; in standard SQL that is an *identifier*, so sql-engine
correctly says no such column exists, while SQLite reinterprets it as a string through a
fallback its own docs call a misfeature. Counting those as unsupported would have
understated coverage by 18 points. I caught it because a 28-point gap between parsing and
binding was implausible — `FAILURES.md` #8.

And the 101 aggregation rejections put a price on a design decision: refusing
`SELECT name, COUNT(*) ... GROUP BY id` costs 9.8% of Spider dev. I still think it is the
right call, because accepting it makes results non-deterministic on exactly the queries an
LLM most often gets wrong, but now I know what it costs.

---

## 6. What would you do next?

In the order I would actually do it:

1. **Row-group skipping.** Push the predicate into the Parquet reader as an Arrow filter
   expression so whole row groups are skipped without decoding. Biggest measured win
   available, and it makes the pushdown story complete.
2. **Semantic verification.** Extend the verifier beyond structure. Two candidates worth
   trying: execute the candidate against a small sample and flag an empty result set for
   review; and generate two independent candidates and flag disagreement. Both attack the
   four failures in section 5 that nothing currently catches.
3. **Spider execution accuracy.** The dev questions and schemas are already here and
   coverage is measured; what is missing is the 20 dev databases and the two features
   that would lift coverage past 90% — subqueries and set operations. Adding those, then
   loading the databases, turns a coverage number into a comparison with published work.
4. **Spill to disk.** `Sort` and `Aggregate` currently die rather than spill. External
   merge sort is well understood and would remove the "must fit in RAM" caveat.
5. **A real cost model.** Equi-depth histograms in the catalog, so the estimates in
   `EXPLAIN` are defensible on correlated predicates rather than assuming independence.
   This only becomes worth doing once there is more than one candidate plan.
6. **Join reordering.** Currently left-deep in written order with the right side always
   as the build side. Choosing the smaller side to build is a two-line change with a
   measurable payoff; full reordering needs item 5 first.

---

---

## Numbers to have ready

| Fact | Value | Where it came from |
| --- | --- | --- |
| Test suite | 483 passed, 53.76 s (six tests skip without the optional Spider files) | `python -m pytest` |
| Differential queries | 108 hand-written; 3 property-based tests generating up to 220 examples (120 + 60 + 40) | `tests/corpus.py`, `tests/test_differential.py` |
| Optimiser speedup | 1.70x – 8.17x | `bench/results.md` |
| vs file-backed SQLite | 0.51x – 3.01x | `bench/results.md` |
| vs pandas | 0.57x – 3.66x | `bench/results.md` |
| Execution accuracy, unverified | 42.2% (19/45) | `evaluation/results.md` |
| Execution accuracy, verified | 42.2% (19/45) | `evaluation/results.md` |
| First answers refused | 0/45 | `evaluation/results.md` |
| Mean repair calls | 0.00 | `evaluation/results.md` |
| Verified but semantically wrong | 26/45 | `evaluation/results.md` |
| Adversarial verdicts correct | 44/48 (91.7%) | `evaluation/adversarial_results.md` |
| Adversarial recall (malicious refused) | 38/42 (90.5%) | `evaluation/adversarial_results.md` |
| Legitimate queries preserved | 6/6 | `evaluation/adversarial_results.md` |
| Model used for fixtures | `groq:openai/gpt-oss-120b` | `evaluation/fixtures/recorded.json` |
| Spider dev coverage (parse) | 84.6% of 1,034 | `evaluation/spider_coverage.md` |
| Spider dev coverage (parse and bind) | 74.9% normalised / 56.9% as written | `evaluation/spider_coverage.md` |
| Real bugs found and recorded | 12 | `FAILURES.md` |
| Batch size | 4096 rows | `sql_engine/executor/batch.py` |
| Demo dataset | 200,000 trades + 60,000 prices | `data/generate.py` |
