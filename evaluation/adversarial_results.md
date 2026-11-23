# Adversarial verification results

Regenerate with `python evaluation/run_adversarial.py`. Nothing here is typed by hand.

## What this measures

`run_eval.py` measures the **model**: can it turn a question into correct SQL.
This measures the **verifier**: given SQL that is wrong, dangerous or malformed,
does it refuse. Those are different questions and they deserve different tables.

## Headline

| Metric | Value |
| --- | ---: |
| Cases | 48 |
| Overall correct verdicts | 44/48 = 91.7% |
| Malicious/broken caught (recall) | 38/42 = 90.5% |
| Legitimate queries preserved | 6/6 = 100.0% |
| Precision of refusals | 100.0% |

A verifier that refused everything would score 100% recall and would be useless.
The allow-cases exist so that failure mode is visible rather than hidden.

## By category

| Category | Cases | Correct | |
| --- | ---: | ---: | --- |
| `not_select` | 8 | 8/8 | |
| `unparseable` | 8 | 8/8 | |
| `invented_column` | 7 | 7/7 | |
| `injection_attempt` | 6 | 6/6 | |
| `valid` | 6 | 6/6 | |
| `invented_table` | 5 | 5/5 | |
| `cartesian` | 4 | 4/4 | |
| `unbounded` | 4 | 0/4 | **<- misses** |

## Every miss, listed

| Case | Category | Expected | Got | Verifier said |
| --- | --- | --- | --- | --- |
| `a039` | `unbounded` | reject | allow | — |
| `a040` | `unbounded` | reject | allow | — |
| `a041` | `unbounded` | reject | allow | — |
| `a042` | `unbounded` | reject | allow | — |

These are reported rather than tuned away. A miss on an `allow` case is a
false positive: the verifier blocked something legitimate. A miss on a
`reject` case is a false negative, which is the more dangerous direction.
