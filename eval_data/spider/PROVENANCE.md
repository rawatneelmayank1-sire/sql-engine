# Spider dev set — provenance

**Status: OBTAINED.** Acquired 2026-09-09 by `_fetch_spider.py` (re-runnable; skips files already on disk).

## What was obtained

| Artifact | Rows / count | File |
|---|---|---|
| Dev questions + gold SQL (normalised) | **1034** rows | `dev.jsonl` |
| Dev split, original upstream form | 1034 rows | `dev-validation.parquet` |
| Database schemas (all Spider DBs) | **166** databases | `tables.json` |
| Database schemas, dev DBs only | **20** databases | `dev_schemas.json` |

1034 is the canonical size of the Spider dev split, and the dev questions span 20 distinct
databases — both match the published benchmark, so the download is believed complete and
uncorrupted.

## Exact sources

### 1. Questions and gold SQL
- **Source:** HuggingFace Hub dataset [`xlangai/spider`](https://huggingface.co/datasets/xlangai/spider)
  (this is the repo that the bare dataset name `spider` resolves to).
- **File:** `spider/validation-00000-of-00001.parquet`
- **URL:** `https://huggingface.co/datasets/xlangai/spider/resolve/main/spider/validation-00000-of-00001.parquet`
- **Repo commit (sha):** `0c350918f3f29ec754f1181c65cdce76cd6c133c`
- **Repo lastModified:** `2024-03-27T13:38:42.000Z`
- **Downloaded size:** 125,887 bytes
- **sha256:** `c3e2a46303899a2d4afe3f6a3a62e59f8d589f241b3cbfb52356479b1f054888`
- **Upstream columns:** `db_id`, `query`, `question`, `query_toks`, `query_toks_no_value`, `question_toks`

Note: the HuggingFace *datasets-server* row/parquet REST endpoints returned HTTP 404 for this
dataset, so the parquet was fetched directly from the hub `resolve` endpoint instead. The
`datasets` Python library is not installed in this environment; no dependency was added for it.

### 2. Database schemas
- **Source:** the official Yale LILY Spider repository,
  [`taoyds/spider`](https://github.com/taoyds/spider) (Yu et al., the benchmark authors).
- **File:** `evaluation_examples/examples/tables.json`
- **URL:** `https://raw.githubusercontent.com/taoyds/spider/master/evaluation_examples/examples/tables.json`
- **Downloaded size:** 810,971 bytes
- **sha256:** `61bb20aa401f03164e2d7f3b16509b7b5f79cc9c943ca7bd159046df1159e2ed`
- Contains `table_names`, `table_names_original`, `column_names`, `column_names_original`,
  `column_types`, `primary_keys` and `foreign_keys` per database.

### 3. What was NOT obtained
The **SQLite database files themselves** (`database/<db_id>/<db_id>.sqlite`) were not downloaded.
The official Yale release distributes them only inside a ~1 GB Google Drive archive, which is not
retrievable non-interactively and would blow the "well under 50 MB, committable" size budget for
this directory. **Consequence: execution accuracy cannot be computed against Spider from this
directory — only the questions, gold SQL and schemas are present.** Do not report Spider execution
numbers on the basis of these files.

## Licence

**CC BY-SA 4.0** (Creative Commons Attribution-ShareAlike 4.0 International), as declared in the
`xlangai/spider` dataset card metadata (`license: cc-by-sa-4.0`, tag `license:cc-by-sa-4.0`).
Attribution and share-alike obligations apply to any redistribution.

**Citation:**
Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li,
Qingning Yao, Shanelle Roman, Zilin Zhang, Dragomir Radev.
*Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and
Text-to-SQL Task.* EMNLP 2018.

## Caveats for SQL Engine

- Spider gold SQL **frequently uses constructs outside the sql-engine supported subset** (subqueries,
  `UNION`/`INTERSECT`/`EXCEPT`, self-joins with aliasing, nested aggregates). Spider is therefore a
  *coverage / robustness* set, not a set the engine is expected to fully satisfy. Filter before use.
- The hand-written `golden/` set is the one that stays inside the supported subset and has verified
  execution results.
