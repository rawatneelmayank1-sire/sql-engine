"""Fetch the Spider text-to-SQL dev set. Resumable: skips any artifact already on disk.

Sources (recorded in PROVENANCE.md):
  dev questions + gold SQL : HuggingFace hub  xlangai/spider  spider/validation-*.parquet
  database schemas         : GitHub taoyds/spider  evaluation_examples/examples/tables.json
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import ssl
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
CTX = ssl.create_default_context()
UA = {"User-Agent": "Mozilla/5.0 (sql-engine eval dataset builder)"}

PARQUET_URL = (
    "https://huggingface.co/datasets/xlangai/spider/resolve/main/"
    "spider/validation-00000-of-00001.parquet"
)
TABLES_URL = (
    "https://raw.githubusercontent.com/taoyds/spider/master/"
    "evaluation_examples/examples/tables.json"
)


def download(url: str, dest: pathlib.Path) -> bytes:
    """Download `url` to `dest`, reusing the file if it already exists."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (already on disk): {dest.name} ({dest.stat().st_size} bytes)")
        return dest.read_bytes()
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120, context=CTX) as resp:
        payload = resp.read()
    dest.write_bytes(payload)
    print(f"  downloaded {dest.name}: {len(payload)} bytes")
    return payload


def main() -> None:
    report: dict[str, object] = {}

    print("[1/3] dev parquet")
    raw_parquet = HERE / "dev-validation.parquet"
    blob = download(PARQUET_URL, raw_parquet)
    report["parquet_sha256"] = hashlib.sha256(blob).hexdigest()
    report["parquet_bytes"] = len(blob)

    print("[2/3] schemas (tables.json)")
    tables_path = HERE / "tables.json"
    tblob = download(TABLES_URL, tables_path)
    report["tables_sha256"] = hashlib.sha256(tblob).hexdigest()
    report["tables_bytes"] = len(tblob)
    tables = json.loads(tblob)
    report["n_databases"] = len(tables)

    print("[3/3] normalise dev split -> dev.jsonl")
    import pandas as pd

    frame = pd.read_parquet(raw_parquet)
    report["parquet_columns"] = list(frame.columns)
    dev_path = HERE / "dev.jsonl"
    n = 0
    with dev_path.open("w", encoding="utf-8") as handle:
        for row in frame.to_dict(orient="records"):
            record = {
                "id": f"spider_dev_{n:04d}",
                "db_id": row.get("db_id"),
                "question": row.get("question"),
                "sql": row.get("query"),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    report["n_dev_rows"] = n
    report["n_unique_dbs_in_dev"] = int(frame["db_id"].nunique())

    (HERE / "_fetch_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("REPORT:", json.dumps(report))


if __name__ == "__main__":
    main()
