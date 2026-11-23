"""Deterministically generate the golden demo database for sql-engine eval.

Domain: a market / instrument reference database.

    exchanges  1---* instruments *---1 sectors        (instruments.sector_id is NULLABLE)
    instruments 1---* daily_bars
    instruments 1---* ratings *---1 analysts

Everything is driven by a single fixed seed, so re-running reproduces byte-identical CSVs.
Resumable: if every output already exists the script exits without regenerating.

Outputs (all under db/):
    <table>.csv, <table>.parquet   for each of the 6 tables
    schema.sql                     CREATE TABLE DDL (the SQLite schema of record)
    schema.json                    machine-readable tables/columns/types/keys
    market.sqlite                  the built database used for gold-SQL verification
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import random
import sqlite3

import pandas as pd

SEED = 20260909
HERE = pathlib.Path(__file__).resolve().parent
DB_DIR = HERE / "db"
TABLES = ["exchanges", "sectors", "instruments", "daily_bars", "analysts", "ratings"]

# --------------------------------------------------------------------------- schema

SCHEMA_SQL = """-- sql-engine golden eval database - market / instrument reference data
-- Generated deterministically by generate_db.py (seed 20260909). Do not hand-edit.

CREATE TABLE exchanges (
    exchange_id   INTEGER PRIMARY KEY,
    code          TEXT    NOT NULL,
    name          TEXT    NOT NULL,
    country       TEXT    NOT NULL,
    currency      TEXT    NOT NULL,
    timezone      TEXT    NOT NULL,
    founded_year  INTEGER NOT NULL
);

CREATE TABLE sectors (
    sector_id    INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    description  TEXT          -- nullable
);

CREATE TABLE instruments (
    instrument_id       INTEGER PRIMARY KEY,
    ticker              TEXT    NOT NULL,
    name                TEXT    NOT NULL,
    exchange_id         INTEGER NOT NULL,
    sector_id           INTEGER,          -- nullable: unclassified instruments
    instrument_type     TEXT    NOT NULL, -- equity | etf | adr
    currency            TEXT    NOT NULL,
    listing_date        TEXT    NOT NULL, -- ISO-8601 date
    shares_outstanding  INTEGER,          -- nullable
    is_active           INTEGER NOT NULL, -- 0/1
    FOREIGN KEY (exchange_id) REFERENCES exchanges (exchange_id),
    FOREIGN KEY (sector_id)   REFERENCES sectors (sector_id)
);

CREATE TABLE daily_bars (
    bar_id         INTEGER PRIMARY KEY,
    instrument_id  INTEGER NOT NULL,
    trade_date     TEXT    NOT NULL, -- ISO-8601 date
    open           REAL    NOT NULL,
    high           REAL    NOT NULL,
    low            REAL    NOT NULL,
    close          REAL    NOT NULL,
    volume         INTEGER NOT NULL,
    adj_close      REAL    NOT NULL,
    FOREIGN KEY (instrument_id) REFERENCES instruments (instrument_id)
);

CREATE TABLE analysts (
    analyst_id  INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    firm        TEXT    NOT NULL,
    country     TEXT,             -- nullable
    start_year  INTEGER NOT NULL
);

CREATE TABLE ratings (
    rating_id      INTEGER PRIMARY KEY,
    instrument_id  INTEGER NOT NULL,
    analyst_id     INTEGER NOT NULL,
    rating_date    TEXT    NOT NULL, -- ISO-8601 date
    rating         TEXT    NOT NULL, -- strong_buy | buy | hold | sell | underperform
    price_target   REAL,             -- nullable
    FOREIGN KEY (instrument_id) REFERENCES instruments (instrument_id),
    FOREIGN KEY (analyst_id)    REFERENCES analysts (analyst_id)
);

CREATE INDEX idx_bars_instrument ON daily_bars (instrument_id);
CREATE INDEX idx_bars_date       ON daily_bars (trade_date);
CREATE INDEX idx_instr_exchange  ON instruments (exchange_id);
CREATE INDEX idx_instr_sector    ON instruments (sector_id);
CREATE INDEX idx_ratings_instr   ON ratings (instrument_id);
"""

# --------------------------------------------------------------------------- fixtures

EXCHANGES = [
    (1, "XNAS", "Nasdaq Stock Market", "United States", "USD", "America/New_York", 1971),
    (2, "XNYS", "New York Stock Exchange", "United States", "USD", "America/New_York", 1792),
    (3, "XLON", "London Stock Exchange", "United Kingdom", "GBP", "Europe/London", 1801),
    (4, "XTKS", "Tokyo Stock Exchange", "Japan", "JPY", "Asia/Tokyo", 1878),
    (5, "XETR", "Deutsche Boerse Xetra", "Germany", "EUR", "Europe/Berlin", 1997),
    (6, "XTSE", "Toronto Stock Exchange", "Canada", "CAD", "America/Toronto", 1861),
    (7, "XHKG", "Hong Kong Stock Exchange", "Hong Kong", "HKD", "Asia/Hong_Kong", 1891),
    (8, "XASX", "Australian Securities Exchange", "Australia", "AUD", "Australia/Sydney", 1987),
]

SECTORS = [
    (1, "Technology", "Software, semiconductors and IT services"),
    (2, "Financials", "Banks, insurers and asset managers"),
    (3, "Healthcare", "Pharmaceuticals, biotech and medical devices"),
    (4, "Energy", "Oil, gas and renewable power"),
    (5, "Consumer Discretionary", "Retail, autos and leisure"),
    (6, "Consumer Staples", None),
    (7, "Industrials", "Capital goods, aerospace and transport"),
    (8, "Materials", "Chemicals, metals and mining"),
    (9, "Utilities", None),
    (10, "Real Estate", "REITs and property developers"),
    (11, "Communication Services", "Telecoms, media and entertainment"),
]
# sector 11 is deliberately left with ZERO instruments so that
# LEFT JOIN ... WHERE <right>.x IS NULL returns a non-empty, meaningful answer.
EMPTY_SECTOR_ID = 11

NAME_HEAD = [
    "Northwind", "Beacon", "Granite", "Vertex", "Harbour", "Cobalt", "Meridian", "Solstice",
    "Ironwood", "Lantern", "Aurora", "Pinnacle", "Cascade", "Ember", "Quarry", "Sable",
    "Trident", "Halcyon", "Juniper", "Keystone", "Onyx", "Redwood", "Summit", "Vantage",
    "Windrow", "Zephyr", "Anvil", "Bluepeak", "Cinder", "Drakemoor",
]
NAME_TAIL = [
    "Holdings", "Industries", "Group", "Corporation", "Partners", "Technologies",
    "Resources", "Capital", "Systems", "Laboratories",
]
FIRMS = [
    "Ashford Securities", "Brightwater Research", "Copeland & Vane", "Delmar Capital Markets",
    "Eastgate Partners", "Fairholme Advisory", "Gladstone Equity", "Harlow Research",
]
ANALYST_FIRST = [
    "Amara", "Bruno", "Chen", "Dilara", "Emeka", "Farida", "Goran", "Hana", "Ines", "Jonas",
    "Kavya", "Lucia", "Mateo", "Nadia", "Oskar", "Priya", "Quentin", "Rania", "Sven", "Tomas",
]
ANALYST_LAST = [
    "Abraham", "Bergstrom", "Costa", "Duarte", "Eriksen", "Fontaine", "Gallagher", "Haddad",
    "Ishikawa", "Jorgensen", "Kowalski", "Lindqvist", "Moreau", "Nakamura", "Okonkwo",
    "Petrov", "Quintero", "Rasmussen", "Salvatore", "Tanaka",
]
COUNTRIES = ["United States", "United Kingdom", "Germany", "Japan", "Canada", "Singapore"]
TYPES = ["equity", "equity", "equity", "equity", "etf", "adr"]
RATINGS = ["strong_buy", "buy", "buy", "hold", "hold", "sell", "underperform"]

N_INSTRUMENTS = 120
N_ANALYSTS = 40
N_TRADING_DAYS = 120
N_RATINGS = 620
START_DATE = dt.date(2024, 1, 2)


def trading_days(start: dt.date, count: int) -> list[dt.date]:
    """`count` consecutive weekdays starting at (or after) `start`."""
    days: list[dt.date] = []
    day = start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += dt.timedelta(days=1)
    return days


def build_frames() -> dict[str, pd.DataFrame]:
    rng = random.Random(SEED)

    exchanges = pd.DataFrame(
        EXCHANGES,
        columns=["exchange_id", "code", "name", "country", "currency", "timezone", "founded_year"],
    )
    sectors = pd.DataFrame(SECTORS, columns=["sector_id", "name", "description"])

    # ---- instruments -----------------------------------------------------
    ccy_by_exchange = {row[0]: row[4] for row in EXCHANGES}
    used_tickers: set[str] = set()
    instr_rows = []
    assignable_sectors = [s[0] for s in SECTORS if s[0] != EMPTY_SECTOR_ID]
    for i in range(1, N_INSTRUMENTS + 1):
        head = rng.choice(NAME_HEAD)
        tail = rng.choice(NAME_TAIL)
        base = (head[:3] + tail[:1]).upper()
        ticker = base
        bump = 0
        while ticker in used_tickers:
            bump += 1
            ticker = f"{base}{bump}"
        used_tickers.add(ticker)

        exchange_id = rng.choice([1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8])
        # ~12% of instruments are deliberately unclassified (sector_id IS NULL)
        sector_id = None if rng.random() < 0.12 else rng.choice(assignable_sectors)
        listing = START_DATE - dt.timedelta(days=rng.randint(400, 9000))
        # ~10% have unknown share count (nullable numeric)
        shares = None if rng.random() < 0.10 else rng.randint(5_000_000, 3_000_000_000)
        instr_rows.append(
            (
                i,
                ticker,
                f"{head} {tail}",
                exchange_id,
                sector_id,
                rng.choice(TYPES),
                ccy_by_exchange[exchange_id],
                listing.isoformat(),
                shares,
                1 if rng.random() < 0.88 else 0,
            )
        )
    instruments = pd.DataFrame(
        instr_rows,
        columns=[
            "instrument_id", "ticker", "name", "exchange_id", "sector_id",
            "instrument_type", "currency", "listing_date", "shares_outstanding", "is_active",
        ],
    )
    instruments["sector_id"] = instruments["sector_id"].astype("Int64")
    instruments["shares_outstanding"] = instruments["shares_outstanding"].astype("Int64")

    # ---- daily_bars ------------------------------------------------------
    days = trading_days(START_DATE, N_TRADING_DAYS)
    bar_rows = []
    bar_id = 0
    for instrument_id in range(1, N_INSTRUMENTS + 1):
        price = round(rng.uniform(4.0, 480.0), 2)
        drift = rng.uniform(-0.0009, 0.0014)
        vol = rng.uniform(0.008, 0.035)
        base_volume = rng.randint(90_000, 14_000_000)
        for day in days:
            bar_id += 1
            shock = rng.gauss(drift, vol)
            open_p = price
            close_p = max(0.51, round(open_p * (1.0 + shock), 2))
            high_p = round(max(open_p, close_p) * (1.0 + abs(rng.gauss(0, vol / 2))), 2)
            low_p = round(min(open_p, close_p) * (1.0 - abs(rng.gauss(0, vol / 2))), 2)
            low_p = max(0.50, min(low_p, open_p, close_p))
            high_p = max(high_p, open_p, close_p)
            volume = int(base_volume * rng.uniform(0.35, 2.4))
            bar_rows.append(
                (
                    bar_id, instrument_id, day.isoformat(),
                    open_p, high_p, low_p, close_p, volume,
                    round(close_p * 0.997, 4),
                )
            )
            price = close_p
    daily_bars = pd.DataFrame(
        bar_rows,
        columns=["bar_id", "instrument_id", "trade_date", "open", "high", "low",
                 "close", "volume", "adj_close"],
    )

    # ---- analysts --------------------------------------------------------
    analyst_rows = []
    for i in range(1, N_ANALYSTS + 1):
        first = ANALYST_FIRST[(i - 1) % len(ANALYST_FIRST)]
        last = ANALYST_LAST[(i * 7) % len(ANALYST_LAST)]
        # ~15% have unknown country (nullable text)
        country = None if rng.random() < 0.15 else rng.choice(COUNTRIES)
        analyst_rows.append((i, f"{first} {last}", rng.choice(FIRMS), country,
                             rng.randint(1998, 2022)))
    analysts = pd.DataFrame(analyst_rows,
                            columns=["analyst_id", "name", "firm", "country", "start_year"])

    # ---- ratings ---------------------------------------------------------
    close_by_instrument = (
        daily_bars.groupby("instrument_id")["close"].last().to_dict()
    )
    rating_rows = []
    for i in range(1, N_RATINGS + 1):
        instrument_id = rng.randint(1, N_INSTRUMENTS)
        rating_date = START_DATE + dt.timedelta(days=rng.randint(0, 170))
        # ~14% have no published price target (nullable numeric)
        if rng.random() < 0.14:
            target = None
        else:
            target = round(close_by_instrument[instrument_id] * rng.uniform(0.72, 1.45), 2)
        rating_rows.append(
            (i, instrument_id, rng.randint(1, N_ANALYSTS), rating_date.isoformat(),
             rng.choice(RATINGS), target)
        )
    ratings = pd.DataFrame(
        rating_rows,
        columns=["rating_id", "instrument_id", "analyst_id", "rating_date",
                 "rating", "price_target"],
    )

    return {
        "exchanges": exchanges, "sectors": sectors, "instruments": instruments,
        "daily_bars": daily_bars, "analysts": analysts, "ratings": ratings,
    }


def write_schema_json(frames: dict[str, pd.DataFrame]) -> dict:
    """Machine-readable schema description, kept in step with SCHEMA_SQL."""
    spec = {
        "database": "market",
        "engine": "sqlite3",
        "seed": SEED,
        "description": "Synthetic market / instrument reference database for sql-engine eval.",
        "tables": [
            {
                "name": "exchanges",
                "primary_key": ["exchange_id"],
                "foreign_keys": [],
                "columns": [
                    {"name": "exchange_id", "type": "INTEGER", "nullable": False},
                    {"name": "code", "type": "TEXT", "nullable": False},
                    {"name": "name", "type": "TEXT", "nullable": False},
                    {"name": "country", "type": "TEXT", "nullable": False},
                    {"name": "currency", "type": "TEXT", "nullable": False},
                    {"name": "timezone", "type": "TEXT", "nullable": False},
                    {"name": "founded_year", "type": "INTEGER", "nullable": False},
                ],
            },
            {
                "name": "sectors",
                "primary_key": ["sector_id"],
                "foreign_keys": [],
                "columns": [
                    {"name": "sector_id", "type": "INTEGER", "nullable": False},
                    {"name": "name", "type": "TEXT", "nullable": False},
                    {"name": "description", "type": "TEXT", "nullable": True},
                ],
            },
            {
                "name": "instruments",
                "primary_key": ["instrument_id"],
                "foreign_keys": [
                    {"column": "exchange_id", "references": "exchanges.exchange_id"},
                    {"column": "sector_id", "references": "sectors.sector_id"},
                ],
                "columns": [
                    {"name": "instrument_id", "type": "INTEGER", "nullable": False},
                    {"name": "ticker", "type": "TEXT", "nullable": False},
                    {"name": "name", "type": "TEXT", "nullable": False},
                    {"name": "exchange_id", "type": "INTEGER", "nullable": False},
                    {"name": "sector_id", "type": "INTEGER", "nullable": True},
                    {"name": "instrument_type", "type": "TEXT", "nullable": False,
                     "enum": ["equity", "etf", "adr"]},
                    {"name": "currency", "type": "TEXT", "nullable": False},
                    {"name": "listing_date", "type": "TEXT", "nullable": False,
                     "format": "date"},
                    {"name": "shares_outstanding", "type": "INTEGER", "nullable": True},
                    {"name": "is_active", "type": "INTEGER", "nullable": False,
                     "enum": [0, 1]},
                ],
            },
            {
                "name": "daily_bars",
                "primary_key": ["bar_id"],
                "foreign_keys": [
                    {"column": "instrument_id", "references": "instruments.instrument_id"},
                ],
                "columns": [
                    {"name": "bar_id", "type": "INTEGER", "nullable": False},
                    {"name": "instrument_id", "type": "INTEGER", "nullable": False},
                    {"name": "trade_date", "type": "TEXT", "nullable": False,
                     "format": "date"},
                    {"name": "open", "type": "REAL", "nullable": False},
                    {"name": "high", "type": "REAL", "nullable": False},
                    {"name": "low", "type": "REAL", "nullable": False},
                    {"name": "close", "type": "REAL", "nullable": False},
                    {"name": "volume", "type": "INTEGER", "nullable": False},
                    {"name": "adj_close", "type": "REAL", "nullable": False},
                ],
            },
            {
                "name": "analysts",
                "primary_key": ["analyst_id"],
                "foreign_keys": [],
                "columns": [
                    {"name": "analyst_id", "type": "INTEGER", "nullable": False},
                    {"name": "name", "type": "TEXT", "nullable": False},
                    {"name": "firm", "type": "TEXT", "nullable": False},
                    {"name": "country", "type": "TEXT", "nullable": True},
                    {"name": "start_year", "type": "INTEGER", "nullable": False},
                ],
            },
            {
                "name": "ratings",
                "primary_key": ["rating_id"],
                "foreign_keys": [
                    {"column": "instrument_id", "references": "instruments.instrument_id"},
                    {"column": "analyst_id", "references": "analysts.analyst_id"},
                ],
                "columns": [
                    {"name": "rating_id", "type": "INTEGER", "nullable": False},
                    {"name": "instrument_id", "type": "INTEGER", "nullable": False},
                    {"name": "analyst_id", "type": "INTEGER", "nullable": False},
                    {"name": "rating_date", "type": "TEXT", "nullable": False,
                     "format": "date"},
                    {"name": "rating", "type": "TEXT", "nullable": False,
                     "enum": ["strong_buy", "buy", "hold", "sell", "underperform"]},
                    {"name": "price_target", "type": "REAL", "nullable": True},
                ],
            },
        ],
    }
    for table in spec["tables"]:
        table["row_count"] = int(len(frames[table["name"]]))
    return spec


def _to_sqlite(value):
    """Convert a pandas/numpy scalar into something sqlite3 stores natively.

    This matters more than it looks: values from a pandas nullable ``Int64``
    column arrive as ``numpy.int64``, which sqlite3 has no adapter for and
    silently persists as an 8-byte BLOB. That makes every join on the column
    match zero rows while the INSERT still "succeeds". Always go via ``.item()``.
    """
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def build_sqlite(frames: dict[str, pd.DataFrame], path: pathlib.Path) -> None:
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA_SQL)
        for name in TABLES:
            frame = frames[name]
            cols = ",".join(frame.columns)
            marks = ",".join("?" * len(frame.columns))
            rows = [
                tuple(_to_sqlite(v) for v in rec)
                for rec in frame.itertuples(index=False, name=None)
            ]
            conn.executemany(f"INSERT INTO {name} ({cols}) VALUES ({marks})", rows)
        conn.commit()

        # Guard: no column may be stored as a BLOB. A BLOB here always means a
        # numpy scalar leaked through and the data is silently corrupt.
        for name in TABLES:
            for col in frames[name].columns:
                bad = conn.execute(
                    f"SELECT count(*) FROM {name} WHERE typeof({col}) = 'blob'"
                ).fetchone()[0]
                if bad:
                    raise RuntimeError(
                        f"{name}.{col}: {bad} rows stored as BLOB - type coercion failed"
                    )
        # Guard: foreign keys must actually resolve.
        orphans = conn.execute(
            "SELECT count(*) FROM instruments i "
            "LEFT JOIN sectors s ON i.sector_id = s.sector_id "
            "WHERE i.sector_id IS NOT NULL AND s.sector_id IS NULL"
        ).fetchone()[0]
        if orphans:
            raise RuntimeError(f"{orphans} instruments have an unresolvable sector_id")

        conn.execute("VACUUM")
    finally:
        conn.close()


def main() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    expected = (
        [DB_DIR / f"{t}.csv" for t in TABLES]
        + [DB_DIR / f"{t}.parquet" for t in TABLES]
        + [DB_DIR / "schema.sql", DB_DIR / "schema.json", DB_DIR / "market.sqlite"]
    )
    if all(p.exists() and p.stat().st_size > 0 for p in expected):
        print("all outputs already present - nothing to do (re-run is a no-op)")
        return

    frames = build_frames()
    for name in TABLES:
        frame = frames[name]
        frame.to_csv(DB_DIR / f"{name}.csv", index=False)
        frame.to_parquet(DB_DIR / f"{name}.parquet", index=False)
        print(f"  {name}: {len(frame)} rows")

    (DB_DIR / "schema.sql").write_text(SCHEMA_SQL, encoding="utf-8")
    (DB_DIR / "schema.json").write_text(
        json.dumps(write_schema_json(frames), indent=2), encoding="utf-8"
    )
    build_sqlite(frames, DB_DIR / "market.sqlite")

    total = sum(len(f) for f in frames.values())
    size_mb = sum(p.stat().st_size for p in DB_DIR.iterdir()) / 1e6
    print(f"TOTAL ROWS: {total}")
    print(f"DB_DIR SIZE: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
