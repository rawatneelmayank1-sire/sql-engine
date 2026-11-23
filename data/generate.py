"""Generate the demo dataset.

A synthetic trading book: instruments, traders, executed trades and daily settlement
prices. The shape matters more than the story -- it gives the engine a large fact table
(``trades``), two small dimensions to join against, and a second fact table so multi-way
joins are possible.

Deliberate properties, each there to exercise something:

* ``trades.venue`` and ``traders.desk`` contain NULLs, so three-valued logic, NULL
  ordering and NULL-key joins are all reachable from ordinary queries.
* ``trades.quantity`` includes negative values, so integer division and MIN/MAX have a
  sign to get wrong.
* Dates are ISO-8601 TEXT, exactly as SQLite stores them, so a comparison against a date
  literal means the same thing in both engines.
* ``traders`` and ``instruments`` are CSV; the fact tables are Parquet. Both readers are
  therefore exercised by the normal test suite rather than by a special case.

Rerunning is safe and cheap: files are only written if missing, unless --force is given.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SEED = 20260909
DEFAULT_TRADES = 200_000
DEFAULT_PRICES = 60_000

ASSET_CLASSES = ["Rates", "Energy", "Metals", "Agriculture", "Equity Index"]
EXCHANGES = ["CME", "ICE", "EUREX", "LME", "SGX"]
CURRENCIES = ["USD", "EUR", "GBP", "JPY"]
DESKS = ["Spreads", "Outrights", "Curve", "Basis", None]
REGIONS = ["EMEA", "APAC", "AMER"]
VENUES = ["SCREEN", "BLOCK", "TAS", None]
SIDES = ["BUY", "SELL"]


def _dates(rng: np.random.Generator, count: int, start: str = "2025-01-01") -> np.ndarray:
    """Random ISO dates within one year of ``start``, as TEXT."""
    origin = np.datetime64(start)
    offsets = rng.integers(0, 365, size=count).astype("timedelta64[D]")
    return np.array([str(d) for d in (origin + offsets)], dtype=object)


def write_instruments(path: Path, rng: np.random.Generator, count: int = 120) -> None:
    rows = []
    for index in range(1, count + 1):
        asset_class = ASSET_CLASSES[index % len(ASSET_CLASSES)]
        rows.append(
            {
                "instrument_id": index,
                "symbol": f"{asset_class[:2].upper()}{index:03d}",
                "asset_class": asset_class,
                "exchange": EXCHANGES[int(rng.integers(0, len(EXCHANGES)))],
                "currency": CURRENCIES[int(rng.integers(0, len(CURRENCIES)))],
                "tick_size": float(np.round(rng.choice([0.01, 0.05, 0.25, 0.5]), 2)),
                "contract_size": int(rng.choice([1, 10, 100, 1000])),
            }
        )
    _write_csv(path, rows)


def write_traders(path: Path, rng: np.random.Generator, count: int = 40) -> None:
    joined = _dates(rng, count, "2019-01-01")
    rows = []
    for index in range(1, count + 1):
        desk = DESKS[int(rng.integers(0, len(DESKS)))]
        rows.append(
            {
                "trader_id": index,
                "trader_name": f"trader_{index:03d}",
                # A NULL desk is what makes `WHERE desk IS NULL` and NULL-key joins
                # reachable without contriving a special test fixture.
                "desk": desk if desk is not None else "",
                "region": REGIONS[int(rng.integers(0, len(REGIONS)))],
                "joined_date": joined[index - 1],
                "seniority": int(rng.integers(1, 6)),
            }
        )
    _write_csv(path, rows)


def write_trades(
    path: Path, rng: np.random.Generator, count: int, instruments: int, traders: int
) -> None:
    instrument_id = rng.integers(1, instruments + 1, size=count)
    trader_id = rng.integers(1, traders + 1, size=count)
    quantity = rng.integers(-500, 500, size=count)
    quantity[quantity == 0] = 1  # zero-quantity trades are not a thing
    price = np.round(rng.uniform(10.0, 4000.0, size=count), 4)
    venue_index = rng.integers(0, len(VENUES), size=count)
    venue = np.array([VENUES[i] for i in venue_index], dtype=object)
    side = np.array([SIDES[i] for i in rng.integers(0, 2, size=count)], dtype=object)

    table = pa.table(
        {
            "trade_id": pa.array(np.arange(1, count + 1, dtype=np.int64)),
            "instrument_id": pa.array(instrument_id.astype(np.int64)),
            "trader_id": pa.array(trader_id.astype(np.int64)),
            "trade_date": pa.array(_dates(rng, count), type=pa.string()),
            "side": pa.array(side, type=pa.string()),
            "quantity": pa.array(quantity.astype(np.int64)),
            "price": pa.array(price.astype(np.float64)),
            "notional": pa.array((quantity * price).astype(np.float64)),
            "venue": pa.array(venue, type=pa.string()),
        }
    )
    pq.write_table(table, path, compression="snappy")


def write_daily_prices(
    path: Path, rng: np.random.Generator, count: int, instruments: int
) -> None:
    instrument_id = rng.integers(1, instruments + 1, size=count)
    close = np.round(rng.uniform(10.0, 4000.0, size=count), 4)
    spread = np.round(rng.uniform(0.5, 40.0, size=count), 4)
    volume = rng.integers(0, 250_000, size=count)
    table = pa.table(
        {
            "instrument_id": pa.array(instrument_id.astype(np.int64)),
            "price_date": pa.array(_dates(rng, count), type=pa.string()),
            "open_price": pa.array(np.round(close - spread / 2, 4).astype(np.float64)),
            "high_price": pa.array(np.round(close + spread, 4).astype(np.float64)),
            "low_price": pa.array(np.round(close - spread, 4).astype(np.float64)),
            "close_price": pa.array(close.astype(np.float64)),
            "volume": pa.array(volume.astype(np.int64)),
        }
    )
    pq.write_table(table, path, compression="snappy")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate(
    directory: Path,
    *,
    trades: int = DEFAULT_TRADES,
    prices: int = DEFAULT_PRICES,
    force: bool = False,
) -> list[Path]:
    """Write the dataset into ``directory``. Existing files are kept unless ``force``."""
    directory.mkdir(parents=True, exist_ok=True)
    targets = {
        "instruments.csv": directory / "instruments.csv",
        "traders.csv": directory / "traders.csv",
        "trades.parquet": directory / "trades.parquet",
        "daily_prices.parquet": directory / "daily_prices.parquet",
    }
    if not force and all(path.exists() for path in targets.values()):
        return list(targets.values())

    rng = np.random.default_rng(SEED)
    write_instruments(targets["instruments.csv"], rng)
    write_traders(targets["traders.csv"], rng)
    write_trades(targets["trades.parquet"], rng, trades, instruments=120, traders=40)
    write_daily_prices(targets["daily_prices.parquet"], rng, prices, instruments=120)
    return list(targets.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the sql-engine demo dataset.")
    parser.add_argument(
        "--directory", type=Path, default=Path(__file__).resolve().parent / "warehouse"
    )
    parser.add_argument("--trades", type=int, default=DEFAULT_TRADES)
    parser.add_argument("--prices", type=int, default=DEFAULT_PRICES)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    written = generate(
        args.directory, trades=args.trades, prices=args.prices, force=args.force
    )
    for path in written:
        print(f"{path}  {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
