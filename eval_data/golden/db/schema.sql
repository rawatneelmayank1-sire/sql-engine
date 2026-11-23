-- sql-engine golden eval database - market / instrument reference data
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
