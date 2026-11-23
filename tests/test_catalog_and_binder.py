"""Catalog and binder tests.

The catalog is the other half of the safety argument: the parser guarantees the
statement is a SELECT, the catalog guarantees every name in it is real. If the catalog
could drift from the files on disk, "this column exists" would stop being a guarantee
and become a hope.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sql_engine import Catalog, DataType, Engine
from sql_engine.binder import Binder
from sql_engine.catalog import infer_csv_type
from sql_engine.errors import (
    AmbiguousColumnError,
    BindError,
    CatalogError,
    InvalidAggregationError,
    UnknownColumnError,
)
from sql_engine.parser import parse


# --------------------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------------------


def test_catalog_discovers_every_table(catalog: Catalog):
    assert set(catalog.table_names) == {"trades", "daily_prices", "instruments", "traders"}


def test_schema_is_read_from_the_files_not_a_manifest(catalog: Catalog):
    """A hand-written manifest could drift; an inferred schema cannot."""
    trades = catalog.table("trades")
    assert trades.column("trade_id").dtype is DataType.INTEGER
    assert trades.column("price").dtype is DataType.REAL
    assert trades.column("side").dtype is DataType.TEXT
    assert trades.row_count == 200_000


def test_both_file_formats_are_represented(catalog: Catalog):
    formats = {table.name: table.file_format for table in catalog}
    assert formats["trades"] == "parquet"
    assert formats["traders"] == "csv"


def test_column_lookup_is_case_insensitive(catalog: Catalog):
    trades = catalog.table("trades")
    assert trades.has_column("TRADE_ID")
    assert trades.column("Trade_Id").name == "trade_id"


def test_unknown_table_error_lists_the_real_ones(catalog: Catalog):
    with pytest.raises(CatalogError) as error:
        catalog.table("customers")
    assert "trades" in str(error.value)


def test_unknown_column_error_lists_the_real_ones(catalog: Catalog):
    with pytest.raises(CatalogError) as error:
        catalog.table("traders").column("lifetime_value")
    assert "trader_name" in str(error.value)


def test_statistics_are_collected_for_estimation(catalog: Catalog):
    stats = catalog.table("trades").column("price").stats
    assert stats is not None
    assert stats.minimum is not None and stats.maximum is not None
    assert stats.distinct_count > 1


def test_nullable_csv_column_reports_its_nulls(catalog: Catalog):
    assert catalog.table("traders").column("desk").stats.null_count > 0


@pytest.mark.parametrize(
    "values,expected",
    [
        (["1", "2", "3"], DataType.INTEGER),
        (["1", "", "3"], DataType.INTEGER),
        (["1.5", "2"], DataType.REAL),
        (["a", "1"], DataType.TEXT),
        (["", ""], DataType.TEXT),
        (["2025-01-01"], DataType.TEXT),
    ],
)
def test_csv_type_inference(values: list[str], expected: DataType):
    assert infer_csv_type(values) is expected


def test_empty_directory_is_an_error(tmp_path: Path):
    with pytest.raises(CatalogError):
        Catalog.from_directory(tmp_path)


def test_describe_is_compact_and_names_the_types(catalog: Catalog):
    text = catalog.describe()
    assert "trades(" in text
    assert "price REAL" in text
    assert "200000 rows" in text


# --------------------------------------------------------------------------------------
# Binder
# --------------------------------------------------------------------------------------


def bind(catalog: Catalog, sql: str):
    return Binder(catalog).bind(parse(sql))


def test_unqualified_columns_are_qualified(catalog: Catalog):
    bound = bind(catalog, "SELECT trade_id FROM trades")
    assert str(bound.statement.select_list[0].expr) == "trades.trade_id"


def test_alias_is_used_for_qualification(catalog: Catalog):
    bound = bind(catalog, "SELECT trade_id FROM trades t")
    assert str(bound.statement.select_list[0].expr) == "t.trade_id"


def test_star_is_expanded_at_bind_time(catalog: Catalog):
    bound = bind(catalog, "SELECT * FROM instruments")
    assert len(bound.statement.select_list) == 7
    assert bound.output_names[0] == "instrument_id"


def test_qualified_star_expands_one_table_only(catalog: Catalog):
    bound = bind(
        catalog,
        "SELECT i.* FROM trades t INNER JOIN instruments i "
        "ON t.instrument_id = i.instrument_id",
    )
    assert len(bound.statement.select_list) == 7


def test_unknown_column_raises_its_own_type(catalog: Catalog):
    with pytest.raises(UnknownColumnError):
        bind(catalog, "SELECT lifetime_value FROM traders")


def test_ambiguous_column_raises_its_own_type(catalog: Catalog):
    with pytest.raises(AmbiguousColumnError):
        bind(
            catalog,
            "SELECT instrument_id FROM trades t INNER JOIN instruments i "
            "ON t.instrument_id = i.instrument_id",
        )


def test_duplicate_alias_is_rejected(catalog: Catalog):
    with pytest.raises(BindError):
        bind(catalog, "SELECT 1 FROM trades t INNER JOIN instruments t ON 1 = 1")


def test_unknown_alias_in_a_qualified_reference_is_rejected(catalog: Catalog):
    with pytest.raises(BindError):
        bind(catalog, "SELECT x.trade_id FROM trades t")


def test_bare_column_alongside_an_aggregate_is_rejected(catalog: Catalog):
    """SQLite would pick an arbitrary row; refusing is more useful and deterministic."""
    with pytest.raises(InvalidAggregationError):
        bind(catalog, "SELECT side, price, COUNT(*) FROM trades GROUP BY side")


def test_grouped_column_alongside_an_aggregate_is_fine(catalog: Catalog):
    bind(catalog, "SELECT side, COUNT(*) FROM trades GROUP BY side")


def test_order_by_can_name_a_select_list_alias(catalog: Catalog):
    bound = bind(catalog, "SELECT COUNT(*) AS n FROM trades ORDER BY n DESC")
    assert "COUNT(*)" in str(bound.statement.order_by[0].expr)


def test_order_by_can_use_an_ordinal(catalog: Catalog):
    bound = bind(catalog, "SELECT symbol, exchange FROM instruments ORDER BY 2")
    assert str(bound.statement.order_by[0].expr) == "instruments.exchange"


def test_order_by_ordinal_out_of_range_is_rejected(catalog: Catalog):
    with pytest.raises(BindError):
        bind(catalog, "SELECT symbol FROM instruments ORDER BY 5")


def test_having_can_name_a_select_list_alias(catalog: Catalog):
    bound = bind(
        catalog, "SELECT side, COUNT(*) AS n FROM trades GROUP BY side HAVING n > 1"
    )
    assert "COUNT(*)" in str(bound.statement.having)


def test_column_statistics_are_collected_for_every_relation(catalog: Catalog):
    bound = bind(
        catalog,
        "SELECT t.price FROM trades t INNER JOIN instruments i "
        "ON t.instrument_id = i.instrument_id",
    )
    assert "t.price" in bound.column_stats
    assert "i.symbol" in bound.column_stats


# --------------------------------------------------------------------------------------
# Engine surface
# --------------------------------------------------------------------------------------


def test_compile_reads_no_data(engine: Engine):
    """Compilation must be free of I/O so EXPLAIN and verification stay cheap."""
    compiled = engine.compile("SELECT COUNT(*) FROM trades")
    assert compiled.optimised_plan is not None
    assert compiled.output_names == ("COUNT(*)",)


def test_result_set_helpers(engine: Engine):
    result = engine.query("SELECT side, COUNT(*) AS n FROM trades GROUP BY side ORDER BY side")
    assert result.column_names == ("side", "n")
    assert len(result) == 2
    assert set(result.to_dicts()[0]) == {"side", "n"}


def test_duplicate_output_names_do_not_collide(engine: Engine):
    """`SELECT a AS x, b AS x` must return two columns, not one written twice."""
    result = engine.query("SELECT side AS x, venue AS x FROM trades LIMIT 1")
    assert len(result.rows[0]) == 2
