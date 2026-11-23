"""The differential query corpus.

Every query here is run through sql-engine and through SQLite over identical data, and the
results must match. This list is the project's definition of "correct" -- it is more
important than any single line of engine code, because it is the only thing standing
between a plausible-looking result and a wrong one.

Queries are grouped by the feature they are meant to stress, and the awkward cases are
called out. A corpus of happy paths would pass on day one and catch nothing.
"""

from __future__ import annotations

#: (name, sql) pairs. `name` appears in pytest output so a failure names the feature.
CORPUS: list[tuple[str, str]] = [
    # -- projection and literals ---------------------------------------------------
    ("select_all_columns", "SELECT * FROM instruments ORDER BY instrument_id"),
    ("select_qualified_star", "SELECT i.* FROM instruments i ORDER BY i.instrument_id"),
    ("select_subset", "SELECT symbol, exchange FROM instruments ORDER BY symbol"),
    ("select_alias", "SELECT symbol AS ticker FROM instruments ORDER BY ticker LIMIT 10"),
    ("select_expression", "SELECT tick_size * contract_size AS notional_tick FROM instruments ORDER BY instrument_id"),
    ("select_literal", "SELECT symbol, 1 AS one, 'x' AS letter FROM instruments ORDER BY symbol LIMIT 5"),
    ("select_arithmetic_mix", "SELECT price + 1, price - 1, price * 2, price / 2 FROM trades ORDER BY trade_id LIMIT 20"),

    # -- integer semantics: truncation toward zero, sign of the remainder ----------
    ("integer_division_negative", "SELECT quantity, quantity / 7 FROM trades WHERE quantity < 0 ORDER BY trade_id LIMIT 50"),
    ("integer_modulo_negative", "SELECT quantity, quantity % 7 FROM trades WHERE quantity < 0 ORDER BY trade_id LIMIT 50"),
    ("integer_division_positive", "SELECT quantity, quantity / 7, quantity % 7 FROM trades WHERE quantity > 0 ORDER BY trade_id LIMIT 50"),
    ("division_by_zero_is_null", "SELECT quantity, quantity / (quantity - quantity) FROM trades ORDER BY trade_id LIMIT 10"),

    # -- WHERE ---------------------------------------------------------------------
    ("where_equality", "SELECT COUNT(*) FROM trades WHERE side = 'BUY'"),
    ("where_inequality", "SELECT COUNT(*) FROM trades WHERE price > 2000"),
    ("where_not_equal", "SELECT COUNT(*) FROM trades WHERE side <> 'BUY'"),
    ("where_and", "SELECT COUNT(*) FROM trades WHERE side = 'BUY' AND price > 2000"),
    ("where_or", "SELECT COUNT(*) FROM trades WHERE price < 100 OR price > 3900"),
    ("where_not", "SELECT COUNT(*) FROM trades WHERE NOT (price > 2000)"),
    ("where_nested_boolean", "SELECT COUNT(*) FROM trades WHERE (side = 'BUY' AND price > 2000) OR (side = 'SELL' AND quantity < 0)"),
    ("where_in_text", "SELECT COUNT(*) FROM instruments WHERE exchange IN ('CME', 'ICE')"),
    ("where_not_in_text", "SELECT COUNT(*) FROM instruments WHERE exchange NOT IN ('CME', 'ICE')"),
    ("where_in_numeric", "SELECT COUNT(*) FROM trades WHERE trader_id IN (1, 2, 3, 40)"),
    ("where_between", "SELECT COUNT(*) FROM trades WHERE price BETWEEN 1000 AND 2000"),
    ("where_not_between", "SELECT COUNT(*) FROM trades WHERE price NOT BETWEEN 1000 AND 2000"),
    ("where_like_prefix", "SELECT symbol FROM instruments WHERE symbol LIKE 'RA%' ORDER BY symbol"),
    ("where_like_suffix", "SELECT symbol FROM instruments WHERE symbol LIKE '%1' ORDER BY symbol"),
    ("where_like_contains", "SELECT symbol FROM instruments WHERE symbol LIKE '%0%' ORDER BY symbol LIMIT 20"),
    ("where_like_underscore", "SELECT symbol FROM instruments WHERE symbol LIKE 'EN_0%' ORDER BY symbol"),
    ("where_not_like", "SELECT COUNT(*) FROM instruments WHERE symbol NOT LIKE 'EN%'"),
    ("where_like_case_insensitive", "SELECT COUNT(*) FROM instruments WHERE symbol LIKE 'en%'"),
    ("where_on_text_ordering", "SELECT COUNT(*) FROM instruments WHERE symbol > 'M'"),

    # -- NULL handling: the part most engines get quietly wrong ---------------------
    ("where_is_null", "SELECT COUNT(*) FROM trades WHERE venue IS NULL"),
    ("where_is_not_null", "SELECT COUNT(*) FROM trades WHERE venue IS NOT NULL"),
    ("null_not_equal_excludes_nulls", "SELECT COUNT(*) FROM trades WHERE venue <> 'BLOCK'"),
    ("null_in_list_excludes_nulls", "SELECT COUNT(*) FROM trades WHERE venue IN ('BLOCK', 'TAS')"),
    ("null_not_in_list", "SELECT COUNT(*) FROM trades WHERE venue NOT IN ('BLOCK', 'TAS')"),
    ("null_and_false_is_false", "SELECT COUNT(*) FROM trades WHERE venue = 'BLOCK' AND 1 = 0"),
    ("null_or_true_is_true", "SELECT COUNT(*) FROM trades WHERE venue = 'BLOCK' OR 1 = 1"),
    ("null_not_of_null", "SELECT COUNT(*) FROM trades WHERE NOT (venue = 'BLOCK')"),
    ("null_csv_column", "SELECT COUNT(*) FROM traders WHERE desk IS NULL"),
    ("null_csv_column_not", "SELECT trader_id, desk FROM traders WHERE desk IS NOT NULL ORDER BY trader_id"),
    ("null_ordering_asc", "SELECT trader_id, desk FROM traders ORDER BY desk ASC, trader_id ASC"),
    ("null_ordering_desc", "SELECT trader_id, desk FROM traders ORDER BY desk DESC, trader_id ASC"),

    # -- aggregation ---------------------------------------------------------------
    ("count_star", "SELECT COUNT(*) FROM trades"),
    ("count_column_skips_null", "SELECT COUNT(venue) FROM trades"),
    ("count_distinct", "SELECT COUNT(DISTINCT trader_id) FROM trades"),
    ("sum_integer", "SELECT SUM(quantity) FROM trades"),
    ("sum_real", "SELECT SUM(price) FROM trades"),
    ("avg_real", "SELECT AVG(price) FROM trades"),
    ("min_max_numeric", "SELECT MIN(price), MAX(price) FROM trades"),
    ("min_max_text", "SELECT MIN(symbol), MAX(symbol) FROM instruments"),
    ("aggregate_over_empty_set", "SELECT COUNT(*), SUM(price), AVG(price), MIN(price), MAX(price) FROM trades WHERE price < 0"),
    ("group_by_single", "SELECT side, COUNT(*) FROM trades GROUP BY side ORDER BY side"),
    ("group_by_with_null_key", "SELECT venue, COUNT(*) FROM trades GROUP BY venue ORDER BY venue"),
    ("group_by_csv_null_key", "SELECT desk, COUNT(*) FROM traders GROUP BY desk ORDER BY desk"),
    ("group_by_two_keys", "SELECT side, venue, COUNT(*) FROM trades GROUP BY side, venue ORDER BY side, venue"),
    ("group_by_many_groups", "SELECT trader_id, COUNT(*), SUM(quantity) FROM trades GROUP BY trader_id ORDER BY trader_id"),
    ("group_by_sum_null_column", "SELECT side, COUNT(venue) FROM trades GROUP BY side ORDER BY side"),
    ("group_by_min_max_text", "SELECT exchange, MIN(symbol), MAX(symbol) FROM instruments GROUP BY exchange ORDER BY exchange"),
    ("group_by_avg", "SELECT trader_id, AVG(price) FROM trades GROUP BY trader_id ORDER BY trader_id"),
    ("group_by_count_distinct", "SELECT side, COUNT(DISTINCT trader_id) FROM trades GROUP BY side ORDER BY side"),
    ("group_by_expression_agg", "SELECT side, SUM(quantity * price) FROM trades GROUP BY side ORDER BY side"),
    ("aggregate_arithmetic", "SELECT side, SUM(quantity) / COUNT(*) FROM trades GROUP BY side ORDER BY side"),

    # -- HAVING ---------------------------------------------------------------------
    ("having_on_count", "SELECT trader_id, COUNT(*) FROM trades GROUP BY trader_id HAVING COUNT(*) > 5000 ORDER BY trader_id"),
    ("having_on_sum", "SELECT trader_id, SUM(quantity) FROM trades GROUP BY trader_id HAVING SUM(quantity) > 0 ORDER BY trader_id"),
    ("having_on_group_key", "SELECT side, COUNT(*) FROM trades GROUP BY side HAVING side = 'BUY'"),
    ("having_combined", "SELECT trader_id, COUNT(*) FROM trades GROUP BY trader_id HAVING COUNT(*) > 4900 AND trader_id < 20 ORDER BY trader_id"),
    ("having_no_group_by", "SELECT COUNT(*) FROM trades HAVING COUNT(*) > 0"),

    # -- ORDER BY / LIMIT / OFFSET --------------------------------------------------
    ("order_by_asc", "SELECT trade_id, price FROM trades ORDER BY price ASC, trade_id ASC LIMIT 25"),
    ("order_by_desc", "SELECT trade_id, price FROM trades ORDER BY price DESC, trade_id ASC LIMIT 25"),
    ("order_by_two_keys_mixed", "SELECT trade_id, side, price FROM trades ORDER BY side ASC, price DESC, trade_id ASC LIMIT 40"),
    ("order_by_text", "SELECT symbol FROM instruments ORDER BY symbol DESC LIMIT 20"),
    ("order_by_alias", "SELECT price * 2 AS doubled FROM trades ORDER BY doubled DESC LIMIT 10"),
    ("order_by_ordinal", "SELECT symbol, exchange FROM instruments ORDER BY 2, 1 LIMIT 20"),
    ("order_by_aggregate", "SELECT trader_id, COUNT(*) AS n FROM trades GROUP BY trader_id ORDER BY n DESC, trader_id ASC LIMIT 10"),
    ("order_by_not_in_select_list", "SELECT symbol FROM instruments ORDER BY exchange, symbol LIMIT 20"),
    ("limit_only", "SELECT trade_id FROM trades ORDER BY trade_id LIMIT 7"),
    ("limit_offset", "SELECT trade_id FROM trades ORDER BY trade_id LIMIT 7 OFFSET 13"),
    ("limit_zero", "SELECT trade_id FROM trades ORDER BY trade_id LIMIT 0"),
    ("offset_beyond_end", "SELECT symbol FROM instruments ORDER BY symbol LIMIT 5 OFFSET 500"),
    ("limit_larger_than_table", "SELECT symbol FROM instruments ORDER BY symbol LIMIT 100000"),

    # -- DISTINCT -------------------------------------------------------------------
    ("distinct_single", "SELECT DISTINCT exchange FROM instruments ORDER BY exchange"),
    ("distinct_two_columns", "SELECT DISTINCT exchange, currency FROM instruments ORDER BY exchange, currency"),
    ("distinct_with_nulls", "SELECT DISTINCT venue FROM trades ORDER BY venue"),
    ("distinct_expression", "SELECT DISTINCT quantity % 5 FROM trades ORDER BY quantity % 5"),

    # -- joins -----------------------------------------------------------------------
    ("inner_join_basic", "SELECT COUNT(*) FROM trades t INNER JOIN instruments i ON t.instrument_id = i.instrument_id"),
    ("inner_join_projection", "SELECT t.trade_id, i.symbol FROM trades t INNER JOIN instruments i ON t.instrument_id = i.instrument_id ORDER BY t.trade_id LIMIT 25"),
    ("inner_join_filter_both_sides", "SELECT COUNT(*) FROM trades t INNER JOIN instruments i ON t.instrument_id = i.instrument_id WHERE t.price > 3000 AND i.exchange = 'CME'"),
    ("inner_join_group_by", "SELECT i.asset_class, COUNT(*), SUM(t.quantity) FROM trades t INNER JOIN instruments i ON t.instrument_id = i.instrument_id GROUP BY i.asset_class ORDER BY i.asset_class"),
    ("inner_join_residual_predicate", "SELECT COUNT(*) FROM trades t INNER JOIN instruments i ON t.instrument_id = i.instrument_id AND t.price > i.tick_size * 1000"),
    ("inner_join_three_tables", "SELECT COUNT(*) FROM trades t INNER JOIN instruments i ON t.instrument_id = i.instrument_id INNER JOIN traders r ON t.trader_id = r.trader_id"),
    ("inner_join_two_facts", "SELECT COUNT(*) FROM daily_prices p INNER JOIN instruments i ON p.instrument_id = i.instrument_id WHERE i.exchange = 'LME'"),
    ("left_join_all_match", "SELECT COUNT(*) FROM trades t LEFT JOIN instruments i ON t.instrument_id = i.instrument_id"),
    ("left_join_preserves_unmatched", "SELECT r.trader_id, COUNT(t.trade_id) FROM traders r LEFT JOIN trades t ON r.trader_id = t.trader_id AND t.price > 3999 GROUP BY r.trader_id ORDER BY r.trader_id"),
    ("left_join_empty_right", "SELECT r.trader_id, t.trade_id FROM traders r LEFT JOIN trades t ON r.trader_id = t.trader_id AND t.price > 100000 ORDER BY r.trader_id"),
    ("left_join_null_key", "SELECT COUNT(*) FROM trades t LEFT JOIN traders r ON t.venue = r.desk"),
    ("join_on_nullable_column", "SELECT COUNT(*) FROM trades t INNER JOIN traders r ON t.venue = r.desk"),
    ("join_then_order_limit", "SELECT t.trade_id, i.symbol, t.price FROM trades t INNER JOIN instruments i ON t.instrument_id = i.instrument_id ORDER BY t.price DESC, t.trade_id ASC LIMIT 15"),
    ("join_composite_key", "SELECT COUNT(*) FROM trades t INNER JOIN daily_prices p ON t.instrument_id = p.instrument_id AND t.trade_date = p.price_date"),
    ("self_join", "SELECT COUNT(*) FROM instruments a INNER JOIN instruments b ON a.exchange = b.exchange WHERE a.instrument_id < 10 AND b.instrument_id < 10"),
    ("left_join_where_on_left", "SELECT COUNT(*) FROM traders r LEFT JOIN trades t ON r.trader_id = t.trader_id WHERE r.region = 'EMEA'"),

    # -- pushdown-sensitive shapes ---------------------------------------------------
    ("pushdown_single_table", "SELECT trade_id FROM trades WHERE trader_id = 7 AND side = 'BUY' ORDER BY trade_id LIMIT 20"),
    ("pushdown_across_join", "SELECT COUNT(*) FROM trades t INNER JOIN instruments i ON t.instrument_id = i.instrument_id WHERE i.currency = 'USD'"),
    ("pushdown_having_group_key", "SELECT side, COUNT(*) FROM trades GROUP BY side HAVING side <> 'SELL'"),
    ("pushdown_blocked_by_limit", "SELECT trade_id, price FROM trades ORDER BY trade_id LIMIT 100"),
    ("pushdown_left_join_asymmetry", "SELECT COUNT(*) FROM traders r LEFT JOIN trades t ON r.trader_id = t.trader_id WHERE t.price > 3990"),

    # -- shapes that mix everything --------------------------------------------------
    ("full_pipeline", (
        "SELECT i.asset_class, t.side, COUNT(*) AS trades, SUM(t.quantity) AS net_qty, "
        "AVG(t.price) AS avg_price "
        "FROM trades t INNER JOIN instruments i ON t.instrument_id = i.instrument_id "
        "WHERE t.price BETWEEN 500 AND 3500 AND t.venue IS NOT NULL "
        "GROUP BY i.asset_class, t.side "
        "HAVING COUNT(*) > 100 "
        "ORDER BY trades DESC, i.asset_class ASC, t.side ASC "
        "LIMIT 12"
    )),
    ("full_pipeline_left", (
        "SELECT r.region, r.desk, COUNT(t.trade_id) AS n, MAX(t.price) AS best "
        "FROM traders r LEFT JOIN trades t ON r.trader_id = t.trader_id AND t.side = 'BUY' "
        "GROUP BY r.region, r.desk "
        "ORDER BY r.region ASC, r.desk ASC"
    )),
    ("three_way_with_filters", (
        "SELECT r.desk, i.exchange, COUNT(*) AS n "
        "FROM trades t "
        "INNER JOIN instruments i ON t.instrument_id = i.instrument_id "
        "INNER JOIN traders r ON t.trader_id = r.trader_id "
        "WHERE t.quantity > 100 AND i.exchange IN ('CME','ICE') AND r.seniority >= 3 "
        "GROUP BY r.desk, i.exchange "
        "ORDER BY n DESC, r.desk ASC, i.exchange ASC "
        "LIMIT 20"
    )),
]

#: Queries whose results are order-defined. Everything else is compared as a multiset,
#: because SQL does not define row order without ORDER BY and asserting one would be
#: testing an implementation detail.
def is_ordered(sql: str) -> bool:
    lowered = sql.lower()
    if " order by " not in lowered:
        return False
    # A LIMIT over a non-deterministic ORDER BY (ties not broken) can legitimately pick
    # different rows in the two engines, so those are compared as multisets too.
    return True
