"""Physical operators: a pull-based (Volcano) pipeline that moves batches, not rows.

Each operator is an iterator of :class:`~sql_engine.executor.batch.Batch`. Classic Volcano
pulls one *row* per ``next()``; this pulls one *batch*, which is the difference between
paying Python's per-call overhead 50 million times and paying it 12 thousand times on a
50-million-row scan. Everything else in this file follows from that choice.

Three operators are blocking -- Sort, Aggregate and the build side of HashJoin -- and
each says so in its docstring, because a blocking operator is where memory goes and
where an interviewer will ask about spilling (sql-engine does not spill; see Limitations).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from ..ast_nodes import (
    Expression,
    FunctionCall,
    JoinType,
    OrderByItem,
    SortDirection,
    Star,
)
from ..binder import qualified
from ..catalog import DataType, TableSchema
from ..errors import ExecutionError
from .batch import (
    DEFAULT_BATCH_SIZE,
    Batch,
    Column,
    column_from_arrow,
    concat_batches,
    split_batch,
)
from .expressions import evaluate, truth_mask


class Operator(ABC):
    """A node in the physical pipeline."""

    #: Rows this operator emitted during the last run. Collected for EXPLAIN ANALYZE
    #: style reporting and for the benchmark, so claims about pushdown can be checked
    #: against rows actually produced rather than estimates.
    rows_emitted: int = 0

    @abstractmethod
    def execute(self) -> Iterator[Batch]:
        """Yield output batches."""

    def _count(self, batch: Batch) -> Batch:
        self.rows_emitted += batch.num_rows
        return batch


def empty_like(batch: Batch) -> Batch:
    """A zero-row batch with the same columns and dtypes as ``batch``.

    This is how schema survives a result with no rows. See FAILURES.md #2 for the outer
    join that made it necessary.
    """
    return batch.take(np.empty(0, dtype=np.int64))


# --------------------------------------------------------------------------------------
# Scan
# --------------------------------------------------------------------------------------


class ScanOperator(Operator):
    """Reads a table, applying projection and pushed predicates at the source.

    This is where the optimiser's two rules physically pay off: ``columns`` is handed
    straight to the Parquet reader (so pruned columns are never decoded from disk), and
    ``predicates`` are evaluated on each batch before it leaves this operator (so
    filtered rows never enter the pipeline).
    """

    def __init__(
        self,
        table: TableSchema,
        alias: str,
        columns: Sequence[str] | None = None,
        predicates: Sequence[Expression] = (),
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.table = table
        self.alias = alias
        self.columns = list(columns) if columns is not None else list(table.column_names)
        self.predicates = list(predicates)
        self.batch_size = batch_size
        self.rows_emitted = 0
        #: Rows read off disk before the pushed predicates were applied.
        self.rows_scanned = 0

    def execute(self) -> Iterator[Batch]:
        emitted = False
        last: Batch | None = None
        for arrow_batch in self._read():
            batch = self._to_batch(arrow_batch)
            self.rows_scanned += batch.num_rows
            last = batch
            for predicate in self.predicates:
                if batch.num_rows == 0:
                    break
                batch = batch.filter(truth_mask(evaluate(predicate, batch)))
            if batch.num_rows:
                emitted = True
                yield self._count(batch)
        if not emitted:
            # A scan that filters everything out must still hand the schema downstream:
            # an outer join above needs the column names and dtypes to build its NULL
            # padding. Rows and schema are different things. See FAILURES.md #2.
            yield empty_like(last) if last is not None else self.empty_batch()

    def _read(self) -> Iterator[pa.RecordBatch]:
        if self.table.file_format == "parquet":
            handle = pq.ParquetFile(self.table.path)
            yield from handle.iter_batches(batch_size=self.batch_size, columns=self.columns)
            return
        if self.table.file_format == "csv":
            # CSV has no column index, so the whole file is parsed and then narrowed.
            # Types come from the catalog rather than from pyarrow's inference so that
            # the column dtypes cannot disagree with what the planner assumed.
            convert = pacsv.ConvertOptions(
                column_types={
                    column.name: _arrow_type(column.dtype) for column in self.table.columns
                },
                include_columns=self.columns,
                # Without this an empty TEXT field reads as "" rather than NULL, which
                # silently disagrees with both SQLite and this project's own catalog
                # statistics. See FAILURES.md #1.
                strings_can_be_null=True,
            )
            table = pacsv.read_csv(self.table.path, convert_options=convert)
            yield from table.to_batches(max_chunksize=self.batch_size)
            return
        raise ExecutionError(f"unsupported file format {self.table.file_format!r}")

    def empty_batch(self) -> Batch:
        """A zero-row batch matching this scan's projected schema.

        Needed when the underlying file itself yields no record batches at all, so there
        is no batch to copy a schema from.
        """
        columns: dict[str, Column] = {}
        for name in self.columns:
            schema_column = self.table.column(name)
            dtype = schema_column.dtype.numpy_dtype
            columns[qualified(self.alias, schema_column.name)] = Column(
                np.empty(0, dtype=dtype), np.empty(0, dtype=bool)
            )
        return Batch(columns, 0)

    def _to_batch(self, arrow_batch: pa.RecordBatch) -> Batch:
        columns: dict[str, Column] = {}
        for name in self.columns:
            schema_column = self.table.column(name)
            column = column_from_arrow(arrow_batch.column(name), schema_column.dtype)
            # Columns are keyed by their qualified name so a self-join or a join between
            # tables that share a column name cannot collide.
            columns[qualified(self.alias, schema_column.name)] = column
        return Batch(columns, arrow_batch.num_rows)


def _arrow_type(dtype: DataType) -> pa.DataType:
    return {
        DataType.INTEGER: pa.int64(),
        DataType.REAL: pa.float64(),
        DataType.TEXT: pa.string(),
        DataType.BOOLEAN: pa.bool_(),
    }[dtype]


# --------------------------------------------------------------------------------------
# Filter / Project / Limit
# --------------------------------------------------------------------------------------


class FilterOperator(Operator):
    """Applies WHERE/HAVING conjuncts. Streaming."""

    def __init__(self, child: Operator, predicates: Sequence[Expression]) -> None:
        self.child = child
        self.predicates = list(predicates)
        self.rows_emitted = 0

    def execute(self) -> Iterator[Batch]:
        emitted = False
        last: Batch | None = None
        for batch in self.child.execute():
            last = batch
            for predicate in self.predicates:
                if batch.num_rows == 0:
                    break
                batch = batch.filter(truth_mask(evaluate(predicate, batch)))
            if batch.num_rows:
                emitted = True
                yield self._count(batch)
        if not emitted and last is not None:
            yield empty_like(last)


class ProjectOperator(Operator):
    """Evaluates the select list. Streaming.

    Computed columns are *added* to the batch rather than replacing it. Keeping the base
    columns alive lets an ORDER BY above this operator sort on a column that is not in
    the select list, which SQL allows. The cost is memory held until the sort completes;
    with no sort above, the extra columns are simply dropped at output time.
    """

    def __init__(
        self, child: Operator, expressions: Sequence[Expression], keys: Sequence[str]
    ) -> None:
        if len(expressions) != len(keys):
            raise ExecutionError("projection expression/key count mismatch")
        self.child = child
        self.expressions = list(expressions)
        self.keys = list(keys)
        self.rows_emitted = 0

    def execute(self) -> Iterator[Batch]:
        for batch in self.child.execute():
            computed: dict[str, Column] = {}
            for key, expression in zip(self.keys, self.expressions):
                if key in batch.columns and str(expression) == key:
                    continue  # already materialised (a plain column reference)
                computed[key] = evaluate(expression, batch)
            yield self._count(batch.with_columns(computed) if computed else batch)


class LimitOperator(Operator):
    """LIMIT/OFFSET. Streaming, and stops pulling as soon as it is satisfied."""

    def __init__(self, child: Operator, limit: int | None, offset: int = 0) -> None:
        self.child = child
        self.limit = limit
        self.offset = offset
        self.rows_emitted = 0

    def execute(self) -> Iterator[Batch]:
        skipped = 0
        produced = 0
        for batch in self.child.execute():
            if self.limit is not None and produced >= self.limit:
                return
            if skipped < self.offset:
                take = min(batch.num_rows, self.offset - skipped)
                skipped += take
                batch = batch.slice(take, batch.num_rows - take)
                if batch.num_rows == 0:
                    continue
            if self.limit is not None:
                remaining = self.limit - produced
                if batch.num_rows > remaining:
                    batch = batch.slice(0, remaining)
            produced += batch.num_rows
            yield self._count(batch)


# --------------------------------------------------------------------------------------
# Key encoding, shared by join / group by / distinct / sort
# --------------------------------------------------------------------------------------


def to_masked_series(column: Column) -> pd.Series:
    """A pandas Series over ``column`` with NULLs represented as NA.

    The fast path matters: converting a null-free int64 column to an object array to
    carry NULLs would allocate a Python object per row, which is precisely the
    per-row cost the rest of the executor exists to avoid. Only columns that actually
    contain NULLs pay for the mask.
    """
    if not column.has_nulls:
        return pd.Series(column.values, copy=False)
    if column.values.dtype == object:
        return pd.Series(column.as_nullable_object(), dtype=object, copy=False)
    # Numeric with NULLs: NaN is a valid NA marker for grouping and ordering, and every
    # integer this engine reads is exact in float64.
    return pd.Series(np.where(column.valid, column.values, np.nan), dtype=np.float64)


class KeyEncoder:
    """Maps one or more key columns onto a single int64 code per row.

    Multi-column keys are combined in mixed radix rather than by building tuples,
    because a tuple key forces a Python-object hash per row and would put an
    interpreter loop back into the hottest part of the join.

    NULL gets code 0 and real values start at 1, so a NULL key is representable without
    a sentinel collision. Whether NULLs *match* is then a per-operator decision: joins
    say no (SQL), GROUP BY says yes (SQL).
    """

    def __init__(self, build_columns: Sequence[Column]) -> None:
        if not build_columns:
            raise ExecutionError("a key encoder needs at least one column")
        self._uniques: list[pd.Index] = []
        self._radices: list[int] = []
        self.build_codes = np.zeros(len(build_columns[0]), dtype=np.int64)
        self.build_has_null = np.zeros(len(build_columns[0]), dtype=bool)

        for column in build_columns:
            series = to_masked_series(column)
            codes, uniques = pd.factorize(series, sort=False, use_na_sentinel=True)
            self._uniques.append(pd.Index(uniques))
            radix = len(uniques) + 1
            self._radices.append(radix)
            shifted = codes.astype(np.int64) + 1  # NA (-1) becomes 0
            if self.build_codes.max(initial=0) > (2**62) // max(1, radix):
                raise ExecutionError("composite join key space is too large to encode")
            self.build_codes = self.build_codes * radix + shifted
            self.build_has_null |= ~column.valid

    def encode(self, columns: Sequence[Column]) -> tuple[np.ndarray, np.ndarray]:
        """Encode probe-side columns into the build side's code space.

        Returns ``(codes, has_null)``. A value never seen on the build side is given a
        code that cannot occur there, so it simply fails to match rather than aliasing
        onto a real key.
        """
        if len(columns) != len(self._uniques):
            raise ExecutionError("key column count mismatch between build and probe sides")
        length = len(columns[0])
        codes = np.zeros(length, dtype=np.int64)
        has_null = np.zeros(length, dtype=bool)
        unseen = np.zeros(length, dtype=bool)
        for column, uniques, radix in zip(columns, self._uniques, self._radices):
            series = to_masked_series(column)
            positions = uniques.get_indexer(series)  # hash-table probe, -1 when absent
            shifted = positions.astype(np.int64) + 1
            unseen |= positions < 0
            codes = codes * radix + np.where(shifted < 0, 0, shifted)
            has_null |= ~column.valid
        codes = np.where(unseen, -1, codes)
        return codes, has_null


def _grouped_codes(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a CSR-style index over ``codes``.

    Returns ``(unique_codes, row_order, group_offsets, group_counts)`` where
    ``row_order[group_offsets[g] : group_offsets[g] + group_counts[g]]`` is every row
    belonging to group ``g``. This is what makes the probe phase a vectorised gather
    instead of a per-row dictionary lookup.
    """
    row_order = np.argsort(codes, kind="stable")
    sorted_codes = codes[row_order]
    unique_codes, first_index, counts = np.unique(
        sorted_codes, return_index=True, return_counts=True
    )
    return unique_codes, row_order, first_index.astype(np.int64), counts.astype(np.int64)


def _ragged_expand(
    starts: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """Expand per-group (start, count) pairs into a flat index array.

    ``starts=[10, 40], counts=[2, 3]`` becomes ``[10, 11, 40, 41, 42]``. Written without
    a loop because it runs once per probe batch on every joined row.
    """
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    repeated_starts = np.repeat(starts, counts)
    group_begin = np.repeat(np.cumsum(counts) - counts, counts)
    return repeated_starts + (np.arange(total, dtype=np.int64) - group_begin)


# --------------------------------------------------------------------------------------
# Hash join
# --------------------------------------------------------------------------------------


class HashJoinOperator(Operator):
    """Equi-join. The right input is the build side; the left is probed batch by batch.

    Build side is always the right relation. A real optimiser would pick the smaller
    side, but sql-engine does not reorder joins at all (see Limitations), and for a LEFT
    join the choice is forced anyway: the preserved side must be the probe side so its
    unmatched rows can be emitted.

    Blocking on the build side: the whole right relation is materialised in memory.
    """

    def __init__(
        self,
        left: Operator,
        right: Operator,
        join_type: JoinType,
        equi_keys: Sequence[tuple[Expression, Expression]],
        residual: Sequence[Expression] = (),
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.left = left
        self.right = right
        self.join_type = join_type
        self.equi_keys = list(equi_keys)
        self.residual = list(residual)
        self.batch_size = batch_size
        self.rows_emitted = 0

    def execute(self) -> Iterator[Batch]:
        build = concat_batches(list(self.right.execute()))
        if not self.equi_keys:
            yield from self._nested_loop(build)
            return

        # Emptiness must be checked before the key expressions are evaluated: with no
        # rows there are no columns to evaluate against. See FAILURES.md #2.
        if build.num_rows == 0:
            yield from self._probe_against_empty_build(build)
            return

        build_key_columns = [evaluate(right, build) for _left, right in self.equi_keys]
        encoder = KeyEncoder(build_key_columns)
        # A NULL join key matches nothing in SQL, so those rows are removed from the
        # build side entirely rather than being given a code that could collide.
        build_codes = np.where(encoder.build_has_null, -1, encoder.build_codes)
        keep = build_codes >= 0
        build_rows = np.flatnonzero(keep)
        unique_codes, order_within, offsets, counts = _grouped_codes(build_codes[keep])
        lookup = pd.Index(unique_codes)

        null_build = _null_batch_like(build)
        emitted = False
        last_probe: Batch | None = None

        for probe in self.left.execute():
            last_probe = probe
            probe_key_columns = [evaluate(left, probe) for left, _right in self.equi_keys]
            probe_codes, probe_null = encoder.encode(probe_key_columns)
            probe_codes = np.where(probe_null, -1, probe_codes)

            group = lookup.get_indexer(probe_codes)  # hash probe, vectorised
            group = np.where(probe_codes < 0, -1, group)
            match_counts = np.where(group >= 0, counts[np.maximum(group, 0)], 0)

            matched_rows = np.flatnonzero(match_counts > 0)
            if matched_rows.size:
                per_row = match_counts[matched_rows]
                probe_index = np.repeat(matched_rows, per_row)
                positions = _ragged_expand(offsets[group[matched_rows]], per_row)
                build_index = build_rows[order_within[positions]]
                joined = _combine(probe.take(probe_index), build.take(build_index))
                joined, survivors = self._apply_residual(joined, probe_index)
            else:
                joined = Batch.empty()
                survivors = np.zeros(0, dtype=np.int64)

            if joined.num_rows:
                emitted = True
                for chunk in split_batch(joined, self.batch_size):
                    yield self._count(chunk)

            if self.join_type is JoinType.LEFT:
                unmatched = np.ones(probe.num_rows, dtype=bool)
                unmatched[survivors] = False
                index = np.flatnonzero(unmatched)
                if index.size:
                    emitted = True
                    padded = _combine(probe.take(index), _repeat_nulls(null_build, index.size))
                    for chunk in split_batch(padded, self.batch_size):
                        yield self._count(chunk)

        if not emitted and last_probe is not None:
            yield _combine(empty_like(last_probe), empty_like(build))

    def _apply_residual(
        self, joined: Batch, probe_index: np.ndarray
    ) -> tuple[Batch, np.ndarray]:
        """Filter matched rows by the non-equi part of ON, returning surviving probe rows.

        The surviving set matters for LEFT joins: a probe row whose every match fails the
        residual must still be emitted once, padded with NULLs.
        """
        mask = np.ones(joined.num_rows, dtype=bool)
        for predicate in self.residual:
            mask &= truth_mask(evaluate(predicate, joined))
        if mask.all():
            return joined, probe_index
        return joined.filter(mask), probe_index[mask]

    def _probe_against_empty_build(self, build: Batch) -> Iterator[Batch]:
        """Nothing on the right: INNER yields nothing, LEFT yields every left row padded."""
        if self.join_type is JoinType.INNER:
            return  # nothing on the right can match, so the left is never read
        null_build = _null_batch_like(build)
        emitted = False
        last_probe: Batch | None = None
        for probe in self.left.execute():
            last_probe = probe
            padded = _combine(probe, _repeat_nulls(null_build, probe.num_rows))
            if padded.num_rows:
                emitted = True
                yield self._count(padded)
        if not emitted and last_probe is not None:
            yield _combine(empty_like(last_probe), empty_like(build))

    def _nested_loop(self, build: Batch) -> Iterator[Batch]:
        """Cartesian product, reachable only when the ON clause has no equality.

        The parser makes ``FROM a, b`` illegal and the verifier rejects joins with no
        equi-predicate, so this path is not reachable from a verified LLM query. It
        exists so the engine is honest about what an explicit ``ON 1 = 1`` means.
        """
        for probe in self.left.execute():
            if probe.num_rows == 0 or build.num_rows == 0:
                continue
            probe_index = np.repeat(np.arange(probe.num_rows), build.num_rows)
            build_index = np.tile(np.arange(build.num_rows), probe.num_rows)
            joined = _combine(probe.take(probe_index), build.take(build_index))
            joined, _ = self._apply_residual(joined, probe_index)
            for chunk in split_batch(joined, self.batch_size):
                yield self._count(chunk)


def _combine(left: Batch, right: Batch) -> Batch:
    """Horizontally concatenate two equal-length batches."""
    if left.num_rows != right.num_rows:
        raise ExecutionError("cannot combine batches of different lengths")
    merged = dict(left.columns)
    merged.update(right.columns)
    return Batch(merged, left.num_rows)


def _null_batch_like(template: Batch) -> Batch:
    """A one-row batch with the same columns as ``template``, all NULL."""
    columns = {
        key: Column(np.zeros(1, dtype=col.values.dtype), np.zeros(1, dtype=bool))
        if col.values.dtype != object
        else Column(np.array([None], dtype=object), np.zeros(1, dtype=bool))
        for key, col in template.columns.items()
    }
    return Batch(columns, 1)


def _repeat_nulls(null_row: Batch, count: int) -> Batch:
    index = np.zeros(count, dtype=np.int64)
    return null_row.take(index)


# --------------------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------------------


class AggregateOperator(Operator):
    """Hash aggregation. Blocking: consumes its entire input before emitting anything.

    Grouping is done in one pass over the concatenated input rather than incrementally,
    because an incremental hash aggregate in Python needs a dict lookup per row. One
    materialisation plus a vectorised ``np.bincount`` beats that comfortably at every
    size this engine targets, at the cost of holding the grouping and argument columns
    in memory. That trade is stated in the README.
    """

    def __init__(
        self,
        child: Operator,
        group_keys: Sequence[Expression],
        aggregates: Sequence[FunctionCall],
    ) -> None:
        self.child = child
        self.group_keys = list(group_keys)
        self.aggregates = list(aggregates)
        self.rows_emitted = 0

    def execute(self) -> Iterator[Batch]:
        source = concat_batches(list(self.child.execute()))

        if not self.group_keys:
            yield from self._single_group(source)
            return

        if source.num_rows == 0:
            return  # GROUP BY over no rows produces no groups at all

        key_columns = [evaluate(key, source) for key in self.group_keys]
        # GROUP BY treats NULLs as equal to each other, so the encoder's NULL code is
        # used as an ordinary group rather than being filtered out.
        encoder = KeyEncoder(key_columns)
        codes = encoder.build_codes
        _unique, first_occurrence, inverse = np.unique(
            codes, return_index=True, return_inverse=True
        )
        inverse = inverse.reshape(-1)
        group_count = int(first_occurrence.shape[0])

        output: dict[str, Column] = {}
        for key, column in zip(self.group_keys, key_columns):
            output[str(key)] = column.take(first_occurrence)
        for aggregate in self.aggregates:
            output[str(aggregate)] = self._compute(aggregate, source, inverse, group_count)

        result = Batch(output, group_count)
        for chunk in split_batch(result):
            yield self._count(chunk)

    def _single_group(self, source: Batch) -> Iterator[Batch]:
        """``SELECT COUNT(*) FROM t`` -- exactly one output row, even over zero input rows."""
        codes = np.zeros(source.num_rows, dtype=np.int64)
        output = {
            str(aggregate): self._compute(aggregate, source, codes, 1)
            for aggregate in self.aggregates
        }
        yield self._count(Batch(output, 1))

    def _compute(
        self, aggregate: FunctionCall, source: Batch, codes: np.ndarray, groups: int
    ) -> Column:
        name = aggregate.name
        argument = aggregate.args[0]

        if isinstance(argument, Star):
            counts = np.bincount(codes, minlength=groups).astype(np.int64)
            return Column(counts, np.ones(groups, dtype=bool))

        column = evaluate(argument, source) if source.num_rows else _empty_column()
        if aggregate.distinct:
            return _distinct_aggregate(name, column, codes, groups)

        valid = column.valid
        if name == "COUNT":
            counts = np.bincount(codes[valid], minlength=groups).astype(np.int64)
            return Column(counts, np.ones(groups, dtype=bool))
        if name in ("SUM", "AVG"):
            return _sum_or_average(name, column, codes, groups)
        if name in ("MIN", "MAX"):
            return _min_or_max(name, column, codes, groups)
        raise ExecutionError(f"unsupported aggregate function {name}")


def _empty_column() -> Column:
    return Column(np.zeros(0, dtype=np.float64), np.zeros(0, dtype=bool))


def _sum_or_average(
    name: str, column: Column, codes: np.ndarray, groups: int
) -> Column:
    if column.values.dtype == object:
        raise ExecutionError(f"{name} is not defined for TEXT columns")
    valid = column.valid
    values = np.where(valid, column.values, 0)
    totals = np.bincount(codes[valid], weights=values[valid], minlength=groups)
    counts = np.bincount(codes[valid], minlength=groups)
    # A group with no non-NULL inputs is NULL, not zero -- SQLite agrees, and a silent
    # zero here would corrupt every average computed over sparse data.
    present = counts > 0
    if name == "AVG":
        with np.errstate(invalid="ignore", divide="ignore"):
            averages = np.where(present, totals / np.maximum(counts, 1), 0.0)
        return Column(averages, present)
    if np.issubdtype(column.values.dtype, np.integer):
        # bincount accumulates in float64. Values here are exact below 2^53, which every
        # integer this engine reads from Parquet or CSV is; rounding restores the type.
        return Column(np.rint(totals).astype(np.int64), present)
    return Column(totals, present)


def _min_or_max(name: str, column: Column, codes: np.ndarray, groups: int) -> Column:
    """Grouped MIN/MAX.

    pandas does the grouping here because ``np.minimum.at`` has no object-array path and
    TEXT columns need lexicographic MIN/MAX. The grouping itself is still one vectorised
    C pass, not a Python loop over groups.
    """
    series = to_masked_series(column)
    grouped = series.groupby(codes, sort=True)
    reduced = grouped.min() if name == "MIN" else grouped.max()
    reduced = reduced.reindex(range(groups))
    return _series_to_column(reduced, column.values.dtype)


def _distinct_aggregate(
    name: str, column: Column, codes: np.ndarray, groups: int
) -> Column:
    """DISTINCT aggregates.

    This is the one aggregate path that is not vectorised: pandas applies the reduction
    per group. DISTINCT aggregates are rare and the group count is small in every query
    this engine is meant for, so the simplicity is worth more than the speed. Flagged
    here rather than hidden.
    """
    series = to_masked_series(column)
    grouped = series.groupby(codes, sort=True)
    if name == "COUNT":
        reduced = grouped.nunique()
        reduced = reduced.reindex(range(groups)).fillna(0).astype(np.int64)
        return Column(reduced.to_numpy(dtype=np.int64), np.ones(groups, dtype=bool))
    if name in ("MIN", "MAX"):
        return _min_or_max(name, column, codes, groups)
    reducer = {
        "SUM": lambda s: s.dropna().unique().sum() if s.notna().any() else None,
        "AVG": lambda s: s.dropna().unique().mean() if s.notna().any() else None,
    }[name]
    reduced = grouped.apply(reducer).reindex(range(groups))
    dtype = np.dtype(np.float64) if name == "AVG" else column.values.dtype
    return _series_to_column(reduced, dtype)


def _series_to_column(series: pd.Series, dtype: np.dtype) -> Column:
    """Convert a pandas result (NaN/None for missing) back into values + validity."""
    values = series.to_numpy(dtype=object)
    valid = np.array([v is not None and v == v for v in values], dtype=bool)
    if dtype == object:
        filled = np.empty(values.shape[0], dtype=object)
        filled[:] = [v if ok else None for v, ok in zip(values, valid)]
        return Column(filled, valid)
    numeric = np.zeros(values.shape[0], dtype=dtype)
    for index, (value, ok) in enumerate(zip(values, valid)):
        if ok:
            numeric[index] = value
    return Column(numeric, valid)


# --------------------------------------------------------------------------------------
# Distinct / Sort
# --------------------------------------------------------------------------------------


class DistinctOperator(Operator):
    """SELECT DISTINCT. Blocking.

    Keeps the first occurrence of each key combination in input order. SQL does not
    specify which duplicate survives, but picking the first makes results reproducible
    between runs, which the differential tests depend on.
    """

    def __init__(self, child: Operator, keys: Sequence[Expression]) -> None:
        self.child = child
        self.keys = list(keys)
        self.rows_emitted = 0

    def execute(self) -> Iterator[Batch]:
        source = concat_batches(list(self.child.execute()))
        if source.num_rows == 0:
            return
        key_columns = [evaluate(key, source) for key in self.keys]
        encoder = KeyEncoder(key_columns)
        _unique, first_occurrence = np.unique(encoder.build_codes, return_index=True)
        keep = np.sort(first_occurrence)
        for chunk in split_batch(source.take(keep)):
            yield self._count(chunk)


class SortOperator(Operator):
    """ORDER BY. Blocking.

    Multi-key sorting uses a single ``np.lexsort`` over integer *ranks* rather than over
    the raw values. Ranking first solves two problems at once: it gives DESC without
    reversing (which would break tie stability), and it gives a total order for TEXT
    columns that ``lexsort`` would otherwise handle only slowly.

    NULL ordering follows SQLite: NULLs first ascending, last descending.
    """

    def __init__(self, child: Operator, keys: Sequence[OrderByItem]) -> None:
        self.child = child
        self.keys = list(keys)
        self.rows_emitted = 0

    def execute(self) -> Iterator[Batch]:
        source = concat_batches(list(self.child.execute()))
        if source.num_rows == 0:
            return
        ranks: list[np.ndarray] = []
        for item in self.keys:
            column = evaluate(item.expr, source)
            ranks.extend(_sort_ranks(column, item.direction))
        # lexsort treats its LAST argument as the primary key, so the key list is
        # reversed; within one ORDER BY item the null flag must outrank the value.
        order = np.lexsort(tuple(reversed(ranks)))
        for chunk in split_batch(source.take(order)):
            yield self._count(chunk)


def _sort_ranks(column: Column, direction: SortDirection) -> list[np.ndarray]:
    """Two ascending sort keys for one ORDER BY item: NULL placement, then value.

    Splitting NULL placement into its own key is what lets DESC be expressed as a
    *negated value* rather than as a reversed sort. Reversing would invert the order of
    tied rows and silently break the stability the multi-key sort depends on.

    NULL placement follows SQLite: first when ascending, last when descending.
    """
    null_flag = (
        column.valid.astype(np.int8)  # 0 = NULL, sorts first
        if direction is SortDirection.ASC
        else (~column.valid).astype(np.int8)  # 1 = NULL, sorts last
    )
    if column.values.dtype == object:
        # TEXT needs an ordinal encoding; factorize(sort=True) gives lexicographic codes.
        codes, _uniques = pd.factorize(to_masked_series(column), sort=True, use_na_sentinel=True)
        values: np.ndarray = codes.astype(np.int64)
    elif column.values.dtype == np.bool_:
        values = column.values.astype(np.int8)
    else:
        values = column.values
    if direction is SortDirection.DESC:
        values = -values
    return [null_flag, values]
