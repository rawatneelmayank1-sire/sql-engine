"""Columnar batch: the unit of data every operator passes to the next.

Two decisions here carry the whole executor.

**Columns, not rows.** A batch is a dict of name -> :class:`Column`, each backed by a
NumPy array. Operators call NumPy kernels over 4096 values at a time. The alternative --
a Python loop over rows -- pays the interpreter's ~50-100ns per element on *every*
comparison, which in a Python engine is the entire cost. Nothing else in this project
matters as much for throughput.

**An explicit validity mask, not NaN.** NULL is tracked in a separate boolean array
rather than encoded as NaN, because NaN cannot represent a NULL integer or a NULL
string, and because SQL's three-valued logic needs to distinguish "false" from
"unknown". This is the same choice Apache Arrow makes, for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pyarrow as pa

from ..catalog import DataType
from ..errors import ExecutionError

#: Rows per batch. 4096 is the usual sweet spot: large enough that the per-call Python
#: overhead of a NumPy kernel is amortised to nothing, small enough that a batch and its
#: intermediates stay in L2 cache. Measured in bench/results.md.
DEFAULT_BATCH_SIZE = 4096


@dataclass(frozen=True, slots=True)
class Column:
    """A vector of values plus a validity mask. ``valid[i] is False`` means SQL NULL.

    Values under invalid positions are *unspecified* -- they hold whatever the fill
    value was. Every operation must therefore compute the mask independently rather than
    inferring it from the data.
    """

    values: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        if self.values.shape != self.valid.shape:
            raise ExecutionError(
                f"column values ({self.values.shape}) and validity mask "
                f"({self.valid.shape}) disagree on length"
            )

    def __len__(self) -> int:
        return int(self.values.shape[0])

    @property
    def has_nulls(self) -> bool:
        return not bool(self.valid.all())

    @classmethod
    def from_values(cls, values: np.ndarray, valid: np.ndarray | None = None) -> Column:
        if valid is None:
            valid = np.ones(values.shape[0], dtype=bool)
        return cls(values, valid)

    @classmethod
    def constant(cls, value: Any, length: int) -> Column:
        """Broadcast a scalar (or NULL) to ``length`` rows."""
        if value is None:
            return cls(
                np.zeros(length, dtype=np.float64), np.zeros(length, dtype=bool)
            )
        if isinstance(value, bool):
            return cls(np.full(length, value, dtype=bool), np.ones(length, dtype=bool))
        if isinstance(value, (int, np.integer)):
            return cls(
                np.full(length, value, dtype=np.int64), np.ones(length, dtype=bool)
            )
        if isinstance(value, (float, np.floating)):
            return cls(
                np.full(length, value, dtype=np.float64), np.ones(length, dtype=bool)
            )
        array = np.empty(length, dtype=object)
        array[:] = value
        return cls(array, np.ones(length, dtype=bool))

    def take(self, indices: np.ndarray) -> Column:
        """Gather rows by position. Used by joins and sorts."""
        return Column(self.values[indices], self.valid[indices])

    def filter(self, mask: np.ndarray) -> Column:
        """Keep rows where ``mask`` is True."""
        return Column(self.values[mask], self.valid[mask])

    def with_nulls_at(self, mask: np.ndarray) -> Column:
        """Return a copy with the positions in ``mask`` marked NULL."""
        return Column(self.values, self.valid & ~mask)

    def to_pylist(self) -> list[Any]:
        """Convert to Python objects with None for NULL, for result serialisation."""
        out: list[Any] = []
        values = self.values
        is_float = np.issubdtype(values.dtype, np.floating)
        is_int = np.issubdtype(values.dtype, np.integer)
        is_bool = values.dtype == np.bool_
        for index, ok in enumerate(self.valid):
            if not ok:
                out.append(None)
            elif is_bool:
                out.append(bool(values[index]))
            elif is_int:
                out.append(int(values[index]))
            elif is_float:
                out.append(float(values[index]))
            else:
                out.append(values[index])
        return out

    def as_nullable_object(self) -> np.ndarray:
        """A single object array with ``None`` in the NULL positions.

        Needed wherever pandas does the heavy lifting (grouped MIN/MAX, LIKE), because
        pandas has no concept of a separate validity mask.
        """
        out = self.values.astype(object)
        out[~self.valid] = None
        return out


@dataclass(frozen=True, slots=True)
class Batch:
    """A horizontal slice of a relation: named columns of equal length."""

    columns: dict[str, Column]
    num_rows: int

    @classmethod
    def empty(cls) -> Batch:
        return cls({}, 0)

    def __len__(self) -> int:
        return self.num_rows

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.columns

    def get(self, key: str) -> Column | None:
        return self.columns.get(key)

    def with_columns(self, extra: Mapping[str, Column]) -> Batch:
        """A new batch with ``extra`` added (or replacing) the existing columns."""
        merged = dict(self.columns)
        merged.update(extra)
        return Batch(merged, self.num_rows)

    def select(self, keys: Sequence[str]) -> Batch:
        """Keep only ``keys``, in that order."""
        return Batch({key: self.columns[key] for key in keys}, self.num_rows)

    def filter(self, mask: np.ndarray) -> Batch:
        """Apply a boolean row mask to every column at once."""
        kept = int(mask.sum())
        if kept == self.num_rows:
            return self
        return Batch({k: c.filter(mask) for k, c in self.columns.items()}, kept)

    def take(self, indices: np.ndarray) -> Batch:
        return Batch(
            {k: c.take(indices) for k, c in self.columns.items()}, int(indices.shape[0])
        )

    def slice(self, start: int, length: int) -> Batch:
        stop = min(self.num_rows, start + length)
        if start == 0 and stop == self.num_rows:
            return self
        index = np.arange(start, stop)
        return self.take(index)


def concat_batches(batches: Sequence[Batch]) -> Batch:
    """Vertically concatenate batches produced by the same operator.

    Blocking operators (sort, aggregate, join build side) need their whole input in one
    place; this is where that materialisation happens, and it is the only place, so the
    memory cost of a blocking operator is easy to point at.
    """
    if not batches:
        return Batch.empty()
    non_empty = [b for b in batches if b.num_rows > 0]
    if not non_empty:
        # Every input was empty. Return one of them rather than a schema-less blank:
        # the column names and dtypes still have to reach the operator above.
        return batches[0]
    batches = non_empty
    if len(batches) == 1:
        return batches[0]
    keys = list(batches[0].columns.keys())
    merged: dict[str, Column] = {}
    for key in keys:
        values = np.concatenate([b.columns[key].values for b in batches])
        valid = np.concatenate([b.columns[key].valid for b in batches])
        merged[key] = Column(values, valid)
    return Batch(merged, sum(b.num_rows for b in batches))


def split_batch(batch: Batch, batch_size: int = DEFAULT_BATCH_SIZE) -> Iterator[Batch]:
    """Re-chunk a materialised batch back into pipeline-sized pieces."""
    if batch.num_rows == 0:
        yield batch
        return
    for start in range(0, batch.num_rows, batch_size):
        yield batch.slice(start, batch_size)


_ARROW_FILL: dict[DataType, Any] = {
    DataType.INTEGER: 0,
    DataType.REAL: 0.0,
    DataType.TEXT: "",
    DataType.BOOLEAN: False,
}

_TARGET_DTYPE: dict[DataType, np.dtype] = {
    DataType.INTEGER: np.dtype(np.int64),
    DataType.REAL: np.dtype(np.float64),
    DataType.TEXT: np.dtype(object),
    DataType.BOOLEAN: np.dtype(bool),
}


def column_from_arrow(array: pa.Array | pa.ChunkedArray, dtype: DataType) -> Column:
    """Convert an Arrow array to a :class:`Column`.

    Arrow's ``to_numpy`` promotes a nullable integer column to float64 with NaN, which
    would silently turn INTEGER into REAL and break both the type system and the
    comparison with SQLite. So the mask is extracted first and the nulls are filled with
    a type-appropriate dummy *before* conversion.
    """
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    valid = np.asarray(array.is_valid())
    filled = array.fill_null(_ARROW_FILL[dtype])
    values = filled.to_numpy(zero_copy_only=False)
    target = _TARGET_DTYPE[dtype]
    if values.dtype != target:
        values = values.astype(target)
    return Column(values, valid)
