"""Table catalog: name -> file, schema and statistics.

The catalog is the second half of the safety argument. The parser guarantees a query is
a SELECT; the catalog guarantees every table and column it names actually exists. An LLM
that invents ``customers.lifetime_value`` fails here, before a single byte is read,
with an error naming the column and listing the ones that do exist -- which is exactly
the feedback the repair loop needs.

Statistics are deliberately crude (row count, per-column distinct-value count, min/max
for numerics). They exist to make ``EXPLAIN`` show *estimated* row counts that move when
the optimiser pushes a predicate down. They are not a serious cost model and the README
says so.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .errors import CatalogError


class DataType(str, Enum):
    """The five storage types sql-engine understands.

    There is no DATE type on purpose: dates are stored as ISO-8601 TEXT, exactly as
    SQLite stores them. Keeping the type lattice identical to SQLite's removes a whole
    class of spurious differences from the differential test suite -- any mismatch that
    remains is a real bug in this engine rather than a modelling disagreement.
    """

    INTEGER = "INTEGER"
    REAL = "REAL"
    TEXT = "TEXT"
    BOOLEAN = "BOOLEAN"

    @property
    def numpy_dtype(self) -> np.dtype:
        return _NUMPY_DTYPES[self]

    @property
    def is_numeric(self) -> bool:
        return self in (DataType.INTEGER, DataType.REAL)


_NUMPY_DTYPES: dict[DataType, np.dtype] = {
    DataType.INTEGER: np.dtype(np.int64),
    DataType.REAL: np.dtype(np.float64),
    DataType.TEXT: np.dtype(object),
    DataType.BOOLEAN: np.dtype(bool),
}

_ARROW_TO_DATATYPE: list[tuple[Any, DataType]] = [
    (pa.types.is_boolean, DataType.BOOLEAN),
    (pa.types.is_integer, DataType.INTEGER),
    (pa.types.is_floating, DataType.REAL),
    (pa.types.is_string, DataType.TEXT),
    (pa.types.is_large_string, DataType.TEXT),
]


@dataclass(frozen=True, slots=True)
class ColumnStats:
    """Per-column statistics used only for row-count estimation in EXPLAIN."""

    distinct_count: int
    null_count: int
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True, slots=True)
class ColumnSchema:
    """One column of a table."""

    name: str
    dtype: DataType
    stats: ColumnStats | None = None


@dataclass(frozen=True, slots=True)
class TableSchema:
    """A table: where its data lives, what columns it has, how big it is."""

    name: str
    path: Path
    file_format: str  # "parquet" | "csv"
    columns: tuple[ColumnSchema, ...]
    row_count: int
    #: Uncompressed on-disk size, used by the verifier's scan-size guard.
    byte_size: int = 0

    _by_name: dict[str, ColumnSchema] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        # Column lookup is case-insensitive to match SQLite, but the *stored* name keeps
        # its original case so result headers read naturally.
        object.__setattr__(self, "_by_name", {c.name.lower(): c for c in self.columns})

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def has_column(self, name: str) -> bool:
        return name.lower() in self._by_name

    def column(self, name: str) -> ColumnSchema:
        try:
            return self._by_name[name.lower()]
        except KeyError:
            raise CatalogError(
                f"column {name!r} does not exist on table {self.name!r}; "
                f"available columns: {', '.join(self.column_names)}"
            ) from None

    def average_row_bytes(self) -> float:
        """Rough bytes-per-row, for the scan-size guard. Returns 0 when unknown."""
        if self.row_count == 0:
            return 0.0
        return self.byte_size / self.row_count


class Catalog:
    """An immutable-by-convention map of table name -> :class:`TableSchema`."""

    def __init__(self, tables: Iterable[TableSchema] = ()) -> None:
        self._tables: dict[str, TableSchema] = {}
        for table in tables:
            self.register(table)

    def register(self, table: TableSchema) -> None:
        key = table.name.lower()
        if key in self._tables:
            raise CatalogError(f"table {table.name!r} is already registered")
        self._tables[key] = table

    @property
    def table_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self._tables.values())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.lower() in self._tables

    def __iter__(self):
        return iter(self._tables.values())

    def __len__(self) -> int:
        return len(self._tables)

    def table(self, name: str) -> TableSchema:
        try:
            return self._tables[name.lower()]
        except KeyError:
            raise CatalogError(
                f"table {name!r} does not exist; "
                f"available tables: {', '.join(sorted(self.table_names))}"
            ) from None

    # -- construction ------------------------------------------------------------

    @classmethod
    def from_directory(cls, directory: str | Path) -> Catalog:
        """Build a catalog by inspecting every .parquet and .csv file in ``directory``.

        Schemas are read from the files themselves rather than from a hand-written
        manifest, so the catalog cannot drift out of sync with the data -- an
        out-of-sync catalog would silently invalidate every "the column exists"
        guarantee the verifier makes.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise CatalogError(f"catalog directory {directory} does not exist")
        tables: list[TableSchema] = []
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() == ".parquet":
                tables.append(_schema_from_parquet(path))
            elif path.suffix.lower() == ".csv":
                tables.append(_schema_from_csv(path))
        if not tables:
            raise CatalogError(f"no .parquet or .csv files found in {directory}")
        return cls(tables)

    def describe(self) -> str:
        """Render the schema as compact DDL-ish text for the LLM prompt.

        Deliberately not real ``CREATE TABLE`` DDL: showing a model DDL it can imitate
        invites it to emit DDL back. The prompt shows a shape it cannot copy verbatim.
        """
        lines: list[str] = []
        for table in sorted(self._tables.values(), key=lambda t: t.name):
            cols = ", ".join(f"{c.name} {c.dtype.value}" for c in table.columns)
            lines.append(f"{table.name}({cols})  -- {table.row_count} rows")
        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialise the catalog (used by the web UI and by fixture recording)."""
        payload = {
            table.name: {
                "path": str(table.path),
                "format": table.file_format,
                "row_count": table.row_count,
                "columns": [
                    {"name": c.name, "type": c.dtype.value} for c in table.columns
                ],
            }
            for table in sorted(self._tables.values(), key=lambda t: t.name)
        }
        return json.dumps(payload, indent=2)


def _arrow_type_to_datatype(arrow_type: pa.DataType, column: str, path: Path) -> DataType:
    for predicate, dtype in _ARROW_TO_DATATYPE:
        if predicate(arrow_type):
            return dtype
    raise CatalogError(
        f"column {column!r} in {path.name} has unsupported type {arrow_type}; "
        "sql-engine supports INTEGER, REAL, TEXT and BOOLEAN only"
    )


def _stats_from_array(array: pa.ChunkedArray, dtype: DataType) -> ColumnStats:
    """Compute the small statistics bundle used for cardinality estimation."""
    null_count = int(array.null_count)
    try:
        distinct = len(array.unique())
    except pa.ArrowNotImplementedError:  # pragma: no cover - defensive
        distinct = max(1, len(array) // 2)
    minimum = maximum = None
    if dtype.is_numeric and len(array) > null_count:
        values = array.drop_null().to_numpy(zero_copy_only=False)
        if values.size:
            minimum = float(np.min(values))
            maximum = float(np.max(values))
    return ColumnStats(
        distinct_count=max(1, distinct), null_count=null_count, minimum=minimum, maximum=maximum
    )


def _schema_from_parquet(path: Path) -> TableSchema:
    table = pq.read_table(path)
    columns: list[ColumnSchema] = []
    for name in table.column_names:
        dtype = _arrow_type_to_datatype(table.schema.field(name).type, name, path)
        columns.append(ColumnSchema(name, dtype, _stats_from_array(table.column(name), dtype)))
    return TableSchema(
        name=path.stem,
        path=path,
        file_format="parquet",
        columns=tuple(columns),
        row_count=table.num_rows,
        byte_size=int(table.nbytes),
    )


def _schema_from_csv(path: Path) -> TableSchema:
    """Infer a CSV schema by scanning the file once.

    pyarrow's CSV reader would be faster, but its type inference differs from the
    conversion the executor applies at read time; inferring here with the same rules the
    reader uses keeps the catalog honest about what will actually come back.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise CatalogError(f"{path.name} is empty") from None
        rows = list(reader)

    columns: list[ColumnSchema] = []
    for index, name in enumerate(header):
        raw = [row[index] if index < len(row) else "" for row in rows]
        dtype = infer_csv_type(raw)
        non_null = [v for v in raw if v != ""]
        stats = ColumnStats(
            distinct_count=max(1, len(set(non_null))),
            null_count=len(raw) - len(non_null),
        )
        if dtype.is_numeric and non_null:
            numbers = [float(v) for v in non_null]
            stats = ColumnStats(
                distinct_count=stats.distinct_count,
                null_count=stats.null_count,
                minimum=min(numbers),
                maximum=max(numbers),
            )
        columns.append(ColumnSchema(name, dtype, stats))

    return TableSchema(
        name=path.stem,
        path=path,
        file_format="csv",
        columns=tuple(columns),
        row_count=len(rows),
        byte_size=path.stat().st_size,
    )


def infer_csv_type(values: Iterable[str]) -> DataType:
    """Infer a :class:`DataType` from raw CSV strings.

    Order matters: everything that parses as an int is INTEGER, otherwise everything
    that parses as a float is REAL, otherwise TEXT. An empty column is TEXT rather than
    an error so a fully-null column does not break catalog construction.
    """
    saw_value = False
    all_int = True
    all_real = True
    for value in values:
        if value == "":
            continue
        saw_value = True
        try:
            int(value)
        except ValueError:
            all_int = False
            try:
                float(value)
            except ValueError:
                all_real = False
                break
    if not saw_value:
        return DataType.TEXT
    if all_int:
        return DataType.INTEGER
    if all_real:
        return DataType.REAL
    return DataType.TEXT


def catalog_from_mapping(spec: Mapping[str, Mapping[str, Any]]) -> Catalog:
    """Build a catalog from an explicit mapping. Used by tests that need a fake schema
    without touching the filesystem."""
    tables = []
    for name, entry in spec.items():
        columns = tuple(
            ColumnSchema(col["name"], DataType(col["type"])) for col in entry["columns"]
        )
        tables.append(
            TableSchema(
                name=name,
                path=Path(entry.get("path", f"{name}.parquet")),
                file_format=entry.get("format", "parquet"),
                columns=columns,
                row_count=int(entry.get("row_count", 0)),
                byte_size=int(entry.get("byte_size", 0)),
            )
        )
    return Catalog(tables)
