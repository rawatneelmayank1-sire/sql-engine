"""sql-engine -- a SELECT-only SQL engine with a verified LLM text-to-SQL front-end.

The engine implements only ``SELECT``. Not "rejects everything else" -- implements only
``SELECT``. There is no AST node for INSERT, no parser production for DROP, and no code
path that writes to a data file. That is the whole security argument, and it is a
property of the type system rather than of a filter that could be bypassed.
"""

from __future__ import annotations

from .catalog import Catalog, ColumnSchema, DataType, TableSchema
from .engine import CompiledQuery, Engine, ResultSet
from .errors import (
    BindError,
    CatalogError,
    ExecutionError,
    SQLEngineError,
    ParseError,
    TokeniseError,
    VerificationError,
)
from .optimiser import Optimiser
from .parser import parse

__version__ = "0.1.0"

__all__ = [
    "BindError",
    "Catalog",
    "CatalogError",
    "ColumnSchema",
    "CompiledQuery",
    "DataType",
    "Engine",
    "ExecutionError",
    "SQLEngineError",
    "Optimiser",
    "ParseError",
    "ResultSet",
    "TableSchema",
    "TokeniseError",
    "VerificationError",
    "parse",
]
