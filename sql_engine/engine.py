"""Public API: the whole pipeline behind one object.

    engine = Engine.from_directory("data")
    result = engine.query("SELECT region, SUM(total) FROM orders GROUP BY region")
    print(engine.explain("SELECT ..."))

Everything the LLM layer and the web UI need is here, and nothing here knows either of
them exists. The engine is usable, testable and defensible on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ast_nodes import SelectStatement
from .binder import Binder, BoundStatement
from .catalog import Catalog
from .estimation import Estimator
from .executor import DEFAULT_BATCH_SIZE, ExecutionStats, ResultSet, run
from .optimiser import Optimiser, OptimiserReport
from .parser import parse
from .plan import LogicalPlan
from .planner import Planner, project_output_keys


@dataclass(slots=True)
class CompiledQuery:
    """Everything produced between SQL text and the first byte of data being read.

    Exposed as a value rather than hidden inside ``query()`` because the web UI shows
    the plan, the verifier inspects the AST, and the benchmark needs to compile once and
    execute many times. Compilation touches no data.
    """

    sql: str
    statement: SelectStatement
    bound: BoundStatement
    logical_plan: LogicalPlan
    optimised_plan: LogicalPlan
    report: OptimiserReport
    estimator: Estimator = field(repr=False)

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.bound.output_names

    @property
    def output_keys(self) -> tuple[str, ...]:
        return project_output_keys(self.bound.statement.select_list)

    @property
    def estimated_rows(self) -> int:
        return self.optimised_plan.estimated_rows(self.estimator)

    def explain(self) -> str:
        """Both plans, before and after optimisation, with row estimates.

        Printing both is the point. "The optimiser pushes predicates down" is a claim;
        two plans side by side where one has the predicate at the scan and a smaller row
        estimate is evidence.
        """
        return (
            "SQL\n"
            f"  {self.sql.strip()}\n\n"
            "Logical plan (before optimisation)\n"
            f"{_indent(self.logical_plan.explain(self.estimator))}\n\n"
            "Logical plan (after optimisation)\n"
            f"{_indent(self.optimised_plan.explain(self.estimator))}\n\n"
            "Optimiser\n"
            f"  {self.report.summary()}\n"
        )


def _indent(text: str) -> str:
    return "\n".join("  " + line for line in text.splitlines())


class Engine:
    """A SELECT-only SQL engine over a fixed catalog of Parquet and CSV files."""

    def __init__(
        self,
        catalog: Catalog,
        *,
        optimiser: Optimiser | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.catalog = catalog
        self.optimiser = optimiser if optimiser is not None else Optimiser()
        self.batch_size = batch_size
        self._binder = Binder(catalog)
        self._planner = Planner(catalog)

    @classmethod
    def from_directory(cls, directory: str | Path, **kwargs: Any) -> Engine:
        return cls(Catalog.from_directory(directory), **kwargs)

    # -- compilation -------------------------------------------------------------

    def compile(self, sql: str) -> CompiledQuery:
        """Parse, bind, plan and optimise ``sql`` without reading any data.

        Raises ParseError / TokeniseError for anything that is not a SELECT in this
        grammar, and BindError / CatalogError for anything that names something the
        catalog does not have. Those two families are the entire verification surface.
        """
        statement = parse(sql)
        bound = self._binder.bind(statement)
        logical = self._planner.plan(bound)
        estimator = Estimator(bound.column_stats)
        optimised, report = self.optimiser.optimise(logical)
        return CompiledQuery(
            sql=sql,
            statement=statement,
            bound=bound,
            logical_plan=logical,
            optimised_plan=optimised,
            report=report,
            estimator=estimator,
        )

    # -- execution ---------------------------------------------------------------

    def query(self, sql: str) -> ResultSet:
        """Compile and execute ``sql``."""
        return self.execute(self.compile(sql))

    def execute(self, compiled: CompiledQuery) -> ResultSet:
        """Execute an already-compiled query."""
        return run(
            compiled.optimised_plan,
            compiled.output_keys,
            compiled.output_names,
            batch_size=self.batch_size,
        )

    def explain(self, sql: str) -> str:
        """Render the before/after plans for ``sql``."""
        return self.compile(sql).explain()

    # -- introspection -----------------------------------------------------------

    def schema_text(self) -> str:
        return self.catalog.describe()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Engine(tables={len(self.catalog)}, batch_size={self.batch_size})"


__all__ = ["Engine", "CompiledQuery", "ExecutionStats", "ResultSet"]
