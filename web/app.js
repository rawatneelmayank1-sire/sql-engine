/* ---------------------------------------------------------------------------
   SQL Engine -- static demo.

   The rendering below is the live server's page script (sql_engine/web/static/
   index.html), kept as close to verbatim as possible. The only change is where the
   data comes from: instead of POSTing to /api/ask and /api/verify, the page looks
   up the exact JSON those endpoints returned when scripts/export_static_demo.py
   ran the real engine. Nothing here computes a result.
--------------------------------------------------------------------------- */
(() => {
  "use strict";

  const DEMO = window.SQL_ENGINE_DEMO || {};

  const STAGE_LABEL = {
    tokenise: "Tokenise",
    parse: "Parse",
    bind: "Bind",
    plan: "Plan",
    verify: "Verify",
    execute: "Execute",
  };

  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  /* --- theme ------------------------------------------------------------- */

  const applyTheme = (value) => {
    if (value === "light" || value === "dark") {
      document.documentElement.setAttribute("data-theme", value);
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
  };

  const readStoredTheme = () => {
    try { return localStorage.getItem("sql-engine-theme"); } catch { return null; }
  };

  applyTheme(readStoredTheme());

  $("theme-toggle").addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const next = current
      ? (current === "dark" ? "light" : "dark")
      : (prefersDark ? "light" : "dark");
    applyTheme(next);
    try { localStorage.setItem("sql-engine-theme", next); } catch { /* private mode */ }
  });

  /* --- tabs --------------------------------------------------------------- */

  const TABS = [["tab-ask", "panel-ask"], ["tab-sql", "panel-sql"], ["tab-schema", "panel-schema"]];
  TABS.forEach(([tabId, panelId]) => {
    $(tabId).addEventListener("click", () => {
      TABS.forEach(([t, p]) => {
        const active = t === tabId;
        $(t).setAttribute("aria-selected", String(active));
        $(p).hidden = !active;
      });
      if (panelId === "panel-ask") $("question").focus();
      if (panelId === "panel-sql") $("sql").focus();
    });
  });

  /* --- rendering ---------------------------------------------------------- */

  let planData = { before: "", after: "", summary: "" };
  let planView = "after";

  const renderRail = (stages, ok) => {
    const rail = $("rail");
    rail.innerHTML = "";
    rail.classList.remove("animate");
    void rail.offsetWidth; // restart the animation
    rail.classList.add("animate");

    stages.forEach((stage) => {
      const node = el("div", "stage");
      node.dataset.status = stage.status;
      node.appendChild(el("span", "node"));
      node.appendChild(el("span", "stage-name", STAGE_LABEL[stage.name] || stage.name));
      node.appendChild(el("span", "stage-readout", stage.readout || ""));
      rail.appendChild(node);
    });

    const stopped = stages.find((s) => s.status === "refused");
    const verdict = $("rail-verdict");
    verdict.className = "rail-verdict " + (ok ? "passed" : "refused");
    verdict.textContent = ok
      ? "Verified and executed"
      : "Halted at " + (STAGE_LABEL[stopped ? stopped.name : "verify"] || "verify");
    $("rail-shell").hidden = false;
    $("live").textContent = verdict.textContent;
  };

  const renderVerdict = (data) => {
    const body = $("verdict-body");
    body.innerHTML = "";

    (data.failures || []).forEach((failure) => {
      const box = el("div", "callout");
      box.appendChild(el("span", "callout-title", failure.category.replace(/_/g, " ")));
      box.appendChild(el("div", "callout-body", failure.message));
      body.appendChild(box);
    });

    (data.repairs || []).forEach((repair) => {
      const box = el("div", "callout info");
      box.appendChild(el("span", "callout-title", "Repaired: " + repair.kind));
      box.appendChild(el("div", "callout-body", repair.detail));
      body.appendChild(box);
    });

    // A check that ran and passed, a check that ran and failed, and a check the
    // pipeline never reached are three different things. Showing the third as "OK"
    // would claim the catalog was consulted for a query that never got past the parser.
    const failedCategories = new Set((data.failures || []).map((f) => f.category));
    const stages = data.stages || [];
    const haltIndex = stages.findIndex((s) => s.status === "refused");

    const stageOf = (name) => stages.findIndex((s) => s.name === name);
    const reached = (name) => {
      if (haltIndex < 0) return true;
      const index = stageOf(name);
      return index >= 0 && index <= haltIndex;
    };

    const checks = [
      ["parse", "Parses as a SELECT under the SQL Engine grammar",
       !failedCategories.has("not_select") && !failedCategories.has("unparseable"),
       "There is no AST node for INSERT, UPDATE, DELETE or DROP, so those cannot be built."],
      ["bind", "Every table and column exists in the catalog",
       !["unknown_table", "unknown_column", "ambiguous_column", "invalid_aggregation"]
         .some((c) => failedCategories.has(c)),
       "Checked against the schema read from the data files themselves."],
      ["verify", "No join without an equality predicate",
       !failedCategories.has("cartesian_product"),
       "A cartesian product would pair every row on the left with every row on the right."],
      ["verify", "A LIMIT is present",
       true,
       (data.repairs || []).some((r) => r.kind === "limit-injected")
         ? "One was injected because the query had none."
         : "The query supplied its own."],
      ["verify", "Estimated scan is within budget",
       !failedCategories.has("scan_too_large"),
       "Measured on the optimised plan, so pruning columns lowers the cost."],
    ];

    checks.forEach(([stage, label, passed, detail]) => {
      const ran = reached(stage);
      const state = !ran ? "skip" : (passed ? "pass" : "fail");
      const mark = { pass: "OK", fail: "NO", skip: "--" }[state];
      const row = el("div", "check " + state);
      row.appendChild(el("span", "check-mark", mark));
      const text = el("div", "check-text");
      text.appendChild(el("div", "check-label", label));
      text.appendChild(el("div", "check-detail",
        ran ? detail : "Not reached; the query was refused earlier."));
      row.appendChild(text);
      body.appendChild(row);
    });
  };

  const renderAttempts = (attempts) => {
    const panel = $("attempts-panel");
    if (!attempts || attempts.length <= 1) { panel.hidden = true; return; }
    panel.hidden = false;
    $("attempts-note").textContent =
      attempts.length + " model calls, " + (attempts.length - 1) + " caused by the verifier";

    const body = $("attempts-body");
    body.innerHTML = "";
    attempts.forEach((attempt) => {
      const box = el("div", "attempt " + (attempt.ok ? "pass" : "fail"));
      const head = el("div", "attempt-head");
      head.appendChild(el("span", null, "Attempt " + (attempt.index + 1)));
      head.appendChild(el("span", "attempt-status", attempt.ok ? "accepted" : "refused"));
      box.appendChild(head);

      const inner = el("div", "attempt-body");
      inner.appendChild(el("pre", "code", attempt.sql));
      (attempt.failures || []).forEach((failure) => {
        inner.appendChild(el("div", "attempt-reason", failure.message));
      });
      box.appendChild(inner);
      body.appendChild(box);
    });
  };

  const renderPlan = () => {
    const text = planView === "after" ? planData.after : planData.before;
    $("plan-panel").hidden = !text;
    $("plan-tree").textContent = text || "";
    $("plan-summary").textContent = planData.summary || "";
    $("plan-after").setAttribute("aria-pressed", String(planView === "after"));
    $("plan-before").setAttribute("aria-pressed", String(planView === "before"));
  };

  $("plan-after").addEventListener("click", () => { planView = "after"; renderPlan(); });
  $("plan-before").addEventListener("click", () => { planView = "before"; renderPlan(); });

  const renderRows = (result) => {
    const panel = $("result-panel");
    if (!result || !result.columns.length) { panel.hidden = true; return; }
    panel.hidden = false;

    const parts = [result.total.toLocaleString() + " row" + (result.total === 1 ? "" : "s")];
    if (result.stats) {
      parts.push(result.stats.rows_scanned.toLocaleString() + " scanned");
      parts.push(result.stats.columns_read + " columns read");
    }
    if (result.truncated) parts.push("showing first " + result.rows.length);
    $("result-note").textContent = parts.join("  ·  ");

    const table = $("result-table");
    table.innerHTML = "";
    const thead = el("thead");
    const headRow = el("tr");
    result.columns.forEach((name) => headRow.appendChild(el("th", null, name)));
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = el("tbody");
    result.rows.forEach((row) => {
      const tr = el("tr");
      row.forEach((value) => {
        const isNull = value === null;
        const isNumber = typeof value === "number";
        const td = el("td", isNull ? "null" : (isNumber ? "num" : null),
                      isNull ? "NULL" : String(value));
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
  };

  const render = (data) => {
    $("idle").hidden = true;
    $("output").hidden = false;

    renderRail(data.stages || [], data.ok);
    $("final-sql").textContent = data.sql || data.original_sql || "";
    $("sql-note").textContent = data.elapsed_ms
      ? data.elapsed_ms.toFixed(1) + " ms"
      : "";
    renderVerdict(data);
    renderAttempts(data.attempts);
    planData = data.plan || { before: "", after: "", summary: "" };
    renderPlan();
    renderRows(data.result);
  };

  // `command`, when given, is shown as code on its own line under the message.
  const renderMessage = (title, message, command) => {
    $("idle").hidden = true;
    $("output").hidden = false;
    $("rail-shell").hidden = true;
    $("final-sql").textContent = "";
    $("sql-note").textContent = "";
    const body = $("verdict-body");
    body.innerHTML = "";
    const box = el("div", "callout info");
    box.appendChild(el("span", "callout-title", title));
    const text = el("div", "callout-body", message);
    if (command) {
      text.appendChild(document.createElement("br"));
      text.appendChild(el("code", null, command));
    }
    box.appendChild(text);
    body.appendChild(box);
    $("attempts-panel").hidden = true;
    $("plan-panel").hidden = true;
    $("result-panel").hidden = true;
    $("live").textContent = message;
  };

  /* --- evaluation: the verified answer against the hand-written gold query -- */

  // Verification is structural; it cannot tell whether a query answers the question
  // that was asked. The evaluation can, so each recorded answer says how it scored.
  // Kept visibly separate from the verification checks, and placed first so a green
  // rail is never read as "correct".
  const renderEvaluation = (question, data, wrongCount, total) => {
    if (!data.ok) return;
    const body = $("verdict-body");
    const box = el("div", question.gold_match ? "callout match" : "callout");
    if (question.gold_match) {
      box.appendChild(el("span", "callout-title", "Evaluation · matches the gold query"));
      box.appendChild(el("div", "callout-body",
        "Returns the same rows as the hand-written gold query. This comparison is how "
        + "the answers were scored; it is not one of the verifier's checks."));
    } else {
      box.appendChild(el("span", "callout-title", "Evaluation · verified, but wrong rows"));
      box.appendChild(el("div", "callout-body",
        "Passed every check below and still returns different rows from the "
        + "hand-written gold query. Structural verification cannot catch a semantic "
        + "error: " + wrongCount + " of the " + total + " recorded answers are like this. "
        + "The gold query:"));
      box.appendChild(el("pre", "code", question.gold_sql));
    }
    body.insertBefore(box, body.firstChild);
  };

  /* --- lookups (in place of requests) ----------------------------------------- */

  // Mirrors _normalise() in sql_engine/web/app.py, so a typed question resolves exactly
  // as it would against the live server's recorded-question lookup.
  const normalise = (question) => question.toLowerCase().split(/\s+/).filter(Boolean)
    .join(" ").replace(/\?+$/, "").trim();

  const normaliseSql = (sql) => sql.replace(/\r\n?/g, "\n").trim();

  const questions = DEMO.questions || [];
  const answers = DEMO.answers || {};
  const queries = DEMO.queries || {};
  const byKey = new Map(questions.map((q) => [q.key, q]));
  const wrongCount = questions.filter((q) => !q.gold_match).length;
  const modelName = (DEMO.meta && DEMO.meta.recorded_from) || "a recorded model";

  const LOCAL_ASK =
    'pip install -e ".[web,llm]" && python data/generate.py && python -m sql_engine.cli serve';
  const LOCAL_VERIFY =
    'pip install -e . && python data/generate.py && python -m sql_engine.cli verify "<your SQL>"';

  $("ask-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const question = $("question").value.trim();
    if (!question) return;
    const entry = byKey.get(normalise(question));
    const data = entry && answers[entry.id];
    if (!data) {
      renderMessage(
        "Not one of the recorded questions",
        "This static demo can only replay the " + questions.length + " questions listed "
          + "below; it cannot call a model. To ask your own, clone the repository, set "
          + "GROQ_API_KEY, and run:",
        LOCAL_ASK,
      );
      return;
    }
    render(data);
    renderEvaluation(entry, data, wrongCount, questions.length);
  });

  $("sql-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const sql = normaliseSql($("sql").value);
    if (!sql) return;
    const data = queries[sql];
    if (!data) {
      renderMessage(
        "Not one of the precomputed queries",
        "This static demo shows the engine's verdicts for the queries listed below. "
          + "To verify your own SQL, clone the repository and run (no API key needed):",
        LOCAL_VERIFY,
      );
      return;
    }
    render(data);
  });

  /* --- boot --------------------------------------------------------------- */

  const schema = DEMO.schema;
  if (!schema) {
    $("model-pill").textContent = "data missing";
    $("idle").textContent = "The demo data did not load. Serve the web/ folder over HTTP or "
      + "open index.html from disk, keeping the data/ folder next to it.";
    return;
  }

  const pill = $("model-pill");
  pill.textContent = "recorded · no live model";
  pill.title = "LLM answers were recorded from " + modelName;

  $("modeline-text").textContent =
    "Every result was computed by the real engine; LLM answers are recorded responses from "
    + modelName + ".";

  $("ask-hint").textContent =
    "Answers replay responses recorded from " + modelName + ". Pick one of the "
    + questions.length + " recorded questions, or type one of them.";

  const addQuestion = (container, question) => {
    const button = el("button", "suggestion", question);
    button.type = "button";
    button.addEventListener("click", () => {
      $("question").value = question;
      $("ask-form").requestSubmit();
    });
    container.appendChild(button);
    return button;
  };

  // The live page shows the first twelve; the rest sit one click away.
  const asks = $("ask-suggestions");
  const shown = new Set(schema.examples);
  schema.examples.forEach((question) => addQuestion(asks, question));
  const rest = questions.map((q) => q.question).filter((q) => !shown.has(q));
  if (rest.length) {
    const more = el("button", "suggestion more", "+" + rest.length + " more questions");
    more.type = "button";
    more.addEventListener("click", () => {
      more.remove();
      const added = rest.map((question) => addQuestion(asks, question));
      if (added.length) added[0].focus();
    });
    asks.appendChild(more);
  }

  const addQuery = (container, example) => {
    const refusal = example.kind === "refusal";
    const button = el("button", "suggestion " + (refusal ? "danger" : "sql"), example.sql);
    button.type = "button";
    button.title = example.why;
    button.addEventListener("click", () => {
      $("sql").value = example.sql;
      $("sql-form").requestSubmit();
    });
    container.appendChild(button);
  };

  (DEMO.query_examples || []).forEach((example) => {
    addQuery(example.kind === "refusal" ? $("sql-suggestions") : $("sql-valid-suggestions"),
             example);
  });

  const grid = $("schema-grid");
  schema.tables.forEach((table) => {
    const card = el("div", "schema-table");
    card.appendChild(el("div", "schema-name", table.name));
    card.appendChild(el("div", "schema-meta",
      table.format + "  ·  " + table.rows.toLocaleString() + " rows"));
    table.columns.forEach((column) => {
      const row = el("div", "schema-col");
      row.appendChild(document.createTextNode(column.name + " "));
      row.appendChild(el("span", null, column.type));
      card.appendChild(row);
    });
    grid.appendChild(card);
  });

  // A shareable link runs its query on load. Useful for pointing someone at a
  // specific refusal ("open this and watch it stop at BIND") rather than describing it.
  const params = new URLSearchParams(window.location.search);
  const preloadSql = params.get("sql");
  const preloadQuestion = params.get("q");
  if (preloadSql !== null) {
    $("tab-sql").click();
    $("sql").value = preloadSql;
    $("sql-form").requestSubmit();
  } else if (preloadQuestion) {
    $("question").value = preloadQuestion;
    $("ask-form").requestSubmit();
  }
})();
