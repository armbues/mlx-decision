// A batch on the Batches page: what it ran, how far it got, per question its
// answers and, with labels, its metrics, confusion matrix and reliability;
// then its rows in a table the Studio pages and filters. The inspector shows
// the row picked, whole, with each answer's probabilities.
import { signal } from "@preact/signals";
import { html } from "htm/preact";
import { useEffect, useRef } from "preact/hooks";
import { api } from "../api.js";
import { Confusion, Reliability } from "../components/charts.js";
import { MenuButton, MenuItem, MenuSeparator } from "../components/menu.js";
import { closeSheet, openSheet } from "../components/sheet.js";
import { plural, shellWord } from "../format.js";
import { href } from "../router.js";
import {
  batchByName,
  batchProgressText,
  deleteBatch,
  resultsChanged,
  runAgain,
  runRest,
  stopBatch,
} from "../store/batches.js";
import { models, status } from "../store/status.js";
import { toast } from "../store/ui.js";
import { openSaveDataset } from "./start.js";

export const PAGE = 100;
const UNSURE = 0.6;
// An answer's probability as the Playground's cards show it.
const probability = (value) => `${Math.round((value ?? 0) * 100)}%`;
const METRICS = [
  ["accuracy", "Accuracy", true],
  ["macro_f1", "Macro-F1", true],
  ["ece", "Calibration error", false],
  ["mae", "Mean error (levels)", false],
];

// The open batch's results ({name, batch, results}), read again as it runs.
const opened = signal(null);
const openedError = signal(null);
// The table's filters and page, for the batch named (``created`` tells a new
// batch of a deleted one's name from it).
const NONE = { question: null, answer: null, gold: null, show: "all", search: "", offset: 0 };
export const table = signal({ name: null, created: null, ...NONE });
// The row to pick when the page an arrow key asked for arrives: "first" or "last".
let pickOnPage = null;
const rowsPage = signal(null);
// The row picked: {name, row}, and that row whole.
export const pickedRow = signal(null);
const wholeRow = signal(null);

const encoded = (name) => encodeURIComponent(name);
const percent = (value) => `${(value * 100).toFixed(1)}%`;

async function loadResults(name) {
  try {
    const answer = await api(`/batches/${encoded(name)}`);
    if (table.value.name !== name) return;
    opened.value = { name, ...answer };
    openedError.value = null;
  } catch (error) {
    if (table.value.name === name) openedError.value = error.message;
  }
}

async function loadRows(query) {
  const params = new URLSearchParams({ offset: query.offset, limit: PAGE });
  if (query.question) params.set("question", query.question);
  if (query.answer != null) params.set("answer", query.answer);
  if (query.gold != null) params.set("gold", query.gold);
  if (query.show === "wrong") params.set("wrong", "true");
  if (query.show === "unsure") params.set("unsure", UNSURE);
  if (query.show === "errors") params.set("errors", "true");
  if (query.search.trim()) params.set("search", query.search.trim());
  try {
    const answer = await api(`/batches/${encoded(query.name)}/rows?${params}`);
    if (table.value !== query) return;
    rowsPage.value = { query, ...answer };
    if (pickOnPage && answer.rows.length) {
      const row = pickOnPage === "first" ? answer.rows[0] : answer.rows[answer.rows.length - 1];
      pickOnPage = null;
      pick(query.name, row.row);
      requestAnimationFrame(() => document.querySelector(`.batch-rows [data-row="${row.row}"]`)?.focus());
    }
  } catch (error) {
    if (table.value === query) rowsPage.value = { query, matching: 0, offset: 0, rows: [], error: error.message };
  }
}

function filter(fields) {
  table.value = { ...table.value, offset: 0, ...fields };
}

export function openBatch(name, created = null) {
  const current = table.value;
  if (current.name === name && (created == null || current.created == null || current.created === created)) {
    if (created != null && current.created == null) table.value = { ...current, created };
    return;
  }
  table.value = { name, created, ...NONE };
  opened.value = null;
  openedError.value = null;
  rowsPage.value = null;
  pickedRow.value = null;
  wholeRow.value = null;
}

async function pick(name, row) {
  pickedRow.value = { name, row };
  wholeRow.value = null;
  try {
    const answer = await api(`/batches/${encoded(name)}/rows?row=${row}`);
    if (pickedRow.value?.name === name && pickedRow.value.row === row) wholeRow.value = answer.row;
  } catch (error) {
    if (pickedRow.value?.name === name && pickedRow.value.row === row) wholeRow.value = { error: error.message };
  }
}

// ---------- the head ----------

function Head({ batch }) {
  const running = batch.state === "running";
  const rest = !running && batch.answered < batch.total && batch.state !== "ready";
  const save = (format) => exportFile(batch.name, format);
  const questions = batch.questions_from ? html`the set <a href=${href("playground", batch.questions_from)}>${batch.questions_from}</a>` : "its own questions";
  const rows = batch.sample
    ? `a sample of ${batch.sample.toLocaleString("en")} of ${batch.rows_in_split.toLocaleString("en")} rows (seed ${batch.seed})`
    : plural(batch.total, "row");
  return html`
    <div class="pane-head">
      <div>
        <h1 class="mono">${batch.name}</h1>
        <p>
          <a href=${href("datasets", batch.dataset)}>${batch.dataset}</a> · <code>${batch.split}</code> · ${rows} · on <b>${batch.model}</b> · ${questions}
        </p>
      </div>
      <div class="actions">
        ${running && html`<button class="btn" disabled=${batch.stop_requested} onClick=${() => stopBatch(batch.name)}>${batch.stop_requested ? "Stopping…" : "Stop"}</button>`}
        ${rest && html`<button class="btn primary" onClick=${() => runRest(batch.name)} title="Answer the rows not answered yet">Run the Rest</button>`}
        ${!running && html`<button class="btn" onClick=${() => runAgain(batch.name)} title="A new batch of the same rows, questions and model">Run Again</button>`}
        <${MenuButton} id="batch-export" className="btn" label="Export" menuStyle="top:32px; right:0; width:220px">
          <${MenuItem} onSelect=${() => save("csv")}>CSV<//>
          <${MenuItem} onSelect=${() => save("jsonl")}>JSON Lines<//>
        <//>
        <${MenuButton} id="batch-more" className="btn" ariaLabel=${`More actions for ${batch.name}`}
          label=${html`<svg viewBox="0 0 20 20" fill="currentColor" width="15" height="15"><circle cx="5" cy="10" r="1.5" /><circle cx="10" cy="10" r="1.5" /><circle cx="15" cy="10" r="1.5" /></svg>`}
          menuStyle="top:32px; right:0; width:240px">
          <${MenuItem} disabled=${running || batch.answered === 0} onSelect=${() => openSaveDataset(batch)}>Save Answers as a Dataset…<//>
          <${MenuItem} onSelect=${() => copyCommand(batch)}>Copy Command<//>
          <${MenuSeparator} />
          <${MenuItem} disabled=${running} onSelect=${() => confirmDelete(batch)}>Delete…<//>
        <//>
      </div>
    </div>
  `;
}

// The export as a file: fetched first, so a failure is a toast, not a page of JSON.
async function exportFile(name, format) {
  try {
    const response = await fetch(`/studio/api/batches/${encoded(name)}/export?format=${format}`);
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      throw new Error(body?.error?.message ?? `the Studio answered ${response.status}`);
    }
    const link = document.createElement("a");
    link.href = URL.createObjectURL(await response.blob());
    link.download = `${name}.${format}`;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 10_000);
  } catch (error) {
    toast(`Could not export ${name}: ${error.message}`);
  }
}

export function confirmDelete(batch) {
  openSheet({
    title: `Delete the batch ${batch.name}?`,
    purpose: `Its folder batches/${batch.name}/ goes, with its ${plural(batch.answered, "answer")}. The dataset stays.`,
    actions: [
      {
        label: "Delete",
        primary: true,
        danger: true,
        onSelect: () => {
          closeSheet();
          deleteBatch(batch.name);
        },
      },
    ],
  });
}

// The command that answers the same rows from a terminal (the workspace as the folder).
export function command(batch) {
  const model = models.value?.find((item) => item.name === batch.model);
  const short = model?.cli?.short ?? ["-m", batch.model];
  const folder = `batches/${batch.name}`;
  return ["mlx-decision", "run", ...short, "-q", `${folder}/questions.json`, "--states", `${folder}/rows.jsonl`, "--json"]
    .map(shellWord)
    .join(" ");
}

function copyCommand(batch) {
  navigator.clipboard.writeText(command(batch)).then(
    () => toast("Copied the command"),
    () => toast("Could not copy the command"),
  );
}

// What a screen reader hears when a batch changes state (not each row).
const STATES = { running: "Running", stopped: "Stopped", done: "Done", failed: "Failed" };

function Progress({ batch }) {
  const fraction = batch.total ? batch.answered / batch.total : 0;
  const running = batch.state === "running";
  return html`
    <div class=${`batch-progress ${batch.state}`}>
      <div class="bar" role="progressbar" aria-label="Rows answered" aria-valuemin="0" aria-valuemax=${batch.total} aria-valuenow=${batch.answered}>
        <i style=${`width:${(fraction * 100).toFixed(1)}%`}></i></div>
      <span class="num">${batchProgressText(batch)}${batch.errors ? ` · ${plural(batch.errors, "error")}` : ""}</span>
      ${!running && batch.message && html`<span class="warn-text">${batch.message}</span>`}
      <span class="sr-only" role="status">${STATES[batch.state] ?? ""}</span>
    </div>
  `;
}

function Terminal({ batch }) {
  const workspace = status.value?.workspace;
  return html`
    <section class="gsec batch-cli"><div class="gsec-head"><h2>From a terminal</h2>
        <span class="tag">The same rows, questions and model</span></div>
      <div class="group"><pre class="code-line">${workspace ? `cd ${shellWord(workspace)}\n` : ""}${command(batch)}</pre></div>
    </section>
  `;
}

// ---------- the results per question ----------

function Question({ id, item, batch }) {
  const current = table.value;
  const total = Object.values(item.answers).reduce((sum, count) => sum + count, 0) || 1;
  const show = (fields) => filter({ question: id, answer: null, gold: null, show: "all", ...fields });
  return html`
    <section class="gsec batch-q" data-q=${id}>
      <div class="gsec-head"><h2 class="mono">${id}</h2><span class="tag">${item.type}${item.labelled ? " · with labels" : ""}</span></div>
      <div class="group bq-group">
        <div class="bq-answers" role="group" aria-label=${`Answers to ${id}`}>
          ${Object.entries(item.answers).map(
            ([answer, count]) => html`<button class=${`bq-answer${current.question === id && current.answer === answer && current.gold == null ? " on" : ""}`}
                key=${answer} disabled=${!count} onClick=${() => show({ answer })} title=${`Show the rows answered ${answer}`}
                aria-label=${`${answer}: ${count}; show these rows`}>
              <code>${answer}</code><span class="bq-bar"><i style=${`width:${((count / total) * 100).toFixed(1)}%`}></i></span><span class="num">${count.toLocaleString("en")}</span>
            </button>`,
          )}
        </div>
        <div class="bq-counts">
          <button class="btn sm" disabled=${!item.unsure} onClick=${() => show({ show: "unsure" })}
            title=${`Answers with a confidence below ${UNSURE}`}>${plural(item.unsure, "unsure answer")}</button>
          ${item.labelled && html`<button class="btn sm" disabled=${!item.wrong} onClick=${() => show({ show: "wrong" })}>${item.wrong.toLocaleString("en")} wrong</button>`}
        </div>
        ${item.labelled && !item.metrics && html`<p class="bq-note">Accuracy and the rest show as its labelled rows are answered.</p>`}
        ${item.metrics &&
        html`<div class="bq-metrics">
          <dl class="metric-row">
            ${METRICS.filter(([key]) => item.metrics[key] != null).map(
              ([key, label]) => html`<div class="metric" key=${key}><dt>${label}</dt><dd class="num">${key === "mae" ? item.metrics[key].toFixed(2) : percent(item.metrics[key])}</dd></div>`,
            )}
            <div class="metric"><dt>Labelled</dt><dd class="num">${item.metrics.n.toLocaleString("en")}</dd></div>
          </dl>
          <div class="bq-charts">
            <${Confusion} question=${item} onCell=${(gold, predicted) => show({ gold, answer: predicted === "refused" ? null : predicted, show: predicted === "refused" ? "errors" : "all" })} />
            <${Reliability} bins=${item.reliability} />
          </div>
        </div>`}
        ${!item.labelled && html`<p class="bq-note">No labels for this question: its answers only. A dataset whose labels answer it gives accuracy and the rest.</p>`}
      </div>
    </section>
  `;
}

// ---------- the table ----------

function Search() {
  const field = useRef(null);
  const timer = useRef(null);
  const name = table.value.name;
  useEffect(() => {
    if (field.current) field.current.value = table.value.search;
  }, [name]);
  useEffect(() => () => clearTimeout(timer.current), [name]); // typed for another batch: dropped
  return html`<input ref=${field} class="ex-search" type="search" placeholder="Search the states" aria-label="Search the states"
    spellcheck=${false} defaultValue=${table.value.search}
    onInput=${(event) => {
      const text = event.target.value;
      clearTimeout(timer.current);
      timer.current = setTimeout(() => filter({ search: text }), 200);
    }} />`;
}

function Pager({ result }) {
  const { offset, matching } = result;
  const first = matching ? offset + 1 : 0;
  const last = Math.min(offset + PAGE, matching);
  return html`
    <div class="ex-pager">
      <span class="num">${matching ? `${first.toLocaleString("en")}–${last.toLocaleString("en")} of ${matching.toLocaleString("en")}` : "None"}</span>
      <button class="btn sm" aria-label="Previous page" disabled=${offset === 0}
              onClick=${() => (table.value = { ...table.value, offset: Math.max(0, offset - PAGE) })}>Previous</button>
      <button class="btn sm" aria-label="Next page" disabled=${offset + PAGE >= matching}
              onClick=${() => (table.value = { ...table.value, offset: offset + PAGE })}>Next</button>
    </div>
  `;
}

const SHOW = [
  ["all", "All"],
  ["wrong", "Wrong"],
  ["unsure", "Unsure"],
  ["errors", "Errors"],
];

function Rows({ batch, results }) {
  const current = table.value;
  const questions = Object.keys(results.questions);
  const question = current.question ?? questions[0];
  const item = results.questions[question];
  const labelled = item?.labelled;
  const result = rowsPage.value?.query === current ? rowsPage.value : null;
  const picked = pickedRow.value?.name === batch.name ? pickedRow.value.row : null;
  // The row Tab reaches: the picked one when it is on this page, else the first.
  const focusable = result?.rows.some((row) => row.row === picked) ? picked : (result?.rows[0]?.row ?? null);
  const onKeyDown = (event) => {
    if (!result || (event.key !== "ArrowDown" && event.key !== "ArrowUp")) return;
    const rows = result.rows;
    const at = rows.findIndex((row) => row.row === picked);
    const down = event.key === "ArrowDown";
    // Past a page's edge: the next or previous page, its first or last row.
    if (down && at === rows.length - 1 && result.offset + PAGE < result.matching) {
      event.preventDefault();
      pickOnPage = "first";
      table.value = { ...table.value, offset: result.offset + PAGE };
      return;
    }
    if (!down && at === 0 && result.offset > 0) {
      event.preventDefault();
      pickOnPage = "last";
      table.value = { ...table.value, offset: Math.max(0, result.offset - PAGE) };
      return;
    }
    const next = down ? Math.min(rows.length - 1, at + 1) : Math.max(0, at - 1);
    if (rows[next]) {
      event.preventDefault();
      pick(batch.name, rows[next].row);
      event.currentTarget.querySelector(`[data-row="${rows[next].row}"]`)?.focus();
    }
  };
  return html`
    <section class="gsec ex-sec batch-rows"><div class="gsec-head"><h2>Rows</h2>
        <span class="tag num">${plural(batch.answered, "row")} answered</span></div>
      <div class="group ex-group">
        <div class="ex-tools">
          <${Search} />
          ${questions.length > 1 &&
          html`<select aria-label="Question" value=${question} onChange=${(event) => {
            const next = event.target.value;
            const show = current.show === "wrong" && !results.questions[next]?.labelled ? "all" : current.show;
            filter({ question: next, answer: null, gold: null, show });
          }}>
            ${questions.map((id) => html`<option value=${id}>${id}</option>`)}
          </select>`}
          <select aria-label="Answer" value=${current.answer ?? ""} onChange=${(event) => filter({ question, answer: event.target.value || null })}>
            <option value="">Any answer</option>
            ${Object.keys(item?.answers ?? {}).map((answer) => html`<option value=${answer}>${answer}</option>`)}
          </select>
          <div class="seg" role="tablist" aria-label="Show">
            ${SHOW.filter(([key]) => key !== "wrong" || labelled).map(
              ([key, label]) => html`<button role="tab" aria-selected=${current.show === key} aria-pressed=${current.show === key}
                onClick=${() => filter({ question, show: key })}>${label}</button>`,
            )}
          </div>
          ${current.gold != null && html`<span class="chip">expected <code>${current.gold}</code>
            <button aria-label="Any expected answer" onClick=${() => filter({ gold: null })}>×</button></span>`}
          ${result && html`<${Pager} result=${result} />`}
        </div>
        ${result?.error && html`<p class="warn-text">${result.error}</p>`}
        <div class=${`ex-head batch-head${labelled ? " labelled" : ""}`} aria-hidden="true">
          <span>Id</span><span>State</span><span>${questions.length > 1 ? "Answers" : html`<span class="mono">${question}</span>`}</span>${labelled && html`<span>Expected <span class="mono">${questions.length > 1 ? question : ""}</span></span>`}
        </div>
        <div class="ex-rows" role="listbox" aria-label="Rows" onKeyDown=${onKeyDown}>
          ${result?.rows.map((row) => {
            const answer = row.answers[question];
            const expected = row.labels[question];
            const right = expected != null && answer ? answer.answer === expected : null;
            return html`<div class=${`ex-row batch-row${labelled ? " labelled" : ""}`} key=${row.row} role="option" data-row=${row.row}
                tabindex=${row.row === focusable ? "0" : "-1"}
                aria-selected=${row.row === picked ? "true" : "false"} onClick=${() => pick(batch.name, row.row)}>
              <span class="ex-id mono" title=${row.id}>${row.id}</span>
              <span class="ex-state" title=${row.state}>${row.state}</span>
              <span class="batch-answer">${
                row.error
                  ? html`<span class="lab err" title=${row.error.message}>error</span>`
                  : [question, ...questions.filter((id) => id !== question)]
                      .filter((id) => row.answers[id])
                      .map((id) => {
                        const one = row.answers[id];
                        const many = questions.length > 1;
                        return html`<span key=${id} class=${`chip-answer${id === question ? " main" : ""}`} title=${`${id}: ${one.answer} ${probability(one.probability)}, confidence ${one.confidence.toFixed(2)}${one.confidence < UNSURE ? `, below ${UNSURE}: unsure` : ""}`}>
                          ${many && html`<i>${id}</i>`}<span class=${`lab${one.confidence < UNSURE ? " unsure" : ""}`}>${one.answer}</span><small class="num">${probability(one.probability)}</small>
                        </span>`;
                      })
              }</span>
              ${labelled && html`<span class=${`batch-expected${right === false ? " wrong" : right ? " right" : ""}`}>${
                expected != null ? html`${right === false ? "✗ " : right ? "✓ " : ""}<code>${expected}</code>` : html`<span class="no-label">none</span>`
              }</span>`}
            </div>`;
          })}
          ${result && result.matching === 0 && html`<div class="ex-none">No row ${batch.answered ? "passes these filters" : "answered yet"}.</div>`}
        </div>
      </div>
    </section>
  `;
}

// ---------- the centre ----------

export function BatchView({ name }) {
  const batch = batchByName(name);
  openBatch(name, batch?.created ?? null);
  const answered = batch?.answered ?? 0;
  const running = batch?.state === "running";
  const changes = resultsChanged.value;
  const last = useRef(0);
  // The results again: on open, when the batch ends, and while it runs at most once a second.
  useEffect(() => {
    const wait = running ? Math.max(0, 1000 - (Date.now() - last.current)) : 0;
    const timer = setTimeout(() => {
      last.current = Date.now();
      loadResults(name);
    }, wait);
    return () => clearTimeout(timer);
  }, [name, running, changes, running ? answered : -1]);
  const current = table.value;
  useEffect(() => {
    if (current.name === name) loadRows(current);
  }, [current, opened.value]);
  if (!batch) return html`<div class="empty"><b>No batch named ${name}</b><span>It may have been deleted. Pick another on the left.</span></div>`;
  const results = opened.value?.name === name ? opened.value.results : null;
  return html`
    <${Head} batch=${batch} />
    <${Progress} batch=${batch} />
    ${openedError.value && html`<p class="warn-text">${openedError.value}</p>`}
    ${results && Object.entries(results.questions).map(([id, item]) => html`<${Question} key=${id} id=${id} item=${item} batch=${batch} />`)}
    ${results && html`<${Rows} batch=${batch} results=${results} />`}
    <${Terminal} batch=${batch} />
  `;
}

// ---------- the inspector ----------

function stateText(state) {
  return typeof state === "string" ? state : JSON.stringify(state, null, 2);
}

function Probabilities({ answer }) {
  const entries = answer.type === "noul" ? [["true", answer.noul], ["false", 1 - answer.noul]] : Object.entries(answer.probabilities ?? {});
  return html`<div class="probs">${entries.map(
    ([key, value]) => html`<div class="prob" key=${key}><code>${key}</code><span class="bq-bar"><i style=${`width:${(value * 100).toFixed(1)}%`}></i></span><span class="num">${(value * 100).toFixed(1)}%</span></div>`,
  )}</div>`;
}

export function RowInspector({ name }) {
  const picked = pickedRow.value?.name === name ? pickedRow.value : null;
  const row = picked ? wholeRow.value : null;
  if (!name) return html`<div class="empty"><b>No batch picked</b><span>Pick a batch on the left.</span></div>`;
  if (!picked) return html`<div class="empty"><b>No row picked</b><span>Pick one in the table to see it whole.</span></div>`;
  if (!row) return null;
  if (row.error && !row.id) return html`<p class="warn-text">${row.error}</p>`;
  const answers = row.response?.answers ?? {};
  return html`
    <div class="idet batch-inspector">
      <div class="idet-head"><div><h2 class="mono">${row.id}</h2><p>Row ${(row.row + 1).toLocaleString("en")} of the split${
        row.latency_ms != null ? ` · ${Math.round(row.latency_ms)} ms · ${plural(row.input_tokens, "token")}` : ""
      }${row.truncated ? " · cut to fit" : ""}</p></div></div>
      <section class="igroup"><h4>State</h4><pre class=${`ex-whole${typeof row.state === "string" ? "" : " json"}`}>${stateText(row.state)}</pre></section>
      ${row.error && html`<section class="igroup"><h4>Error</h4><p class="warn-text">${row.error.message}</p></section>`}
      ${Object.entries(answers).map(
        ([id, answer]) => html`<section class="igroup" key=${id}><h4 class="mono">${id}</h4>
          <dl>
            <div class="kv"><dt>Answer</dt><dd><code>${row.answers[id]?.answer}</code> <small>Confidence ${(row.answers[id]?.confidence ?? 0).toFixed(2)}</small></dd></div>
            ${row.labels[id] != null && html`<div class="kv"><dt>Expected</dt><dd><code>${row.labels[id]}</code>${
              row.answers[id]?.answer === row.labels[id] ? html` <small class="right">right</small>` : html` <small class="wrong">wrong</small>`
            }</dd></div>`}
          </dl>
          <${Probabilities} answer=${answer} />
        </section>`,
      )}
    </div>
  `;
}
