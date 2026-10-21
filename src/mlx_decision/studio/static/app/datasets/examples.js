// A dataset split's examples: a paged table (the state cut to a line, the
// labels, else the label column's values), filtered by a label or a value and
// searched in the states, read a page at a time from GET /data/NAME/examples;
// the example picked shows whole in the inspector.
import { signal } from "@preact/signals";
import { html } from "htm/preact";
import { useEffect, useRef } from "preact/hooks";
import { api } from "../api.js";
import { plural } from "../format.js";
import { modifiedOf } from "../store/data.js";
import { useSummary, valueWords } from "./summary.js";

export const PAGE = 50;

// What the table shows: {name, split, offset, question, label, column, value, search}.
const NONE = { offset: 0, question: null, label: null, column: null, value: null, search: "" };
export const query = signal({ name: null, split: null, ...NONE });
const page = signal(null); // the answer for the query, {name, ..., examples}
const pageError = signal(null);
// The example picked: {name, split, index}; and that example whole, once read.
export const pickedExample = signal(null);
const whole = signal(null);

// The table for another dataset (or split) starts on its first page, unfiltered.
export function showDataset(name, split = null) {
  if (query.value.name === name && query.value.split === split) return;
  query.value = { name, split, ...NONE };
  page.value = null;
  pickedExample.value = null;
}

const change = (fields) => {
  query.value = { ...query.value, offset: 0, ...fields };
};

// A label picked in Questions or the menu: the examples with it; again: all.
export function filterLabel(question, label) {
  const same = query.value.question === question && query.value.label === label;
  change(same ? { question: null, label: null } : { question, label, column: null, value: null });
}

// A value of a label column picked: the examples with it; again: all.
export function filterValue(column, value) {
  const same = query.value.column === column && query.value.value === value;
  change(same ? { column: null, value: null } : { column, value, question: null, label: null });
}

let asked = 0;

async function load(current) {
  const ask = ++asked;
  const params = new URLSearchParams({ offset: current.offset, limit: PAGE });
  if (current.split) params.set("split", current.split);
  if (current.question != null) {
    params.set("question", current.question);
    params.set("label", current.label);
  }
  if (current.column != null) {
    params.set("column", current.column);
    params.set("value", current.value);
  }
  if (current.search.trim()) params.set("search", current.search.trim());
  try {
    const result = await api(`/data/${encodeURIComponent(current.name)}/examples?${params}`);
    if (ask !== asked) return; // a newer query is out
    page.value = { ...result, name: current.name, asked: current.split };
    pageError.value = null;
  } catch (error) {
    if (ask !== asked) return;
    page.value = null;
    pageError.value = error.message;
  }
}

async function loadWhole(picked) {
  whole.value = null;
  if (!picked) return;
  try {
    const query = picked.split ? `?split=${encodeURIComponent(picked.split)}` : "";
    const example = await api(`/data/${encodeURIComponent(picked.name)}/examples/${picked.index}${query}`);
    if (pickedExample.value === picked) whole.value = example;
  } catch (error) {
    if (pickedExample.value === picked) whole.value = { error: error.message };
  }
}

function pick(name, split, index) {
  const picked = { name, split, index };
  pickedExample.value = picked;
  loadWhole(picked);
}

// "tech", or "team: tech" when the file has more than one question; without
// labels (no questions yet), the label column's values.
function LabelChips({ row, many, summary }) {
  const entries = Object.entries(row.labels);
  if (entries.length === 0) {
    const values = Object.entries(row.values ?? {});
    if (values.length === 0) return html`<span class="no-label">no labels</span>`;
    return values.map(([column, value]) => {
      const words = valueWords(summary, column, value);
      return html`<span class="lab value" key=${column} title=${`${column}: ${words}`}>${words}</span>`;
    });
  }
  return entries.map(
    ([question, label]) => html`<span class="lab" key=${question} title=${`${question}: ${label}`}>
      ${many && html`<i>${question}</i>`}${label}</span>`,
  );
}

function Filter({ summary }) {
  const current = query.value;
  let value = "";
  if (current.question != null) value = JSON.stringify(["q", current.question, current.label]);
  else if (current.column != null) value = JSON.stringify(["v", current.column, current.value]);
  const columns = Object.entries(summary.values ?? {});
  return html`
    <select class="ex-filter" aria-label="Label" value=${value}
            onChange=${(event) => {
              const picked = event.target.value;
              if (!picked) change({ question: null, label: null, column: null, value: null });
              else {
                const [kind, key, label] = JSON.parse(picked);
                if (kind === "q") filterLabel(key, label);
                else filterValue(key, label);
              }
            }}>
      <option value="">${summary.questions.length ? "All labels" : "All values"}</option>
      ${summary.questions.map((question) => {
        const counts = new Map(question.labels);
        const options = (question.options ?? question.labels.map(([label]) => label)).filter((label) => counts.get(label));
        const many = summary.questions.length > 1;
        const items = options.map(
          (label) => html`<option value=${JSON.stringify(["q", question.id, label])}>${label} (${counts.get(label).toLocaleString("en")})</option>`,
        );
        return many ? html`<optgroup label=${question.id}>${items}</optgroup>` : items;
      })}
      ${columns.map(
        ([column, values]) => html`<optgroup label=${`${column} values`}>
          ${values.filter(([, count]) => count).map(
            ([item, count]) => html`<option value=${JSON.stringify(["v", column, item])}>${valueWords(summary, column, item)} (${count.toLocaleString("en")})</option>`,
          )}
        </optgroup>`,
      )}
    </select>
  `;
}

function Search() {
  const field = useRef(null);
  const timer = useRef(null);
  const name = query.value.name;
  // Another dataset: the field empties with the query.
  useEffect(() => {
    if (field.current && field.current.value !== query.value.search) field.current.value = query.value.search;
  }, [name]);
  return html`
    <input ref=${field} class="ex-search" type="search" placeholder="Search the states" aria-label="Search the states"
           spellcheck=${false} defaultValue=${query.value.search}
           onInput=${(event) => {
             const text = event.target.value;
             clearTimeout(timer.current);
             timer.current = setTimeout(() => change({ search: text }), 200);
           }} />
  `;
}

function Pager({ result }) {
  const { offset, matched } = result;
  const first = matched ? offset + 1 : 0;
  const last = Math.min(offset + PAGE, matched);
  const filtered = matched !== result.total;
  return html`
    <div class="ex-pager">
      <span class="num">${matched ? `${first.toLocaleString("en")}–${last.toLocaleString("en")} of ${matched.toLocaleString("en")}` : "None"}${
        filtered ? ` (of ${plural(result.total, "example")})` : ""
      }</span>
      <button class="btn sm" aria-label="Previous page" disabled=${offset === 0}
              onClick=${() => (query.value = { ...query.value, offset: Math.max(0, offset - PAGE) })}>Previous</button>
      <button class="btn sm" aria-label="Next page" disabled=${offset + PAGE >= matched}
              onClick=${() => (query.value = { ...query.value, offset: offset + PAGE })}>Next</button>
    </div>
  `;
}

// The Examples section of the Datasets page's centre.
export function ExampleTable({ summary }) {
  const current = query.value;
  const modified = modifiedOf(current.name);
  useEffect(() => {
    if (current.name) load(current);
  }, [current, modified]);
  const result = page.value?.name === current.name && page.value.asked === current.split ? page.value : null;
  const many = summary.questions.length > 1;
  const same = pickedExample.value?.name === current.name && pickedExample.value.split === current.split;
  const picked = same ? pickedExample.value.index : null;
  // ↑ and ↓ move the pick within the page.
  const onKeyDown = (event) => {
    if (!result || (event.key !== "ArrowDown" && event.key !== "ArrowUp")) return;
    const rows = result.examples;
    const at = rows.findIndex((row) => row.index === picked);
    const next = event.key === "ArrowDown" ? Math.min(rows.length - 1, at + 1) : Math.max(0, at - 1);
    if (rows[next]) {
      event.preventDefault();
      pick(current.name, current.split, rows[next].index);
      event.currentTarget.querySelector(`[data-index="${rows[next].index}"]`)?.focus();
    }
  };
  return html`
    <section class="gsec ex-sec"><div class="gsec-head"><h2>Examples</h2>
        <span class="tag num">${plural(summary.examples, "example")}</span></div>
      <div class="group ex-group">
        <div class="ex-tools"><${Search} /><${Filter} summary=${summary} />${result && html`<${Pager} result=${result} />`}</div>
        ${pageError.value && html`<p class="warn-text">${pageError.value}</p>`}
        <div class="ex-head" aria-hidden="true"><span>Id</span><span>State</span><span>${
          summary.questions.length === 0 ? (Object.keys(summary.values ?? {})[0] ?? "Values") : many ? "Labels" : summary.questions[0].id
        }</span></div>
        <div class="ex-rows" role="listbox" aria-label="Examples" onKeyDown=${onKeyDown}>
          ${result?.examples.map(
            (row) => html`<div class="ex-row" key=${row.index} role="option" tabindex=${row.index === picked || (picked == null && row === result.examples[0]) ? "0" : "-1"}
                 data-index=${row.index} aria-selected=${row.index === picked ? "true" : "false"}
                 onClick=${() => pick(current.name, current.split, row.index)}>
              <span class="ex-id mono" title=${row.id}>${row.id}</span>
              <span class="ex-state" title=${row.state}>${row.state}</span>
              <span class="ex-labels"><${LabelChips} row=${row} many=${many} summary=${summary} /></span>
            </div>`,
          )}
          ${result && result.matched === 0 &&
          html`<div class="ex-none">No example ${current.search.trim() ? html`holds “${current.search.trim()}”` : ""}${
            current.question != null ? html`${current.search.trim() ? " with" : " has"} the label <code>${current.label}</code>` : ""
          }${current.column != null ? html`${current.search.trim() ? " with" : " has"} the value <code>${current.value}</code>` : ""}.</div>`}
        </div>
      </div>
    </section>
  `;
}

// ---------- the inspector ----------

function stateText(state) {
  return typeof state === "string" ? state : JSON.stringify(state, null, 2);
}

// The option a label names, in the question's words (a choice's criterion, a score's level).
function meaning(question, label) {
  const criteria = question?.criteria;
  if (question?.type === "choice" && criteria && !Array.isArray(criteria)) return criteria[label] ?? null;
  if (question?.type === "choice" && Array.isArray(criteria)) return null;
  if (question?.type === "score" && Array.isArray(criteria)) return criteria[Number(label)] ?? null;
  if (question?.type === "noul" && criteria && typeof criteria === "object") return criteria[label] ?? null;
  return null;
}

// ``split``: the split asked for (null: the default one); ``shown``: its name.
export function ExampleInspector({ name, split = null, shown = null }) {
  const summary = useSummary(name, split);
  const current = pickedExample.value;
  const picked = current?.name === name && current.split === (shown ?? split) ? current : null;
  const example = picked ? whole.value : null;
  if (!name) return html`<div class="empty"><b>No dataset picked</b><span>Pick a dataset on the left.</span></div>`;
  return html`
    <div class="idet data-inspector">
      ${!picked
        ? html`<div class="empty"><b>No example picked</b><span>Pick one in the table to see it whole.</span></div>`
        : example?.error
          ? html`<p class="warn-text">${example.error}</p>`
          : example &&
            html`<div class="idet-head"><div><h2 class="mono">${example.id}</h2><p>Example ${(example.index + 1).toLocaleString("en")}${summary?.examples ? ` of ${summary.examples.toLocaleString("en")}` : ""}</p></div></div>
              <section class="igroup"><h4>State</h4><pre class=${`ex-whole${typeof example.state === "string" ? "" : " json"}`}>${stateText(example.state)}</pre></section>
              ${Object.keys(example.values ?? {}).length > 0 &&
              html`<section class="igroup"><h4>Values</h4><dl>
                ${Object.entries(example.values).map(
                  ([column, value]) => html`<div class="kv" key=${column}><dt class="mono">${column}</dt><dd>
                    <code>${value}</code>${summary?.columns?.[column]?.values?.[value]?.name && html`<small>${summary.columns[column].values[value].name}</small>`}
                  </dd></div>`,
                )}
              </dl></section>`}
              ${Object.keys(example.questions).length > 0 &&
              html`<section class="igroup"><h4>Labels</h4><dl>
                ${Object.keys(example.questions).map((id) => {
                  const label = example.labels[id];
                  const question = example.questions[id];
                  const words = label != null ? meaning(question, label) : null;
                  return html`<div class="kv" key=${id}><dt class="mono">${id}</dt><dd>
                    ${label != null ? html`<code>${label}</code>` : html`<span class="no-label">not labelled</span>`}
                    ${words && html`<small>${words}</small>`}
                  </dd></div>`;
                })}
              </dl></section>`}`}
    </div>
  `;
}
