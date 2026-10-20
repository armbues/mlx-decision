// A dataset's examples: a paged table (the state cut to a line, the labels),
// filtered by a label and searched in the states, read a page at a time from
// GET /data/NAME/examples; the example picked shows whole in the inspector.
import { signal } from "@preact/signals";
import { html } from "htm/preact";
import { useEffect, useRef } from "preact/hooks";
import { api } from "../api.js";
import { plural } from "../format.js";
import { modifiedOf } from "../store/data.js";
import { FileGroup, useSummary } from "./summary.js";

export const PAGE = 50;

// What the table shows: {name, offset, question, label, search}.
export const query = signal({ name: null, offset: 0, question: null, label: null, search: "" });
const page = signal(null); // the answer for the query, {name, ..., examples}
const pageError = signal(null);
// The example picked: {name, index}; and that example whole, once read.
export const pickedExample = signal(null);
const whole = signal(null);

// The table for another dataset starts on its first page, unfiltered.
export function showDataset(name) {
  if (query.value.name === name) return;
  query.value = { name, offset: 0, question: null, label: null, search: "" };
  page.value = null;
  pickedExample.value = null;
}

const change = (fields) => {
  query.value = { ...query.value, offset: 0, ...fields };
};

// A label picked in Questions or the menu: the examples with it; again: all.
export function filterLabel(question, label) {
  const same = query.value.question === question && query.value.label === label;
  change(same ? { question: null, label: null } : { question, label });
}

let asked = 0;

async function load(current) {
  const ask = ++asked;
  const params = new URLSearchParams({ offset: current.offset, limit: PAGE });
  if (current.question != null) {
    params.set("question", current.question);
    params.set("label", current.label);
  }
  if (current.search.trim()) params.set("search", current.search.trim());
  try {
    const result = await api(`/data/${encodeURIComponent(current.name)}/examples?${params}`);
    if (ask !== asked) return; // a newer query is out
    page.value = { ...result, name: current.name };
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
    const example = await api(`/data/${encodeURIComponent(picked.name)}/examples/${picked.index}`);
    if (pickedExample.value === picked) whole.value = example;
  } catch (error) {
    if (pickedExample.value === picked) whole.value = { error: error.message };
  }
}

function pick(name, index) {
  const picked = { name, index };
  pickedExample.value = picked;
  loadWhole(picked);
}

// "tech", or "team: tech" when the file has more than one question.
function LabelChips({ labels, many }) {
  const entries = Object.entries(labels);
  if (entries.length === 0) return html`<span class="no-label">no labels</span>`;
  return entries.map(
    ([question, label]) => html`<span class="lab" key=${question} title=${`${question}: ${label}`}>
      ${many && html`<i>${question}</i>`}${label}</span>`,
  );
}

function Filter({ summary }) {
  const current = query.value;
  const value = current.question == null ? "" : JSON.stringify([current.question, current.label]);
  return html`
    <select class="ex-filter" aria-label="Label" value=${value}
            onChange=${(event) => {
              const picked = event.target.value;
              if (!picked) change({ question: null, label: null });
              else {
                const [question, label] = JSON.parse(picked);
                change({ question, label });
              }
            }}>
      <option value="">All labels</option>
      ${summary.questions.map((question) => {
        const counts = new Map(question.labels);
        const options = (question.options ?? question.labels.map(([label]) => label)).filter((label) => counts.get(label));
        const many = summary.questions.length > 1;
        const items = options.map(
          (label) => html`<option value=${JSON.stringify([question.id, label])}>${label} (${counts.get(label).toLocaleString("en")})</option>`,
        );
        return many ? html`<optgroup label=${question.id}>${items}</optgroup>` : items;
      })}
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
  const result = page.value?.name === current.name ? page.value : null;
  const many = summary.questions.length > 1;
  const picked = pickedExample.value?.name === current.name ? pickedExample.value.index : null;
  // ↑ and ↓ move the pick within the page.
  const onKeyDown = (event) => {
    if (!result || (event.key !== "ArrowDown" && event.key !== "ArrowUp")) return;
    const rows = result.examples;
    const at = rows.findIndex((row) => row.index === picked);
    const next = event.key === "ArrowDown" ? Math.min(rows.length - 1, at + 1) : Math.max(0, at - 1);
    if (rows[next]) {
      event.preventDefault();
      pick(current.name, rows[next].index);
      event.currentTarget.querySelector(`[data-index="${rows[next].index}"]`)?.focus();
    }
  };
  return html`
    <section class="gsec ex-sec"><div class="gsec-head"><h2>Examples</h2>
        <span class="tag num">${plural(summary.examples, "example")}</span></div>
      <div class="group ex-group">
        <div class="ex-tools"><${Search} /><${Filter} summary=${summary} />${result && html`<${Pager} result=${result} />`}</div>
        ${pageError.value && html`<p class="warn-text">${pageError.value}</p>`}
        <div class="ex-head" aria-hidden="true"><span>Id</span><span>State</span><span>${many ? "Labels" : summary.questions[0]?.id ?? "Label"}</span></div>
        <div class="ex-rows" role="listbox" aria-label="Examples" onKeyDown=${onKeyDown}>
          ${result?.examples.map(
            (row) => html`<div class="ex-row" key=${row.index} role="option" tabindex=${row.index === picked || (picked == null && row === result.examples[0]) ? "0" : "-1"}
                 data-index=${row.index} aria-selected=${row.index === picked ? "true" : "false"}
                 onClick=${() => pick(current.name, row.index)}>
              <span class="ex-id mono" title=${row.id}>${row.id}</span>
              <span class="ex-state" title=${row.state}>${row.state}</span>
              <span class="ex-labels"><${LabelChips} labels=${row.labels} many=${many} /></span>
            </div>`,
          )}
          ${result && result.matched === 0 &&
          html`<div class="ex-none">No example ${current.search.trim() ? html`holds “${current.search.trim()}”` : ""}${
            current.question != null ? html`${current.search.trim() ? " with" : " has"} the label <code>${current.label}</code>` : ""
          }.</div>`}
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

export function ExampleInspector({ name }) {
  const summary = useSummary(name);
  const picked = pickedExample.value?.name === name ? pickedExample.value : null;
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
              <section class="igroup"><h4>Labels</h4><dl>
                ${Object.keys(example.questions).map((id) => {
                  const label = example.labels[id];
                  const question = example.questions[id];
                  const words = label != null ? meaning(question, label) : null;
                  return html`<div class="kv" key=${id}><dt class="mono">${id}</dt><dd>
                    ${label != null ? html`<code>${label}</code>` : html`<span class="no-label">not labelled</span>`}
                    ${words && html`<small>${words}</small>`}
                  </dd></div>`;
                })}
              </dl></section>`}
      ${summary && !summary.error && html`<${FileGroup} summary=${summary} />`}
    </div>
  `;
}
