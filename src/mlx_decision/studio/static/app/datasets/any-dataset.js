// Another dataset: any Hugging Face dataset, one split read as a preview, with
// the column that becomes the state and the label column whose values the rows
// keep (named from the dataset card when it lists them); the columns apply to
// every split in the table. It runs `data` as a job; the dataset arrives
// without questions, which are written afterwards.
import { computed, signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import { StatusIcon } from "../components/icons.js";
import { plural } from "../format.js";
import { NameAndSeed, SplitTable, freeBase, knowRows, seeded, setSplits, splitState, splitsBody, splitsProblem } from "./splits.js";

export const source = signal("");
const previewed = signal(null); // the preview route's answer
const reading = signal(null); // what is being read, while it is
const failed = signal(null);
const config = signal(null);
const split = signal(null);
const textColumn = signal(null);
const labelColumn = signal(null);
const table = splitState(); // every split's role and sample
const seed = signal("0");
const name = signal("");
const named = signal(false);

// Before another dataset is read; ``all`` also forgets the source, the columns
// and the sample (after a download, so the next one starts afresh).
export function resetAny(all = false) {
  previewed.value = null;
  failed.value = null;
  config.value = null;
  split.value = null;
  named.value = false;
  table.value = {};
  rowsOf.value = {};
  if (!all) return;
  source.value = "";
  textColumn.value = labelColumn.value = null;
  seed.value = "0";
  name.value = "";
}

async function read(fields, what) {
  reading.value = what;
  failed.value = null;
  try {
    const result = await api("/datasets/preview", { method: "POST", body: { source: source.value.trim(), ...fields } });
    const before = previewed.value;
    previewed.value = result;
    config.value = result.config;
    split.value = result.split;
    if (result.splits) {
      // Another config: its own splits, from their defaults.
      const fresh = before?.config !== result.config;
      setSplits(table, result.splits.map((name) => ({ name })), result.split, fresh);
    }
    if (result.rows != null) {
      textColumn.value = result.text;
      labelColumn.value = result.label;
      rowsOf.value = { ...rowsOf.value, [result.split]: result.rows };
      knowRows(table, result.split, result.rows);
    }
  } catch (error) {
    failed.value = error.message;
  } finally {
    reading.value = null;
  }
}

// Rows by split, as far as the previews read them.
const rowsOf = signal({});
const splits = computed(() => (previewed.value?.splits ?? []).map((name) => ({ name, rows: rowsOf.value[name] ?? null })));
const suggested = computed(() => {
  const result = previewed.value;
  if (!result) return "";
  return freeBase(result.source.split("/").pop());
});
const base = computed(() => (named.value ? name.value.trim() : suggested.value));

export function check() {
  resetAny();
  const value = source.value.trim();
  if (!value.startsWith("hf:")) source.value = `hf:${value}`;
  read({}, "Asking the Hub for its configs and splits…");
}

const labelInfo = computed(() => previewed.value?.label_columns?.find((column) => column.name === labelColumn.value) ?? null);
export const anyProblem = computed(() => {
  const result = previewed.value;
  if (reading.value) return reading.value;
  if (!result || result.rows == null) return "Read a dataset first";
  if (!textColumn.value) return "Pick the column that becomes the state";
  if (!labelColumn.value) return "Pick the label column";
  if (textColumn.value === labelColumn.value) return "The state and the label are different columns";
  return splitsProblem(table, splits.value, base.value);
});

export function anyJob() {
  const result = previewed.value;
  const body = {
    kind: "data",
    source: result.source,
    splits: splitsBody(table, splits.value),
    text: textColumn.value,
    label: labelColumn.value,
    name: base.value,
  };
  const names = labelInfo.value?.names;
  if (names && Object.keys(names).length) body.value_names = names;
  if (result.config) body.config = result.config;
  if (seeded(table, splits.value)) body.seed = Number(seed.value) || 0;
  return body;
}

// ---------- the pane ----------

function Source() {
  const result = previewed.value;
  const onKeyDown = (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    event.stopPropagation(); // Enter reads the dataset, it does not download it
    check();
  };
  return html`
    <div class="opts">
      <h4>Hugging Face dataset</h4>
      <div class="any-source">
        <input class="setting-input mono" aria-label="Hugging Face dataset" placeholder="hf:ORG/NAME" spellcheck=${false}
               value=${source.value} onInput=${(event) => (source.value = event.target.value)} onKeyDown=${onKeyDown} />
        <button class="btn" disabled=${!source.value.trim() || Boolean(reading.value)} onClick=${check}>Read</button>
      </div>
      ${result?.configs?.length > 1 &&
      html`<label class="any-row"><span>Config</span>
        <select aria-label="Config" value=${config.value ?? ""} onChange=${(event) => read({ config: event.target.value }, "Reading the config's splits…")}>
          ${!config.value && html`<option value="">Pick one</option>`}
          ${result.configs.map((item) => html`<option value=${item}>${item}</option>`)}
        </select></label>`}
      ${result?.splits?.length > 0 &&
      html`<label class="any-row"><span>Preview</span>
        <select aria-label="Split to preview" value=${split.value ?? ""} onChange=${(event) => read({ config: config.value, split: event.target.value }, "Reading the split into the Hugging Face cache…")}>
          ${!split.value && html`<option value="">Pick one</option>`}
          ${result.splits.map((item) => html`<option value=${item}>${item}</option>`)}
        </select></label>`}
      ${reading.value && html`<div class="checking"><${StatusIcon} kind="spin" /> ${reading.value}</div>`}
      ${failed.value && html`<div class="notice"><${StatusIcon} kind="warn" /><span>${failed.value}</span></div>`}
    </div>
  `;
}

function Preview() {
  const result = previewed.value;
  return html`
    <div class="opts">
      <h4>${plural(result.rows, "row")} · the first ${result.sample.length}</h4>
      <div class="any-table"><table>
        <thead><tr>${result.columns.map((column) => html`<th>${column.name}<small>${column.names ? "classes" : column.kind}</small></th>`)}</tr></thead>
        <tbody>${result.sample.map((row) => html`<tr>${result.columns.map((column) => html`<td>${row[column.name]}</td>`)}</tr>`)}</tbody>
      </table></div>
    </div>
  `;
}

function Wiring() {
  const result = previewed.value;
  return html`
    <div class="opts">
      <h4>Examples</h4>
      <label class="any-row"><span>State</span>
        <select aria-label="State column" value=${textColumn.value ?? ""} onChange=${(event) => (textColumn.value = event.target.value)}>
          ${result.text_columns.map((item) => html`<option value=${item}>${item}</option>`)}
        </select><small>the text the model reads</small></label>
      <label class="any-row"><span>Label</span>
        <select aria-label="Label column" value=${labelColumn.value ?? ""} onChange=${(event) => (labelColumn.value = event.target.value)}>
          ${result.label_columns.map((item) => html`<option value=${item.name}>${item.name} (${plural(item.classes.length, "value")})</option>`)}
        </select><small>its values become the answers</small></label>
      ${result.label_columns.length === 0 && html`<p class="warn-text">No column has a few values to use as labels.</p>`}
    </div>
  `;
}

// The label column's values with their counts, and their names when the
// dataset card lists them (they go along to the dataset).
function LabelValues() {
  const info = labelInfo.value;
  if (!info) return null;
  return html`
    <div class="opts">
      <h4>Label values</h4>
      <div class="any-map">
        ${info.classes.map(
          ([value, count]) => html`<div class="any-row" key=${value}>
            <code>${value}</code><span class="num">${count}</span><span class="dim">${info.names?.[value] ?? ""}</span>
          </div>`,
        )}
      </div>
      <small class="hint">${info.names ? "Names from the dataset card. " : ""}The dataset arrives without questions: you write them afterwards, and say what each value means.</small>
    </div>
  `;
}

function Output() {
  return html`
    <div class="opts">
      <h4>Splits</h4>
      <${SplitTable} state=${table} splits=${splits.value} />
      <small class="hint">The preview reads ${split.value}; the columns apply to every split downloaded.</small>
    </div>
    <${NameAndSeed} name=${name} named=${named} suggested=${suggested.value} seed=${seed} sampled=${seeded(table, splits.value)} />
  `;
}

export function AnyDataset() {
  const result = previewed.value;
  const ready = result && result.rows != null;
  return html`
    <p>Any dataset on the Hugging Face Hub: one column becomes the state, and the label column's values are kept with each row. The questions come afterwards, in the Studio, with what each value means.</p>
    <${Source} />
    ${ready && html`<${Preview} /><${Wiring} /><${LabelValues} /><${Output} />`}
    ${ready && anyProblem.value && html`<p class="warn-text">${anyProblem.value}</p>`}
  `;
}
