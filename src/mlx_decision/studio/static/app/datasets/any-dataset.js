// Another dataset: any Hugging Face dataset, one split read as a preview, read
// in one of three shapes (guessed from the columns as the data command guesses
// them): one column as the state, several columns as the state (an object of
// them), each with a label column whose values the rows keep (named from the
// dataset card when it lists them) and arriving without questions, which are
// written afterwards; or multiple choice, each row asking a question written
// here with its own choices. The columns apply to every split in the table.
// It runs `data` as a job.
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
const shape = signal("text"); // "text", "object" or "choice"
const textColumn = signal(null);
const labelColumn = signal(null);
const stateColumns = signal([]); // the object shape's, in the preview's order
const roles = signal({}); // multiple choice: {question, choices, answer, context}
const instructions = signal("");
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
  shape.value = "text";
  stateColumns.value = [];
  roles.value = {};
  instructions.value = "";
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
      applyGuess(result);
      rowsOf.value = { ...rowsOf.value, [result.split]: result.rows };
      knowRows(table, result.split, result.rows);
    }
  } catch (error) {
    failed.value = error.message;
  } finally {
    reading.value = null;
  }
}

// The shape and columns as the data command guesses them, else one text
// column and the likeliest label.
function applyGuess(result) {
  const guess = result.guess;
  shape.value = guess?.shape ?? "text";
  textColumn.value = guess?.shape === "text" ? guess.text : result.text;
  labelColumn.value = (guess?.shape !== "choice" && guess?.label) || result.label;
  stateColumns.value = guess?.shape === "object" ? guess.columns : [];
  roles.value =
    guess?.shape === "choice"
      ? { question: guess.question, choices: guess.choices, answer: guess.answer, context: guess.context ?? null }
      : {};
}

// The columns that can take each role of multiple choice, by kind.
const KINDS = {
  question: ["text"],
  choices: ["list", "object"],
  answer: ["text", "int", "label", "other"],
  context: ["text"],
};
const fitting = (role) => (previewed.value?.columns ?? []).filter((column) => KINDS[role].includes(column.kind)).map((column) => column.name);

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
  if (shape.value === "choice") {
    const picked = roles.value;
    for (const role of ["question", "choices", "answer"]) if (!picked[role]) return `Pick the ${role} column`;
    const named = Object.values(picked).filter(Boolean);
    if (new Set(named).size < named.length) return "Each role takes a column of its own";
    if (!instructions.value.trim()) return "Write the question's instructions";
  } else {
    if (shape.value === "object" && !stateColumns.value.length) return "Tick the columns that become the state";
    if (shape.value === "text" && !textColumn.value) return "Pick the column that becomes the state";
    if (!labelColumn.value) return "Pick the label column";
    const state = shape.value === "object" ? stateColumns.value : [textColumn.value];
    if (state.includes(labelColumn.value)) return "The state and the label are different columns";
  }
  return splitsProblem(table, splits.value, base.value);
});

export function anyJob() {
  const result = previewed.value;
  const body = { kind: "data", source: result.source, splits: splitsBody(table, splits.value), shape: shape.value, name: base.value };
  if (shape.value === "choice") {
    for (const [role, column] of Object.entries(roles.value)) if (column) body[role] = column;
    body.instructions = instructions.value.trim();
  } else {
    if (shape.value === "object") body.columns = stateColumns.value;
    else body.text = textColumn.value;
    body.label = labelColumn.value;
    const names = labelInfo.value?.names;
    if (names && Object.keys(names).length) body.value_names = names;
  }
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

const SHAPES = [
  ["text", "One column", "a text the model reads"],
  ["object", "Several columns", "an object of the columns you tick"],
  ["choice", "Multiple choice", "a question with its own choices in each row"],
];

function LabelSelect() {
  const result = previewed.value;
  return html`
    <label class="any-row"><span>Label</span>
      <select aria-label="Label column" value=${labelColumn.value ?? ""} onChange=${(event) => (labelColumn.value = event.target.value)}>
        ${result.label_columns.map((item) => html`<option value=${item.name}>${item.name} (${plural(item.classes.length, "value")})</option>`)}
      </select><small>its values become the answers</small></label>
    ${result.label_columns.length === 0 && html`<p class="warn-text">No column has a few values to use as labels.</p>`}
  `;
}

function toggle(name, on) {
  const order = previewed.value.columns.map((column) => column.name);
  const next = on ? [...stateColumns.value, name] : stateColumns.value.filter((item) => item !== name);
  stateColumns.value = order.filter((item) => next.includes(item));
}

function RoleSelect({ role, label, optional = false, hint }) {
  const options = fitting(role);
  const value = roles.value[role] ?? "";
  return html`
    <label class="any-row"><span>${label}</span>
      <select aria-label=${`${label} column`} value=${value} onChange=${(event) => (roles.value = { ...roles.value, [role]: event.target.value || null })}>
        ${(optional || !value) && html`<option value="">${optional ? "none" : "Pick one"}</option>`}
        ${options.map((item) => html`<option value=${item}>${item}</option>`)}
      </select><small>${hint}</small></label>
  `;
}

function Wiring() {
  const result = previewed.value;
  return html`
    <div class="opts">
      <h4>Examples</h4>
      <div class="choices any-shapes" role="radiogroup" aria-label="Shape">
        ${SHAPES.map(
          ([value, title, text]) => html`<label class="choice" key=${value}>
            <input type="radio" name="any-shape" value=${value} checked=${shape.value === value} onChange=${() => (shape.value = value)} />
            <span><b>${title}</b><small>${text}</small></span>
          </label>`,
        )}
      </div>
      ${shape.value === "text" &&
      html`<label class="any-row"><span>State</span>
          <select aria-label="State column" value=${textColumn.value ?? ""} onChange=${(event) => (textColumn.value = event.target.value)}>
            ${result.text_columns.map((item) => html`<option value=${item}>${item}</option>`)}
          </select><small>the text the model reads</small></label>
        <${LabelSelect} />`}
      ${shape.value === "object" &&
      html`<div class="any-row cols"><span>State</span>
          <div class="any-cols" role="group" aria-label="State columns">
            ${result.columns
              .filter((column) => column.name !== labelColumn.value)
              .map(
                (column) => html`<label key=${column.name}>
                  <input type="checkbox" checked=${stateColumns.value.includes(column.name)} onChange=${(event) => toggle(column.name, event.target.checked)} />
                  <code>${column.name}</code></label>`,
              )}
          </div></div>
        <${LabelSelect} />`}
      ${shape.value === "choice" &&
      html`<${RoleSelect} role="question" label="Question" hint="the state the model reads" />
        <${RoleSelect} role="choices" label="Choices" hint="each row's options" />
        <${RoleSelect} role="answer" label="Answer" hint="the right option" />
        <${RoleSelect} role="context" label="Context" optional=${true} hint="read with the question" />
        <label class="any-row wide"><span>Instructions</span>
          <input class="setting-input" aria-label="Instructions" placeholder="Which option answers the question?" value=${instructions.value}
                 onInput=${(event) => (instructions.value = event.target.value)} /></label>`}
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
    <p>Any dataset on the Hugging Face Hub. Its rows keep their label's values and get their questions afterwards, in the Studio; multiple choice asks its question here.</p>
    <${Source} />
    ${ready && html`<${Preview} /><${Wiring} />${shape.value !== "choice" && html`<${LabelValues} />`}<${Output} />`}
    ${ready && anyProblem.value && html`<p class="warn-text">${anyProblem.value}</p>`}
  `;
}
