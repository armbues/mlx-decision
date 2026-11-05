// Another dataset: any Hugging Face dataset, one split read as a preview, read
// in one of three shapes (guessed from the columns as the data command guesses
// them): one column as the state, several columns as the state (an object of
// them), each with a label column whose values the rows keep (named from the
// dataset card when it lists them) and arriving without questions, which are
// written afterwards; multiple choice, each row asking a question written
// here with its own choices; or a picture and its label (an image column,
// shown as thumbnails, written into the dataset folder's images/), with a text
// column as the state if one is picked. The columns apply to every split in
// the table. It runs `data` as a job.
import { computed, signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import { StatusIcon } from "../components/icons.js";
import { plural, size as bytesText } from "../format.js";
import { NameAndSeed, SplitTable, freeBase, knowRows, seeded, setSplits, splitState, splitsBody, splitsProblem } from "./splits.js";

export const source = signal("");
const previewed = signal(null); // the preview route's answer
const reading = signal(null); // what is being read, while it is
const failed = signal(null);
const config = signal(null);
const split = signal(null);
const shape = signal("text"); // "text", "object", "choice" or "image"
const textColumn = signal(null);
const imageColumn = signal(null); // the image shape's picture
const captionColumn = signal(null); // the image shape's optional state
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
  textColumn.value = labelColumn.value = imageColumn.value = captionColumn.value = null;
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
      // Another config: its own splits, from their defaults; their rows as
      // the Hub's metadata gives them, before anything is read.
      const fresh = before?.config !== result.config;
      const sizes = result.sizes ?? {};
      setSplits(table, result.splits.map((name) => ({ name, rows: sizes[name]?.rows ?? null })), result.split, fresh);
      const known = Object.fromEntries(Object.entries(sizes).filter(([, item]) => item.rows != null).map(([name, item]) => [name, item.rows]));
      rowsOf.value = { ...rowsOf.value, ...known };
    }
    if (result.sample) {
      applyGuess(result);
      if (result.rows != null) {
        rowsOf.value = { ...rowsOf.value, [result.split]: result.rows };
        knowRows(table, result.split, result.rows);
      }
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
  imageColumn.value = guess?.shape === "image" ? guess.image : (result.image_columns?.[0] ?? null);
  captionColumn.value = guess?.shape === "image" ? guess.text : null;
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
  read({}, "Asking the Hub for its subsets and splits…");
}

const labelInfo = computed(() => previewed.value?.label_columns?.find((column) => column.name === labelColumn.value) ?? null);
export const anyProblem = computed(() => {
  const result = previewed.value;
  if (reading.value) return reading.value;
  if (!result || !result.sample) return "Read a dataset first";
  if (shape.value === "choice") {
    const picked = roles.value;
    for (const role of ["question", "choices", "answer"]) if (!picked[role]) return `Pick the ${role} column`;
    const named = Object.values(picked).filter(Boolean);
    if (new Set(named).size < named.length) return "Each role takes a column of its own";
    if (!instructions.value.trim()) return "Write the question's instructions";
  } else if (shape.value === "image") {
    if (!imageColumn.value) return "Pick the image column";
    if (!labelColumn.value) return "Pick the label column";
    if ([imageColumn.value, captionColumn.value].includes(labelColumn.value)) return "The image, text and label are different columns";
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
    else if (shape.value === "image") {
      body.image = imageColumn.value;
      if (captionColumn.value) body.text = captionColumn.value;
    } else body.text = textColumn.value;
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
      html`<label class="any-row"><span>Subset</span>
        <select aria-label="Subset" value=${config.value ?? ""} onChange=${(event) => read({ config: event.target.value }, "Reading the subset's splits…")}>
          ${!config.value && html`<option value="">Pick one</option>`}
          ${result.configs.map((item) => html`<option value=${item}>${item}</option>`)}
        </select></label>`}
      ${result?.splits?.length > 0 &&
      html`<label class="any-row"><span>Preview</span>
        <select aria-label="Split to preview" value=${split.value ?? ""} onChange=${(event) => read({ config: config.value, split: event.target.value }, readingText(event.target.value))}>
          ${!split.value && html`<option value="">Pick one</option>`}
          ${result.splits.map((item) => html`<option value=${item}>${item}${sizeText(result.sizes?.[item])}</option>`)}
        </select></label>`}
      ${reading.value && html`<div class="checking"><${StatusIcon} kind="spin" /> ${reading.value}</div>`}
      ${failed.value && html`<div class="notice"><${StatusIcon} kind="warn" /><span>${failed.value}</span></div>`}
    </div>
  `;
}

const rowsText = (rows) => `${rows.toLocaleString("en")} ${rows === 1 ? "row" : "rows"}`;

// A split's size as the Hub's metadata gives it: " · 3,600,000 rows · 1.0 GB".
function sizeText(size) {
  if (!size) return "";
  const rows = size.rows != null ? ` · ${rowsText(size.rows)}` : "";
  const bytes = size.bytes != null ? ` · ${size.partial ? "over " : ""}${bytesText(size.bytes)}` : "";
  return rows + bytes;
}

// What reading a split for the preview does: a small one is read whole, a
// large one (or one the Hub gives no size for) only by its first rows.
function readingText(name) {
  return previewed.value?.sizes?.[name]?.whole ? "Reading the split into the Hugging Face cache…" : "Reading its first rows…";
}

function Preview() {
  const result = previewed.value;
  const rows = result.rows != null ? rowsText(result.rows) : "Rows not known";
  return html`
    <div class="opts">
      <h4>${rows} · the first ${result.sample.length}</h4>
      <div class="any-table"><table>
        <thead><tr>${result.columns.map((column) => html`<th>${column.name}<small>${column.names ? "classes" : column.kind}</small></th>`)}</tr></thead>
        <tbody>${result.sample.map((row) => html`<tr>${result.columns.map((column) => html`<td>${Cell(column, row[column.name])}</td>`)}</tr>`)}</tbody>
      </table></div>
    </div>
  `;
}

// A preview cell: an image column's thumbnail, else its text.
function Cell(column, value) {
  if (column.kind !== "image") return value;
  return value ? html`<img class="any-thumb" src=${value} alt="" />` : html`<span class="dim">no image</span>`;
}

const SHAPES = [
  ["text", "One column", "a text the model reads"],
  ["object", "Several columns", "an object of the columns you tick"],
  ["choice", "Multiple choice", "a question with its own choices in each row"],
  ["image", "Image", "a picture and its label"],
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
      <h4>Shape</h4>
      <div class=${`choices any-shapes${result.image_columns?.length ? " four" : ""}`} role="radiogroup" aria-label="Shape">
        ${SHAPES.filter(([value]) => value !== "image" || result.image_columns?.length).map(
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
      ${shape.value === "image" &&
      html`<label class="any-row"><span>Image</span>
          <select aria-label="Image column" value=${imageColumn.value ?? ""} onChange=${(event) => (imageColumn.value = event.target.value)}>
            ${result.image_columns.map((item) => html`<option value=${item}>${item}</option>`)}
          </select><small>kept in the folder's images/</small></label>
        <label class="any-row"><span>Text</span>
          <select aria-label="Text column" value=${captionColumn.value ?? ""} onChange=${(event) => (captionColumn.value = event.target.value || null)}>
            <option value="">none</option>
            ${result.text_columns.map((item) => html`<option value=${item}>${item}</option>`)}
          </select><small>the state, if any</small></label>
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
      ${previewed.value.sampled &&
      html`<small class="hint">Counted in the first ${previewed.value.sampled.toLocaleString("en")} rows; the download reads every row.</small>`}
      ${info.names && html`<small class="hint">Names from the dataset card.</small>`}
    </div>
  `;
}

function Output() {
  return html`
    <div class="opts">
      <h4>Splits</h4>
      <${SplitTable} state=${table} splits=${splits.value} />
      <small class="hint">The columns apply to every split.</small>
    </div>
    <${NameAndSeed} name=${name} named=${named} suggested=${suggested.value} seed=${seed} sampled=${seeded(table, splits.value)} />
  `;
}

export function AnyDataset() {
  const result = previewed.value;
  const ready = result && Boolean(result.sample);
  return html`
    <p>Any dataset on the Hugging Face Hub. You write its questions afterwards; multiple choice asks its question here.</p>
    <${Source} />
    ${ready && html`<${Preview} /><${Wiring} />${shape.value !== "choice" && html`<${LabelValues} />`}<${Output} />`}
    ${ready && anyProblem.value && html`<p class="warn-text">${anyProblem.value}</p>`}
  `;
}
