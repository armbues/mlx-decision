// Another dataset (TRS-6): any Hugging Face dataset, read as a preview, with the
// column that becomes the state, the label column, a question set's question
// the labels answer, and each class mapped to one of its options (or left
// out). It runs `data` as a job with the question written for it.
import { computed, signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import { StatusIcon } from "../components/icons.js";
import { plural } from "../format.js";
import { dataFiles } from "../store/data.js";
import { sets } from "../store/sets.js";

const NAME = /^[A-Za-z0-9_-][A-Za-z0-9._-]{0,79}$/;
const LEAVE_OUT = "";

export const source = signal("");
const previewed = signal(null); // the preview route's answer
const reading = signal(null); // what is being read, while it is
const failed = signal(null);
const config = signal(null);
const split = signal(null);
const textColumn = signal(null);
const labelColumn = signal(null);
const setName = signal(null);
const setQuestions = signal(null); // the set's questions
const questionId = signal(null);
const mapping = signal({}); // class -> option id or LEAVE_OUT
const sampling = signal(false);
const sample = signal("1000");
const seed = signal("0");
const name = signal("");
const named = signal(false);

export function resetAny() {
  previewed.value = null;
  failed.value = null;
  config.value = null;
  split.value = null;
  named.value = false;
}

async function read(fields, what) {
  reading.value = what;
  failed.value = null;
  try {
    const result = await api("/datasets/preview", { method: "POST", body: { source: source.value.trim(), ...fields } });
    previewed.value = result;
    config.value = result.config;
    split.value = result.split;
    if (result.rows != null) {
      textColumn.value = result.text;
      labelColumn.value = result.label;
      mapByName();
      if (!named.value) name.value = freeName(`${result.source.split("/").pop()}-${result.split}`);
    }
  } catch (error) {
    failed.value = error.message;
  } finally {
    reading.value = null;
  }
}

export function check() {
  resetAny();
  const value = source.value.trim();
  if (!value.startsWith("hf:")) source.value = `hf:${value}`;
  read({}, "Asking the Hub for its configs and splits…");
}

function freeName(base) {
  const taken = new Set((dataFiles.value ?? []).map((item) => item.name));
  let candidate = base;
  for (let n = 2; taken.has(candidate); n += 1) candidate = `${base}-${n}`;
  return candidate;
}

const labelInfo = computed(() => previewed.value?.label_columns?.find((column) => column.name === labelColumn.value) ?? null);
const question = computed(() => setQuestions.value?.[questionId.value] ?? null);
const options = computed(() => {
  const q = question.value;
  if (!q) return [];
  if (q.type === "noul") return [["true", "Yes (true)"], ["false", "No (false)"]];
  if (q.type === "choice") return Object.entries(q.criteria ?? {}).map(([id, text]) => [id, typeof text === "string" ? text : id]);
  return [];
});

// Each class to the option of the same name (or yes / no for a noul), else left out.
function mapByName() {
  const classes = labelInfo.value?.classes ?? [];
  const ids = options.value.map(([id]) => id);
  const yes = new Set(["true", "yes", "1", "positive"]);
  const no = new Set(["false", "no", "0", "negative"]);
  const next = {};
  for (const [label] of classes) {
    const lower = label.toLowerCase();
    let option = ids.find((id) => id.toLowerCase() === lower) ?? LEAVE_OUT;
    if (option === LEAVE_OUT && question.value?.type === "noul") option = yes.has(lower) ? "true" : no.has(lower) ? "false" : LEAVE_OUT;
    next[label] = option;
  }
  mapping.value = next;
}

async function pickSet(nameOfSet) {
  setName.value = nameOfSet;
  setQuestions.value = null;
  questionId.value = null;
  if (!nameOfSet) return;
  try {
    const { body } = await api(`/sets/${encodeURIComponent(nameOfSet)}`);
    setQuestions.value = body.questions ?? {};
    const usable = Object.entries(setQuestions.value).find(([, q]) => q.type === "choice" || q.type === "noul");
    questionId.value = usable?.[0] ?? null;
    mapByName();
  } catch (error) {
    failed.value = error.message;
  }
}

export const anyProblem = computed(() => {
  const result = previewed.value;
  if (reading.value) return reading.value;
  if (!result || result.rows == null) return "Read a dataset first";
  if (!textColumn.value) return "Pick the column that becomes the state";
  if (!labelColumn.value) return "Pick the label column";
  if (textColumn.value === labelColumn.value) return "The state and the label are different columns";
  if (!setName.value) return "Pick a question set";
  if (!question.value) return "Pick the question the labels answer";
  if (question.value.type !== "choice" && question.value.type !== "noul") return "A choice or a yes/no question";
  if (!Object.values(mapping.value).some((option) => option !== LEAVE_OUT)) return "Map at least one class to an option";
  const value = name.value.trim();
  if (!NAME.test(value)) return "Name the file: letters, digits, '.', '_' or '-'";
  if ((dataFiles.value ?? []).some((item) => item.name === value)) return `A data file named ${value} exists already`;
  if (sampling.value) {
    const count = Number(sample.value);
    if (!Number.isInteger(count) || count < 1 || count > result.rows) return `A sample is 1 to ${result.rows} rows`;
  }
  return null;
});

export function anyJob() {
  const result = previewed.value;
  const body = {
    kind: "data",
    source: result.source,
    split: result.split,
    text: textColumn.value,
    label: labelColumn.value,
    set: setName.value,
    question: questionId.value,
    label_map: Object.fromEntries(Object.entries(mapping.value).filter(([, option]) => option !== LEAVE_OUT)),
    rows: result.rows,
    name: name.value.trim(),
  };
  if (result.config) body.config = result.config;
  if (sampling.value) {
    body.sample = Number(sample.value);
    body.seed = Number(seed.value) || 0;
  }
  return body;
}

// ---------- the pane ----------

function Source() {
  const result = previewed.value;
  const onKeyDown = (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    event.stopPropagation(); // Enter reads the dataset, it does not get it
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
      html`<label class="any-row"><span>Split</span>
        <select aria-label="Split" value=${split.value ?? ""} onChange=${(event) => read({ config: config.value, split: event.target.value }, "Reading the split into the Hugging Face cache…")}>
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
  const list = sets.value ?? [];
  const questions = Object.entries(setQuestions.value ?? {});
  return html`
    <div class="opts">
      <h4>Examples</h4>
      <label class="any-row"><span>State</span>
        <select aria-label="State column" value=${textColumn.value ?? ""} onChange=${(event) => (textColumn.value = event.target.value)}>
          ${result.text_columns.map((item) => html`<option value=${item}>${item}</option>`)}
        </select><small>the text the model reads</small></label>
      <label class="any-row"><span>Label</span>
        <select aria-label="Label column" value=${labelColumn.value ?? ""} onChange=${(event) => {
          labelColumn.value = event.target.value;
          mapByName();
        }}>
          ${result.label_columns.map((item) => html`<option value=${item.name}>${item.name} (${plural(item.classes.length, "class", "classes")})</option>`)}
        </select><small>the correct answer</small></label>
      <label class="any-row"><span>Question set</span>
        <select aria-label="Question set" value=${setName.value ?? ""} onChange=${(event) => pickSet(event.target.value)}>
          <option value="">Pick one</option>
          ${list.map((item) => html`<option value=${item.name}>${item.name}</option>`)}
        </select></label>
      ${setQuestions.value &&
      html`<label class="any-row"><span>Question</span>
        <select aria-label="Question" value=${questionId.value ?? ""} onChange=${(event) => {
          questionId.value = event.target.value;
          mapByName();
        }}>
          ${questions.map(([id, q]) => html`<option value=${id} disabled=${q.type !== "choice" && q.type !== "noul"}>${id} (${q.type})</option>`)}
        </select><small>${question.value?.instructions ?? ""}</small></label>`}
      ${result.label_columns.length === 0 && html`<p class="warn-text">No column has a few classes to use as labels.</p>`}
    </div>
  `;
}

function Mapping() {
  const info = labelInfo.value;
  if (!info || !question.value) return null;
  return html`
    <div class="opts">
      <h4>Classes to options</h4>
      <div class="any-map">
        ${info.classes.map(
          ([label, count]) => html`<label class="any-row" key=${label}>
            <code>${label}</code><span class="num">${count}</span>
            <select aria-label=${`Option for ${label}`} value=${mapping.value[label] ?? LEAVE_OUT}
                    onChange=${(event) => (mapping.value = { ...mapping.value, [label]: event.target.value })}>
              <option value=${LEAVE_OUT}>leave out</option>
              ${options.value.map(([id, text]) => html`<option value=${id}>${id}${text !== id ? ` · ${text}` : ""}</option>`)}
            </select>
          </label>`,
        )}
      </div>
      <small class="hint">Rows of a class left out are not used.</small>
    </div>
  `;
}

function Output() {
  const result = previewed.value;
  return html`
    <label class="check ds-sample">
      <input type="checkbox" checked=${sampling.value} onChange=${(event) => (sampling.value = event.target.checked)} />
      A random sample of
      <input class="setting-input mono" aria-label="Rows" type="number" min="1" max=${result.rows} value=${sample.value}
             onFocus=${() => (sampling.value = true)} onInput=${(event) => (sample.value = event.target.value)} />
      rows, seed
      <input class="setting-input mono" aria-label="Seed" type="number" min="0" value=${seed.value} onInput=${(event) => (seed.value = event.target.value)} />
    </label>
    <label class="name-field">
      <span>Name</span>
      <input class="sheet-input mono" aria-label="Name" spellcheck=${false} value=${name.value}
             onInput=${(event) => {
               name.value = event.target.value;
               named.value = true;
             }} />
      <span class="suffix">.jsonl</span>
    </label>
  `;
}

export function AnyDataset() {
  const result = previewed.value;
  const ready = result && result.rows != null;
  return html`
    <p>Any dataset on the Hugging Face Hub: one column becomes the state, a question from one of your question sets is asked, and the label column's classes are its answers.</p>
    <${Source} />
    ${ready && html`<${Preview} /><${Wiring} /><${Mapping} /><${Output} />`}
    ${ready && anyProblem.value && html`<p class="warn-text">${anyProblem.value}</p>`}
  `;
}
