// The splits of a Get: a table with each split's rows, its role (train, report,
// evaluate or left out; one split per role) and its own random sample. Both
// panes of Get a Dataset use it: the presets and "Another dataset…".
import { signal } from "@preact/signals";
import { html } from "htm/preact";
import { freeName, nameTaken } from "../store/data.js";

export const OUT = "out";
export const ROLES = [
  ["train", "Train", "what the fit learns from"],
  ["report", "Report", "what the fit's report is scored on"],
  ["evaluate", "Evaluate", "kept for Evaluate"],
  [OUT, "Leave out", "not got"],
];
export const ROLE_WORDS = Object.fromEntries(ROLES.map(([id, label, words]) => [id, { label, words }]));

const NAME = /^[A-Za-z0-9_-][A-Za-z0-9._-]{0,79}$/;

// A sample's default: 1,000 rows, or all of a smaller split's.
export const defaultSample = (rows) => String(rows > 0 ? Math.min(1000, rows) : 1000);

// The roles by default: the train split trains, validation (or dev) reports,
// else test; test is for Evaluate when another split reports; the rest are
// left out. ``first`` trains when no split is named like train.
export function defaultRoles(names, first = null) {
  const find = (pattern, taken) => names.find((name) => pattern.test(name) && !taken.includes(name));
  const train = find(/train/i, []) ?? first ?? names[0];
  const report = find(/valid|dev/i, [train]) ?? find(/test/i, [train]);
  const evaluate = find(/test/i, [train, report]);
  return Object.fromEntries(
    names.map((name) => [name, name === train ? "train" : name === report ? "report" : name === evaluate ? "evaluate" : OUT]),
  );
}

// A table's state: per split {role, sampling, sample, typed}.
export function splitState() {
  return signal({});
}

// The table for ``splits`` ([{name, rows, note}]), keeping what is there for a
// split already listed unless ``fresh``.
export function setSplits(state, splits, first = null, fresh = false) {
  const roles = defaultRoles(splits.map((split) => split.name), first);
  const before = fresh ? {} : state.value;
  state.value = Object.fromEntries(
    splits.map((split) => [
      split.name,
      before[split.name] ?? { role: roles[split.name], sampling: false, sample: defaultSample(split.rows), typed: false },
    ]),
  );
}

// A split's rows came to be known (the preview read it): its sample's default follows.
export function knowRows(state, name, rows) {
  const entry = state.value[name];
  if (entry && !entry.typed) state.value = { ...state.value, [name]: { ...entry, sample: defaultSample(rows) } };
}

function change(state, name, fields) {
  let next = { ...state.value, [name]: { ...state.value[name], ...fields } };
  // One split per role: the split that had it is left out.
  if (fields.role && fields.role !== OUT)
    for (const [other, entry] of Object.entries(next))
      if (other !== name && entry.role === fields.role) next = { ...next, [other]: { ...entry, role: OUT } };
  state.value = next;
}

// The splits got, in the table's order.
export const chosen = (state, splits) => splits.filter((split) => state.value[split.name] && state.value[split.name].role !== OUT);

// The files a Get named ``base`` writes: BASE-SPLIT.
export const filesOf = (base, names) => names.map((name) => `${base}-${name}`);

// The first name like ``base`` whose files are all free.
export function freeBase(base, names) {
  if (names.length === 0) return freeName(base);
  let candidate = base;
  for (let n = 2; filesOf(candidate, names).some((file) => nameTaken(file)); n += 1) candidate = `${base}-${n}`;
  return candidate;
}

// Why the table and the name cannot be got, or null.
export function splitsProblem(state, splits, base) {
  const got = chosen(state, splits);
  if (got.length === 0) return "Pick a role for at least one split";
  if (!NAME.test(base)) return "Name the files: letters, digits, '.', '_' or '-'";
  for (const file of filesOf(base, got.map((split) => split.name))) {
    if (!NAME.test(file)) return `${file} is too long a name`;
    const taken = nameTaken(file);
    if (taken) return taken;
  }
  for (const split of got) {
    const entry = state.value[split.name];
    if (!entry.sampling) continue;
    const count = Number(entry.sample);
    const most = split.rows || Infinity;
    if (!Number.isInteger(count) || count < 1 || count > most)
      return split.rows ? `${split.name}: a sample is 1 to ${split.rows} rows` : `${split.name}: a sample is 1 row or more`;
  }
  return null;
}

// The job's splits: [{split, role, sample, rows}].
export function splitsBody(state, splits) {
  return chosen(state, splits).map((split) => {
    const entry = state.value[split.name];
    const body = { split: split.name, role: entry.role };
    if (entry.sampling) body.sample = Number(entry.sample);
    if (split.rows) body.rows = split.rows;
    return body;
  });
}

export function SplitTable({ state, splits }) {
  return html`
    <div class="split-table" role="table" aria-label="Splits">
      <div class="st-head" role="row"><span>Split</span><span class="r">Rows</span><span>Role</span><span>A random sample</span></div>
      ${splits.map((split) => {
        const entry = state.value[split.name];
        if (!entry) return null;
        const out = entry.role === OUT;
        return html`<div class=${`st-row${out ? " out" : ""}`} role="row" data-split=${split.name} key=${split.name}>
          <span class="st-name"><code>${split.name}</code>${split.note && html`<small>${split.note}</small>`}</span>
          <span class="r num">${split.rows ? split.rows.toLocaleString("en") : "–"}</span>
          <select aria-label=${`Role of ${split.name}`} value=${entry.role} title=${ROLE_WORDS[entry.role].words}
                  onChange=${(event) => change(state, split.name, { role: event.target.value })}>
            ${ROLES.map(([id, label]) => html`<option value=${id}>${label}</option>`)}
          </select>
          <span class="st-sample">
            <input type="checkbox" aria-label=${`Sample ${split.name}`} checked=${entry.sampling} disabled=${out}
                   onChange=${(event) => change(state, split.name, { sampling: event.target.checked })} />
            <input class="setting-input mono" type="number" min="1" max=${split.rows || undefined} aria-label=${`Rows of ${split.name}`}
                   value=${entry.sample} disabled=${out}
                   onFocus=${() => change(state, split.name, { sampling: true })}
                   onInput=${(event) => change(state, split.name, { sample: event.target.value, typed: true })} />
            <span>rows</span>
          </span>
        </div>`;
      })}
    </div>
  `;
}

// The seed of the samples and the Get's name: its files are NAME-SPLIT.jsonl.
// ``name`` holds what was typed (``named`` once typed); ``suggested`` the name
// shown until then.
export function NameAndSeed({ name, named, suggested, seed, sampled }) {
  return html`
    ${sampled &&
    html`<label class="check ds-sample">Seed of the samples
      <input class="setting-input mono" aria-label="Seed" type="number" min="0" value=${seed.value}
             onInput=${(event) => (seed.value = event.target.value)} /></label>`}
    <label class="name-field">
      <span>Name</span>
      <input class="sheet-input mono" aria-label="Name" spellcheck=${false} value=${named.value ? name.value : suggested}
             onInput=${(event) => {
               name.value = event.target.value;
               named.value = true;
             }} />
      <span class="suffix">-SPLIT.jsonl</span>
    </label>
  `;
}
