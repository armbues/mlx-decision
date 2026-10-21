// The splits of a download: a table with each split's rows, its role (train,
// report, evaluate or left out; one split per role) or its parts (a share per
// role, drawn by label: for a dataset with one split) and its own random
// sample. Both panes of Download use it: the presets and "Another dataset…".
import { signal } from "@preact/signals";
import { html } from "htm/preact";
import { freeName, nameTaken } from "../store/data.js";

export const OUT = "out";
export const PARTS = "parts";
export const ROLES = [
  ["train", "Train", "what the fit learns from"],
  ["report", "Report", "what the fit's report is scored on"],
  ["evaluate", "Evaluate", "kept for Evaluate"],
  [OUT, "Leave out", "not downloaded"],
  [PARTS, "Cut into parts", "a share for each role, drawn by label"],
];
export const ROLE_WORDS = Object.fromEntries(ROLES.map(([id, label, words]) => [id, { label, words }]));
const PART_ROLES = ["train", "report", "evaluate"];
const DEFAULT_PARTS = { train: "80", report: "10", evaluate: "10" };

export const NAME = /^[A-Za-z0-9_-][A-Za-z0-9._-]{0,79}$/;

// A sample's default: 1,000 rows, or all of a smaller split's.
export const defaultSample = (rows) => String(rows > 0 ? Math.min(1000, rows) : 1000);

// The roles by default: one split alone is cut into parts; else the train
// split trains, validation (or dev) reports, else test; test is for Evaluate
// when another split reports; the rest are left out. ``first`` trains when no
// split is named like train.
export function defaultRoles(names, first = null) {
  if (names.length === 1) return { [names[0]]: PARTS };
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
      before[split.name] ?? { role: roles[split.name], sampling: false, sample: defaultSample(split.rows), typed: false, parts: { ...DEFAULT_PARTS } },
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
  if (fields.role && fields.role !== OUT && fields.role !== PARTS)
    for (const [other, entry] of Object.entries(next))
      if (other !== name && entry.role === fields.role) next = { ...next, [other]: { ...entry, role: OUT } };
  state.value = next;
}

// The splits downloaded, in the table's order.
export const chosen = (state, splits) => splits.filter((split) => state.value[split.name] && state.value[split.name].role !== OUT);

// A split's parts with a share above 0: [[role, percent]].
const partsOf = (entry) => PART_ROLES.map((role) => [role, Number(entry.parts?.[role])]).filter(([, share]) => share > 0);

// Whether a sample seed applies: a sample, or a split cut into parts.
export const seeded = (state, splits) =>
  chosen(state, splits).some((split) => state.value[split.name].sampling || state.value[split.name].role === PARTS);

// The first free name like ``base``.
export const freeBase = (base) => freeName(base);

// Why the table and the name cannot be downloaded, or null.
export function splitsProblem(state, splits, base) {
  const got = chosen(state, splits);
  if (got.length === 0) return "Pick a role for at least one split";
  if (!NAME.test(base)) return "Name the dataset: letters, digits, '.', '_' or '-'";
  const taken = nameTaken(base);
  if (taken) return taken;
  const roles = [];
  for (const split of got) {
    const entry = state.value[split.name];
    if (entry.role !== PARTS) {
      roles.push(entry.role);
      continue;
    }
    const parts = PART_ROLES.map((role) => [role, entry.parts?.[role] ?? ""]);
    for (const [role, text] of parts) {
      const share = Number(text);
      if (text !== "" && (!Number.isFinite(share) || share < 0 || share > 100)) return `${split.name}: ${role} is a share from 0 to 100%`;
    }
    const kept = partsOf(entry);
    if (kept.length === 0) return `${split.name}: give at least one part a share`;
    if (kept.reduce((sum, [, share]) => sum + share, 0) > 100) return `${split.name}: the parts add up to more than 100%`;
    roles.push(...kept.map(([role]) => role));
  }
  const twice = roles.find((role, index) => roles.indexOf(role) !== index);
  if (twice) return `One split per role: two are ${ROLE_WORDS[twice].label.toLowerCase()}`;
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

// The job's splits: [{split, role or parts, sample, rows}].
export function splitsBody(state, splits) {
  return chosen(state, splits).map((split) => {
    const entry = state.value[split.name];
    const body =
      entry.role === PARTS
        ? { split: split.name, parts: partsOf(entry).map(([role, share]) => ({ role, share: share / 100 })) }
        : { split: split.name, role: entry.role };
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
          ${entry.role === PARTS &&
          html`<span class="st-parts" role="group" aria-label=${`Parts of ${split.name}`}>
            ${PART_ROLES.map(
              (role) => html`<label key=${role} title=${ROLE_WORDS[role].words}>${ROLE_WORDS[role].label}
                <input class="setting-input mono" type="number" min="0" max="100" aria-label=${`${ROLE_WORDS[role].label} share of ${split.name}`}
                       value=${entry.parts?.[role] ?? ""}
                       onInput=${(event) => change(state, split.name, { parts: { ...entry.parts, [role]: event.target.value } })} /> %</label>`,
            )}
          </span>`}
        </div>`;
      })}
    </div>
  `;
}

// The seed of the samples and parts, and the dataset's name: its folder in
// data/. ``name`` holds what was typed (``named`` once typed); ``suggested``
// the name shown until then.
export function NameAndSeed({ name, named, suggested, seed, sampled }) {
  return html`
    ${sampled &&
    html`<label class="check ds-sample">Seed of the samples and parts
      <input class="setting-input mono" aria-label="Seed" type="number" min="0" value=${seed.value}
             onInput=${(event) => (seed.value = event.target.value)} /></label>`}
    <label class="name-field">
      <span>Name</span>
      <input class="sheet-input mono" aria-label="Name" spellcheck=${false} value=${named.value ? name.value : suggested}
             onInput=${(event) => {
               name.value = event.target.value;
               named.value = true;
             }} />
      <span class="suffix">/</span>
    </label>
  `;
}
