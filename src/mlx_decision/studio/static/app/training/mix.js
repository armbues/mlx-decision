// Mix Datasets…: a sheet that picks several of the workspace's datasets into a
// mix, each with an optional cap on its train rows and a Held out mark, with
// a cap for the others, a seed and a name, and saves it in data/ (a folder
// whose dataset.json lists them). The fit form then has the mix picked.
import { signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import { closeSheet, openSheet } from "../components/sheet.js";
import { plural } from "../format.js";
import { dataChanged, dataFiles, splitFor } from "../store/data.js";
import { toast } from "../store/ui.js";

const NAME = /^[A-Za-z0-9_-][A-Za-z0-9._-]{0,79}$/;
// Equal caps of 2,000 train rows kept every dataset of a mix about as good as
// fitted alone, in the measurement the training guide shows.
export const DEFAULT_CAP = 2000;

// {name, cap, seed, picked: {dataset: {cap, held_out}}}: the datasets picked, in the list's order.
const form = signal(null);
const busy = signal(false);
const error = signal(null);

// The datasets a mix can hold: those ready for a fit, mixes left out.
const candidates = () => (dataFiles.value ?? []).filter((item) => !item.error && !item.missing?.length && !item.mix);

const taken = (name) => (dataFiles.value ?? []).some((item) => item.name.toLowerCase() === name.toLowerCase());

function freeName() {
  for (let n = 1; ; n++) {
    const name = n === 1 ? "mix" : `mix-${n}`;
    if (!taken(name)) return name;
  }
}

// The train rows a dataset has, when its list entry knows them.
function trainRows(item) {
  const split = splitFor(item, "train") ?? item.splits?.[0]?.name;
  return item.splits?.find((entry) => entry.name === split)?.examples ?? null;
}

const whole = (value) => value === "" || (Number.isInteger(Number(value)) && Number(value) >= 1);

// Why the mix cannot be saved, or null.
function whyNot() {
  const current = form.value;
  if (!current) return "";
  const picked = Object.entries(current.picked);
  if (picked.length < 2) return "Pick two datasets or more";
  if (picked.every(([, entry]) => entry.held_out)) return "Train on one of them at least: not every dataset can be held out";
  if (!whole(current.cap) || picked.some(([, entry]) => !whole(entry.cap))) return "A cap is a whole number of rows, 1 or more, or empty for every row";
  const name = current.name.trim();
  if (!NAME.test(name)) return "Give the mix a name: letters, digits, '.', '_' or '-'";
  if (taken(name)) return `A dataset named ${name} exists already`;
  return null;
}

// Mix Datasets…; ``first`` is picked to start with, ``onSaved(name)`` runs
// once the mix is saved.
export function openMix({ first = null, onSaved = null } = {}) {
  const ready = candidates().some((item) => item.name === first);
  form.value = { name: freeName(), cap: String(DEFAULT_CAP), seed: 0, picked: ready ? { [first]: { cap: "", held_out: false } } : {} };
  error.value = null;
  openSheet({
    title: "Mix Datasets",
    purpose: "One fit on several datasets learns all their questions at once, and its report shows each dataset.",
    Body: MixForm,
    wide: true,
    actions: () => [{ label: busy.value ? "Saving…" : "Save Mix", primary: true, disabled: Boolean(whyNot()) || busy.value, onSelect: () => save(onSaved) }],
  });
}

async function save(onSaved) {
  const current = form.value;
  if (whyNot() || busy.value) return;
  busy.value = true;
  error.value = null;
  const order = candidates().map((item) => item.name);
  const sources = order
    .filter((name) => current.picked[name])
    .map((name) => {
      const entry = current.picked[name];
      return { name, held_out: entry.held_out, ...(entry.cap !== "" ? { cap: Number(entry.cap) } : {}) };
    });
  const body = { name: current.name.trim(), sources, cap: current.cap === "" ? null : Number(current.cap), seed: Number(current.seed) || 0 };
  try {
    const made = await api("/data/mix", { method: "POST", body });
    dataFiles.value = [...(dataFiles.value ?? []).filter((item) => item.name !== made.name), made].sort((a, b) => a.name.localeCompare(b.name));
    dataChanged();
    closeSheet();
    onSaved?.(made.name);
    toast(`Saved the mix ${made.name} of ${plural(sources.length, "dataset")}`);
  } catch (failure) {
    error.value = failure.message;
  } finally {
    busy.value = false;
  }
}

function MixForm() {
  const current = form.value;
  if (!current) return null;
  const set = (fields) => (form.value = { ...form.value, ...fields });
  const change = (name, fields) => {
    const picked = { ...form.value.picked };
    if (fields === null) delete picked[name];
    else picked[name] = { ...(picked[name] ?? { cap: "", held_out: false }), ...fields };
    set({ picked });
  };
  const list = candidates();
  const why = whyNot();
  const left = (dataFiles.value ?? []).length - list.length;
  return html`
    <div class="split-table mix-table" role="table" aria-label="Datasets">
      <div class="st-head" role="row"><span>Dataset</span><span class="r">Train rows</span><span>Cap</span><span>Held out</span></div>
      ${list.map((item) => {
        const entry = current.picked[item.name];
        const rows = trainRows(item);
        return html`<div class=${`st-row${entry ? "" : " out"}`} role="row" data-dataset=${item.name} key=${item.name}>
          <label class="st-name check"><input type="checkbox" aria-label=${`Mix ${item.name}`} checked=${Boolean(entry)}
            onChange=${(event) => change(item.name, event.target.checked ? {} : null)} /><code>${item.name}</code></label>
          <span class="r num">${rows != null ? rows.toLocaleString("en") : "–"}</span>
          <input class="setting-input mono" type="number" min="1" aria-label=${`Cap of ${item.name}`} disabled=${!entry}
                 placeholder=${current.cap === "" ? "every row" : Number(current.cap).toLocaleString("en")} value=${entry?.cap ?? ""}
                 onInput=${(event) => change(item.name, { cap: event.target.value })} />
          <input type="checkbox" aria-label=${`Hold out ${item.name}`} disabled=${!entry} checked=${Boolean(entry?.held_out)}
                 title="Scored in the fit's report, not trained on" onChange=${(event) => change(item.name, { held_out: event.target.checked })} />
        </div>`;
      })}
    </div>
    <small class="hint role-words">A held-out dataset is scored in the fit's report but not trained on: it shows what the fit does to a question outside its data.${
      left > 0 ? ` Mixes and datasets not ready for a fit are not listed (${left}).` : ""
    }</small>
    <div class="form-grid batch-form mix-form">
      <label for="mix-cap">Cap</label>
      <span class="line"><input id="mix-cap" class="setting-input mono" type="number" min="1" value=${current.cap}
        onInput=${(event) => set({ cap: event.target.value })} /> <small class="hint">train rows from each dataset without a cap of its own, drawn at random; empty for every row. Equal caps keep each one about as good as fitted alone.</small></span>
      <label for="mix-seed">Seed</label>
      <span class="line"><input id="mix-seed" class="setting-input mono" type="number" min="0" value=${current.seed}
        onInput=${(event) => set({ seed: event.target.value })} /> <small class="hint">which rows the caps draw</small></span>
      <label for="mix-name">Name</label>
      <span class="line"><input id="mix-name" class="setting-input mono mix-name" spellcheck=${false} value=${current.name}
        onInput=${(event) => set({ name: event.target.value })} /> <small class="hint">in the workspace's data/</small></span>
    </div>
    ${(error.value || (why && Object.keys(current.picked).length > 0)) && html`<p class="warn-text">${error.value ?? why}</p>`}
  `;
}
