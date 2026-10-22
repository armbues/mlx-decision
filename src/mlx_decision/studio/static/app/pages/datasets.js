// Datasets: the workspace's data/ folder, which fits and evaluations read. The
// sidebar lists them with Download and Add; the centre shows the one picked:
// its splits as tabs (each with its role and count), what keeps it from a fit,
// where it came from, the split's questions with their labels and its label
// values with their counts, and its examples in a paged table; with Fit on It…,
// Evaluate on It…, Run as Batch…, Open in Playground, and Show in Finder and Delete in its "…"
// menu. The inspector shows the example picked, whole, and the files.
import { computed, signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import { MenuButton, MenuItem, MenuSeparator } from "../components/menu.js";
import { closeSheet, openSheet } from "../components/sheet.js";
import { ExampleInspector, ExampleTable, filterLabel, filterValue, pickedExample, query, showDataset } from "../datasets/examples.js";
import { DatasetList, deleteData, pickedDataset, revealData, uploadData } from "../datasets/files.js";
import { openNewBatch } from "../batches/start.js";
import { openGetDataset } from "../datasets/get.js";
import { FileGroup, Labels, Values, useSummary } from "../datasets/summary.js";
import { ROLE_WORDS } from "../datasets/splits.js";
import { fileSize, plural } from "../format.js";
import { navigate } from "../router.js";
import { dataFiles, datasetOf, described, summaryKey } from "../store/data.js";
import { inspectorMode, toast } from "../store/ui.js";
import { evalOn } from "../training/evaluate.js";
import { fitOn } from "../training/fit.js";

// The split shown, per dataset (null: its default split).
const shownSplit = signal({});
const splitOf = (name) => shownSplit.value[name] ?? null;

const status = computed(() => {
  const list = dataFiles.value;
  if (!list) return null;
  const name = pickedDataset();
  const summary = name ? described.value[summaryKey(name, splitOf(name))] : null;
  if (summary?.examples != null) {
    const split = summary.kind === "folder" ? html` · ${summary.split}` : "";
    return { text: html`<b>${name}</b>${split} · ${plural(summary.examples, "example")}` };
  }
  if (list.length === 0) return { text: "No datasets yet" };
  return { text: html`<b>${plural(list.length, "dataset")}</b> in data/` };
});

const action = computed(() => ({
  label: "Download a Dataset…",
  title: "Download a Dataset… (a public dataset from the Hugging Face Hub)",
  icon: false,
  run: openGetDataset,
}));

function Empty() {
  return html`
    <div class="empty ds-empty">
      <b>No datasets yet</b>
      <span>A dataset holds examples: each a state and its label, split into the examples a fit learns from, those its report is scored on and those kept for an evaluation. Download a public dataset from the Hugging Face Hub, or add a file of your own (training data as JSON lines, or rows of columns).</span>
      <span class="empty-actions">
        <button class="btn primary" onClick=${openGetDataset}>Download a Dataset…</button>
        <button class="btn" onClick=${uploadData}>Add a Dataset…</button>
      </span>
    </div>
  `;
}

function Questions({ summary }) {
  const current = query.value;
  if (summary.questions.length === 0) return null;
  return html`
    <section class="gsec"><div class="gsec-head"><h2>Questions</h2><span class="tag">Pick a label to see its examples</span></div>
      <div class="group ds-questions">
        ${summary.questions.map(
          (question) => html`<div class="ds-question" key=${question.id}>
            <div class="ds-q-text">
              <b class="mono">${question.id}</b><span class="ds-q-type">${question.type}</span>
              ${question.instructions && html`<p>${question.instructions}</p>`}
              ${question.answered_by && html`<small>Answered by the values of <code>${question.answered_by}</code></small>`}
            </div>
            <div class="ds-q-labels">
              <${Labels} question=${question} total=${summary.examples}
                active=${current.question === question.id ? current.label : null}
                onPick=${(label) => filterLabel(question.id, label)} />
            </div>
          </div>`,
        )}
      </div>
    </section>
  `;
}

// The label columns' values in this split, each with its count and name.
function ValuesSection({ summary }) {
  const columns = Object.keys(summary.values ?? {});
  if (columns.length === 0) return null;
  const current = query.value;
  return html`
    <section class="gsec"><div class="gsec-head"><h2>Label values</h2><span class="tag">${
      summary.questions.length ? "What the answers are mapped from" : "Pick a value to see its examples"
    }</span></div>
      <div class="group ds-questions">
        ${columns.map(
          (column) => html`<div class="ds-question" key=${column}>
            <div class="ds-q-text"><b class="mono">${column}</b><span class="ds-q-type">${plural(summary.values[column].length, "value")}</span></div>
            <div class="ds-q-labels">
              <${Values} summary=${summary} column=${column} active=${current.column === column ? current.value : null}
                onPick=${(value) => filterValue(column, value)} />
            </div>
          </div>`,
        )}
      </div>
    </section>
  `;
}

const day = (iso) => (iso ? new Date(iso).toLocaleDateString("en", { year: "numeric", month: "short", day: "numeric" }) : "");

// Where the dataset came from (its dataset.json's source) and what the split
// shown holds: the source split, a sample or a part of it.
function Source({ summary }) {
  const source = summary.source ?? {};
  const meta = summary.splits.find((split) => split.name === summary.split) ?? {};
  const hub = source.source?.startsWith("hf:");
  const repo = hub ? source.source.slice(3) : source.source;
  let taken = meta.examples != null ? plural(meta.examples, "example") : "";
  if (meta.part) taken = html`${Math.round(meta.part * 100)}% of ${meta.split ? html`<code>${meta.split}</code>` : "the rows"}, drawn by label`;
  else if (meta.sample) taken = `a random sample of ${meta.sample.toLocaleString("en")} rows (seed ${meta.seed ?? 0})`;
  const role = meta.role ? ROLE_WORDS[meta.role] : null;
  return html`
    <section class="gsec ds-source"><div class="gsec-head"><h2>Where it came from</h2>${summary.created && html`<span class="tag">Downloaded ${day(summary.created)}</span>`}</div>
      <div class="group"><dl>
        ${repo &&
        html`<div class="kv"><dt>Dataset</dt><dd>
          ${source.preset && html`The <b>${source.preset}</b> preset: `}${source.source === "upload" ? "A file you added" : html`<code>${repo}</code>`}${hub ? " on the Hugging Face Hub" : ""}${source.config ? html`, config <code>${source.config}</code>` : ""}
          ${source.revision && html`<small>At commit <code>${source.revision.slice(0, 7)}</code></small>`}
        </dd></div>`}
        <div class="kv"><dt>This split</dt><dd><code>${summary.split}</code>${meta.split && meta.split !== summary.split ? html` from <code>${meta.split}</code>` : ""}${taken ? html`: ${taken}` : ""}</dd></div>
        ${role && html`<div class="kv"><dt>Role</dt><dd>${role.label}<small>${role.words}</small></dd></div>`}
      </dl></div>
    </section>
  `;
}

// The splits as tabs: each with its role and count.
function SplitTabs({ name, summary }) {
  if (summary.kind !== "folder") return null;
  return html`
    <div class="ds-tabs" role="tablist" aria-label="Splits">
      ${summary.splits.map(
        (split) => html`<button role="tab" key=${split.name} aria-selected=${split.name === summary.split ? "true" : "false"}
          data-split=${split.name} title=${split.role ? ROLE_WORDS[split.role].words : "no role"}
          onClick=${() => (shownSplit.value = { ...shownSplit.value, [name]: split.name })}>
          <code>${split.name}</code>${split.role && html`<span class="role">${ROLE_WORDS[split.role].label}</span>`}
          ${split.examples != null && html`<span class="num">${split.examples.toLocaleString("en")}</span>`}
        </button>`,
      )}
    </div>
  `;
}

// The Playground on the dataset: its questions written or edited there.
const writeQuestions = (name) => navigate("playground", `data:${name}`);

// What keeps the dataset from a fit, with what to do about it.
function NotReady({ name, summary }) {
  if (!summary.missing?.length) return null;
  const folder = summary.kind === "folder";
  return html`
    <div class="notice ds-banner" role="status">
      <span><b>Not ready for a fit:</b> ${summary.missing.join("; ")}. A fit and an evaluation need questions, and for every label value the answer it stands for.</span>
      ${folder && html`<button class="btn sm" onClick=${() => writeQuestions(name)}>${summary.questions.length ? "Edit Questions" : "Write Questions"}</button>`}
    </div>
  `;
}

// Fit on It… and Evaluate on It…: the form, or what is missing first.
function onIt(name, run, what) {
  const item = datasetOf(name);
  if (!item?.missing?.length) {
    run(name);
    return;
  }
  const asked = (item.questions ?? []).length > 0;
  openSheet({
    title: `${name} is not ready for ${what}`,
    purpose: `${item.missing.join("; ")}. ${what === "a fit" ? "A fit learns" : "An evaluation scores models"} on questions whose answers the dataset's label values give: its questions first, then for each value the answer it stands for.`,
    actions:
      item.kind === "folder"
        ? [
            {
              label: asked ? "Edit Questions" : "Write Questions",
              primary: true,
              onSelect: () => {
                closeSheet();
                writeQuestions(name);
              },
            },
          ]
        : [{ label: "OK", primary: true, onSelect: closeSheet }],
  });
}

// A question set from the split's questions and first examples, opened in the
// Playground: the base model tried on them before a fit, the fitted one after.
async function openInPlayground(name, split) {
  try {
    const query = split ? `?split=${encodeURIComponent(split)}` : "";
    const made = await api(`/data/${encodeURIComponent(name)}/set${query}`, { method: "POST", body: {} });
    navigate("playground", made.name);
    toast(`Saved as the question set ${made.name}, with ${plural(made.examples, "example")} from ${name}`);
  } catch (error) {
    toast(`Could not open ${name} in the Playground: ${error.message}`);
  }
}

function Picked({ name }) {
  const split = splitOf(name);
  const summary = useSummary(name, split);
  if (summary && query.value.name === name && query.value.split !== summary.split && !summary.error) showDataset(name, summary.split);
  if (!summary) return null;
  if (summary.error)
    return html`<div class="pane-head"><div><h1 class="mono">${name}</h1></div></div><p class="warn-text">${summary.error}</p>`;
  const folder = summary.kind === "folder";
  const asked = summary.questions.length > 0;
  return html`
    <div class="pane-head">
      <div>
        <h1 class="mono">${name}</h1>
        <p>${folder ? `${plural(summary.splits.length, "split")} · ` : ""}${plural(summary.examples, "example")}${folder ? ` in ${summary.split}` : ""} · ${plural(summary.questions.length, "question")} · ${fileSize(summary.bytes)}</p>
      </div>
      <div class="actions">
        <button class="btn primary" onClick=${() => onIt(name, fitOn, "a fit")} title=${`A new fit on ${name}, on the Training page`}>Fit on It…</button>
        <button class="btn" onClick=${() => onIt(name, evalOn, "an evaluation")} title=${`Score a model on ${name}, on the Training page`}>Evaluate on It…</button>
        ${folder
          ? html`<button class="btn" onClick=${() => writeQuestions(name)}
                title=${`The Playground on ${name}: its questions, tried on its rows, saved in its dataset.json`}>${asked ? "Edit Questions" : "Write Questions"}</button>`
          : html`<button class="btn" onClick=${() => openInPlayground(name, summary.split)} disabled=${!asked}
                title=${asked ? `A new question set with ${name}'s questions and the first ${Math.min(summary.examples, 10)} examples, to try a model on` : `${name} has no questions yet`}>Open in Playground</button>`}
        <${MenuButton} id="data-more" className="btn" ariaLabel=${`More actions for ${name}`}
          label=${html`<svg viewBox="0 0 20 20" fill="currentColor" width="15" height="15"><circle cx="5" cy="10" r="1.5" /><circle cx="10" cy="10" r="1.5" /><circle cx="15" cy="10" r="1.5" /></svg>`}
          menuStyle="top:32px; right:0; width:200px">
          <${MenuItem} onSelect=${() => openNewBatch({ dataset: name, split: summary.split })}>Run as Batch…<//>
          <${MenuSeparator} />
          <${MenuItem} onSelect=${() => revealData(name)}>Show in Finder<//>
          <${MenuSeparator} />
          <${MenuItem} onSelect=${() => deleteData(name)}>Delete…<//>
        <//>
      </div>
    </div>
    <${NotReady} name=${name} summary=${summary} />
    <${SplitTabs} name=${name} summary=${summary} />
    ${(summary.source || folder) && html`<${Source} summary=${summary} />`}
    <${Questions} summary=${summary} />
    <${ValuesSection} summary=${summary} />
    <${ExampleTable} summary=${summary} />
    <div class="ds-files"><${FileGroup} summary=${summary} /></div>
  `;
}

function Centre() {
  const list = dataFiles.value;
  const name = pickedDataset();
  if (name && query.value.name !== name) showDataset(name, splitOf(name));
  let body;
  if (!list) body = null;
  else if (list.length === 0) body = html`<${Empty} />`;
  else if (!name)
    body = html`
      <div class="pane-head"><div><h1>Datasets</h1><p>${plural(list.length, "dataset")} in the workspace's data/ folder</p></div></div>
      <div class="empty"><b>Pick a dataset</b><span>Pick one on the left to see its questions, labels and examples.</span></div>`;
  else if (!list.some((item) => item.name === name))
    body = html`<div class="empty"><b>No dataset named ${name}</b><span>It may have been deleted. Pick another on the left.</span></div>`;
  else body = html`<${Picked} key=${name} name=${name} />`;
  return html`<div class="centre-inner">${body}</div>`;
}

function Inspector() {
  const name = pickedDataset();
  const summary = name ? described.value[summaryKey(name, splitOf(name))] : null;
  return html`<${ExampleInspector} name=${name} split=${splitOf(name)} shown=${summary?.split ?? null} />`;
}

// The inspector only while an example is picked (closing it unpicks it).
inspectorMode("datasets", {
  mode: "pick",
  picked: computed(() => pickedExample.value != null && pickedExample.value.name === pickedDataset()),
  close: () => (pickedExample.value = null),
});

export default {
  id: "datasets",
  title: "Datasets",
  action,
  status,
  Centre,
  SidebarList: DatasetList,
  tabs: [{ id: "example", label: "Example", Body: Inspector }],
};
