// Datasets: the files of labelled examples in the workspace's data/ folder,
// which fits and evaluations read. The sidebar lists them with Get a Dataset…
// and Add…; the centre shows the one picked (its questions with their labels
// and counts, its examples in a paged table) with Fit on It…, Show in Finder
// and Delete; the inspector the example picked, whole, and the file.
import { computed } from "@preact/signals";
import { html } from "htm/preact";
import { ExampleInspector, ExampleTable, filterLabel, query, showDataset } from "../datasets/examples.js";
import { DatasetList, deleteData, pickedDataset, revealData, uploadData } from "../datasets/files.js";
import { openGetDataset } from "../datasets/get.js";
import { Labels, useSummary } from "../datasets/summary.js";
import { ROLE_WORDS } from "../datasets/splits.js";
import { fileSize, plural } from "../format.js";
import { href } from "../router.js";
import { dataFiles, described } from "../store/data.js";
import { fitOn } from "../training/fit.js";

const status = computed(() => {
  const list = dataFiles.value;
  if (!list) return null;
  const name = pickedDataset();
  const summary = name ? described.value[name] : null;
  if (summary?.examples != null) return { text: html`<b>${name}</b> · ${plural(summary.examples, "example")}` };
  if (list.length === 0) return { text: "No datasets yet · Get a Dataset… or Add…" };
  return { text: html`<b>${plural(list.length, "dataset")}</b> in data/` };
});

const action = computed(() => ({
  label: "Get a Dataset…",
  title: "Get a Dataset… (a public dataset from the Hugging Face Hub)",
  icon: false,
  run: openGetDataset,
}));

function Empty() {
  return html`
    <div class="empty ds-empty">
      <b>No datasets yet</b>
      <span>A dataset is a file of labelled examples: each a state, the questions asked about it and the correct answers. Fits learn from one and evaluations score models on one. Get a public dataset from the Hugging Face Hub, or add a JSON-lines file of your own.</span>
      <span class="empty-actions">
        <button class="btn primary" onClick=${openGetDataset}>Get a Dataset…</button>
        <button class="btn" onClick=${uploadData}>Add…</button>
      </span>
    </div>
  `;
}

function Questions({ summary }) {
  const current = query.value;
  return html`
    <section class="gsec"><div class="gsec-head"><h2>Questions</h2><span class="tag">Pick a label to see its examples</span></div>
      <div class="group ds-questions">
        ${summary.questions.map(
          (question) => html`<div class="ds-question" key=${question.id}>
            <div class="ds-q-text">
              <b class="mono">${question.id}</b><span class="ds-q-type">${question.type}</span>
              ${question.instructions && html`<p>${question.instructions}</p>`}
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

const day = (iso) => (iso ? new Date(iso).toLocaleDateString("en", { year: "numeric", month: "short", day: "numeric" }) : "");

// Where the dataset came from: its record, written by the Get (data --record).
function Source({ name, record }) {
  const hub = record.source?.startsWith("hf:");
  const repo = hub ? record.source.slice(3) : record.source;
  const rows = record.rows?.toLocaleString("en");
  let taken = `all ${rows} rows`;
  if (record.sample) taken = `a random sample of ${record.sample.toLocaleString("en")} of ${rows} rows (seed ${record.seed ?? 0})`;
  else if (record.limit) taken = `the first ${record.limit.toLocaleString("en")} of ${rows} rows`;
  const others = (record.get?.files ?? []).filter((file) => file.name !== name);
  const listed = new Set((dataFiles.value ?? []).map((item) => item.name));
  const role = record.role ? ROLE_WORDS[record.role] : null;
  return html`
    <section class="gsec ds-source"><div class="gsec-head"><h2>Where it came from</h2>${record.created && html`<span class="tag">Got ${day(record.created)}</span>`}</div>
      <div class="group"><dl>
        <div class="kv"><dt>Dataset</dt><dd>
          ${record.preset && html`The <b>${record.preset}</b> preset: `}<code>${repo}</code>${hub ? " on the Hugging Face Hub" : ""}${record.config ? html`, config <code>${record.config}</code>` : ""}
          ${record.revision && html`<small>At commit <code>${record.revision.slice(0, 7)}</code></small>`}
        </dd></div>
        <div class="kv"><dt>Split</dt><dd><code>${record.split ?? "–"}</code>: ${taken}</dd></div>
        ${role && html`<div class="kv"><dt>Role</dt><dd>${role.label}<small>${role.words}</small></dd></div>`}
        ${others.length > 0 &&
        html`<div class="kv"><dt>Got with it</dt><dd class="ds-siblings">
          ${others.map(
            (file) => html`<span>${listed.has(file.name) ? html`<a href=${href("datasets", file.name)}>${file.name}</a>` : html`<span class="gone" title="Deleted">${file.name}</span>`}${
              file.role ? html` <small>${ROLE_WORDS[file.role]?.label.toLowerCase() ?? file.role}</small>` : ""
            }</span>`,
          )}
        </dd></div>`}
        ${record.label_map &&
        html`<div class="kv"><dt>Classes</dt><dd>${Object.entries(record.label_map).map(([raw, option]) => html`<span class="ds-map"><code>${raw}</code> → <code>${option}</code></span>`)}</dd></div>`}
      </dl></div>
    </section>
  `;
}

function Picked({ name }) {
  const summary = useSummary(name);
  if (!summary) return null;
  if (summary.error)
    return html`<div class="pane-head"><div><h1 class="mono">${name}</h1></div></div><p class="warn-text">${summary.error}</p>`;
  return html`
    <div class="pane-head">
      <div>
        <h1 class="mono">${name}</h1>
        <p>${plural(summary.examples, "example")} · ${plural(summary.questions.length, "question")} · ${fileSize(summary.bytes)}</p>
      </div>
      <div class="actions">
        <button class="btn primary" onClick=${() => fitOn(name)} title=${`A new fit on ${name}, on the Training page`}>Fit on It…</button>
        <button class="btn" onClick=${() => revealData(name)}>Show in Finder</button>
        <button class="btn" onClick=${() => deleteData(name)}>Delete…</button>
      </div>
    </div>
    ${summary.record && html`<${Source} name=${name} record=${summary.record} />`}
    <${Questions} summary=${summary} />
    <${ExampleTable} summary=${summary} />
  `;
}

function Centre() {
  const list = dataFiles.value;
  const name = pickedDataset();
  if (name && query.value.name !== name) showDataset(name);
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

const Inspector = () => html`<${ExampleInspector} name=${pickedDataset()} />`;

export default {
  id: "datasets",
  title: "Datasets",
  action,
  status,
  Centre,
  SidebarList: DatasetList,
  tabs: [{ id: "example", label: "Example", Body: Inspector }],
};
