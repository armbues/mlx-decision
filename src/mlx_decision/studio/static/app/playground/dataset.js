// The Playground on a dataset ("data:NAME"): its questions are the document,
// edited as a set's are and saved into its dataset.json; its rows (a sample of
// a split, about two per label value, or those a search finds) stand in for
// the examples, read-only, each with its label values. Start from a Question
// Set copies a set's questions in; Save as Question Set (files.js) copies the
// questions and the rows shown into sets/.
import { effect, signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import { closeSheet, openSheet } from "../components/sheet.js";
import { dataChanged, dataFiles, datasetOf, reloadData } from "../store/data.js";
import { sets } from "../store/sets.js";
import { toast } from "../store/ui.js";
import { answersFor, answersFromFile, fromBody, replaceQuestions, rowsFrom, toFile } from "./draft.js";
import { bodyText, datasetOfDocument, drafts, isDataset, isEdited, loaders, setEntry, setName } from "./session.js";

const encode = encodeURIComponent;

async function fetchDocument(name, { split = null, seed = 0, search = "" } = {}) {
  const params = new URLSearchParams({ seed: String(seed) });
  if (split) params.set("split", split);
  if (search.trim()) params.set("search", search.trim());
  return api(`/data/${encode(name)}/document?${params}`);
}

// What the page shows about the dataset besides its draft.
const factsOf = (doc, seed, search) => ({
  name: doc.name,
  split: doc.split,
  splits: doc.splits,
  columns: doc.columns,
  missing: doc.missing,
  column: doc.column,
  total: doc.total,
  matched: doc.matched,
  seed,
  search,
});

export async function loadDataset(documentName, options = {}) {
  const name = datasetOfDocument(documentName);
  if (!options.search) rowSearch.value = "";
  if (!dataFiles.value) await reloadData(); // its modified time: a change elsewhere reloads it
  try {
    const doc = await fetchDocument(name, options);
    const draft = fromBody({ state: "", questions: doc.questions });
    draft.examples = rowsFrom(doc.rows);
    draft.current = draft.examples[0].uid;
    draft.answers = answersFromFile(draft, doc.answers);
    setEntry(documentName, {
      sha256: doc.sha256,
      modified: datasetOf(name)?.modified ?? null,
      baseline: bodyText(draft, true),
      draft,
      onDisk: null,
      dataset: factsOf(doc, options.seed ?? 0, options.search ?? ""),
    });
  } catch (error) {
    setEntry(documentName, { error: error.message, status: error.status });
  }
}
loaders.dataset = loadDataset;

// Other rows: another split, another draw (Shuffle) or a search. The questions
// and their edits stay. ``rowSearch`` is the search box's text: empty again
// for another split or dataset.
export const rowsLoading = signal(false);
export const rowSearch = signal("");

export async function showRows(change) {
  const documentName = setName.value;
  const value = drafts.value[documentName];
  if (!value?.dataset) return;
  const options = { split: value.dataset.split, seed: value.dataset.seed, search: value.dataset.search, ...change };
  if (change.split !== undefined || change.search === "") rowSearch.value = "";
  rowsLoading.value = true;
  try {
    const doc = await fetchDocument(value.dataset.name, options);
    const current = drafts.value[documentName];
    const examples = rowsFrom(doc.rows);
    setEntry(documentName, {
      ...current,
      draft: { ...current.draft, examples, current: examples[0].uid },
      dataset: { ...current.dataset, ...factsOf(doc, options.seed, options.search) },
    });
  } catch (error) {
    toast(`Could not read the rows: ${error.message}`);
  } finally {
    rowsLoading.value = false;
  }
}

export const shuffle = () => showRows({ seed: (drafts.value[setName.value]?.dataset?.seed ?? 0) + 1, search: "" });

export async function saveDataset() {
  const documentName = setName.value;
  const value = drafts.value[documentName];
  if (!value?.draft || !value.dataset) return;
  const name = value.dataset.name;
  const { body, problems } = toFile(value.draft);
  const own = problems.filter((problem) => !problem.where?.example);
  if (own.length) {
    toast(`${own[0].message}: fix it before saving`);
    return;
  }
  try {
    const result = await api(`/data/${encode(name)}/questions`, {
      method: "PUT",
      body: { questions: body.questions, answers: answersFor(value.draft), loaded: value.sha256 },
    });
    const current = drafts.value[documentName];
    setEntry(documentName, {
      ...current,
      sha256: result.sha256,
      baseline: bodyText(current.draft, true),
      dataset: { ...current.dataset, missing: result.missing },
    });
    dataChanged();
    toast(`Saved data/${name}/dataset.json`);
  } catch (error) {
    if (error.status === 409 && error.current) changedOnDisk(documentName, name);
    else toast(`Not saved: ${error.message}`); // e.g. a fit or an evaluation reads it
  }
}

function changedOnDisk(documentName, name) {
  openSheet({
    title: `data/${name}/dataset.json changed on disk`,
    purpose: "Something changed it after you opened it, so saving would overwrite that. Your edits are not saved yet.",
    note: "Reload shows the dataset as it is now and drops your edits.",
    actions: [
      {
        label: "Reload",
        onSelect: () => {
          closeSheet();
          loadDataset(documentName);
          toast(`Reloaded ${name}; your edits are gone`);
        },
      },
    ],
  });
}

// Start from a Question Set: a set's questions replace the dataset's (an
// edit, saved with Save).
const startFrom = signal("");

export function startFromSet() {
  const documentName = setName.value;
  if (!drafts.value[documentName]?.dataset) return;
  const list = sets.value ?? [];
  if (!list.length) {
    toast("No question sets yet: make one in the Playground first");
    return;
  }
  startFrom.value = list[0].name;
  openSheet({
    title: "Start from a Question Set",
    purpose: "The set's questions replace the dataset's here; its examples stay where they are. Nothing is saved until you save.",
    Body: () => html`
      <label class="any-row"><span>Question set</span>
        <select aria-label="Question set" value=${startFrom.value} onChange=${(event) => (startFrom.value = event.target.value)}>
          ${list.map((item) => html`<option value=${item.name}>${item.name}</option>`)}
        </select></label>`,
    actions: () => [
      {
        label: "Use Its Questions",
        primary: true,
        onSelect: async () => {
          try {
            const { body } = await api(`/sets/${encode(startFrom.value)}`);
            const current = drafts.value[documentName];
            setEntry(documentName, { ...current, draft: replaceQuestions(current.draft, body.questions ?? {}) });
            closeSheet();
            toast(`The questions of ${startFrom.value}: save to keep them`);
          } catch (error) {
            toast(`Could not read ${startFrom.value}: ${error.message}`);
          }
        },
      },
    ],
  });
}

// A dataset open here that changes elsewhere (saved in another tab, deleted,
// downloaded again): reloaded when it has no edits, its split, draw and search
// kept; with edits it stays, and a save says it changed.
effect(() => {
  const list = dataFiles.value;
  if (!list) return;
  for (const [documentName, value] of Object.entries(drafts.peek())) {
    if (!isDataset(documentName) || !value.dataset && !value.error) continue;
    const item = list.find((entry) => entry.name === datasetOfDocument(documentName));
    const modified = item?.modified ?? null;
    if (value.dataset && modified === value.modified) continue;
    if (value.dataset && value.modified === null && item) {
      setEntry(documentName, { ...value, modified }); // opened before the list came: as it is
      continue;
    }
    if (!value.dataset && !item) continue; // still gone
    if (isEdited(value)) continue;
    const keep = value.dataset ? { split: value.dataset.split, seed: value.dataset.seed, search: value.dataset.search } : {};
    loadDataset(documentName, keep);
  }
});
