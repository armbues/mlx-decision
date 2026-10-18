// The set's runs from its history file: the History tab, and what the answer
// cards say about the run before.
//
// Lines are grouped into entries: a single run, or a Run All (the lines that
// share a `batch` id, D161). Entries are numbered from #1 (the first in the
// file); a line belongs to an example (`example`, or the set's first for
// lines from before examples).
import { batch, computed, effect, signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import { two } from "../charts/answers.js";
import { closeSheet, openSheet } from "../components/sheet.js";
import { showTab } from "../components/inspector.js";
import { clock, duration, plural } from "../format.js";
import { workspaceChange } from "../stream.js";
import { toast } from "../store/ui.js";
import { answerMax, answerValue, requestChanges, shownValue, summary, touched, valueLabel, wordDiff } from "./changes.js";
import { FIRST_EXAMPLE, exampleByName, fromBody, selectExample } from "./draft.js";
import { Grid } from "./grid.js";
import {
  built,
  draft,
  example,
  forgetRuns,
  lastRun,
  modelName,
  rememberRun,
  runKey,
  runs,
  setName,
  updateDraft,
} from "./session.js";

// Set name -> {runs} (oldest first, images as references) or {error}.
const histories = signal({});

async function loadHistory(name) {
  try {
    const { runs: lines } = await api(`/history/${encodeURIComponent(name)}`);
    histories.value = { ...histories.value, [name]: { runs: lines } };
  } catch (error) {
    histories.value = { ...histories.value, [name]: { error: error.message } };
  }
}

effect(() => {
  const name = setName.value;
  if (name !== null && !histories.peek()[name]) loadHistory(name);
});

effect(() => {
  const change = workspaceChange.value;
  if (change?.folder === "history" && histories.peek()[change.name]) loadHistory(change.name);
  if (change?.folder === "sets" && change.action === "renamed") loadHistory(change.name);
});

export const lines = computed(() => histories.value[setName.value]?.runs ?? null);

// The example a line asked: its own name, else the set's first example.
const firstName = computed(() => draft.value?.examples[0]?.name ?? FIRST_EXAMPLE);
export const exampleOf = (line) => line.example ?? firstName.value;

// Entries: [{key, batch, lines, number}], oldest first.
export const entries = computed(() => {
  const list = lines.value;
  if (!list) return null;
  const grouped = [];
  for (const line of list) {
    const last = grouped.at(-1);
    if (line.batch && last?.batch === line.batch) last.lines.push(line);
    else grouped.push({ key: line.batch ? `batch:${line.batch}` : line.id, batch: line.batch ?? null, lines: [line] });
  }
  return grouped.map((entry, index) => ({ ...entry, number: index + 1 }));
});

const entryOfLine = computed(() => {
  const map = new Map();
  for (const entry of entries.value ?? []) for (const line of entry.lines) map.set(line.id, entry);
  return map;
});
const numberOfLine = (id) => entryOfLine.value.get(id)?.number ?? 0;

// The line before `index` on the same model and example (-1 when none).
function previousOf(list, index, model, name) {
  for (let i = index - 1; i >= 0; i -= 1)
    if ((model === null || list[i].model === model) && exampleOf(list[i]) === name) return i;
  return -1;
}

// Answers after a reload: the latest run of this set on this model and example.
effect(() => {
  const list = lines.value;
  const set = setName.peek();
  const name = modelName.value;
  const current = example.value?.name;
  if (!list || !name || !current || lastRun.peek()) return;
  const latest = list.findLast((line) => line.model === name && exampleOf(line) === current);
  if (!latest) return;
  api(`/history/${encodeURIComponent(set)}/${encodeURIComponent(latest.id)}`)
    .then((line) => {
      if (!runs.peek()[runKey(set, name, current)]) rememberRun(set, name, current, line);
    })
    .catch(() => {});
});

// For the answer cards: the run shown, the run before on the same model and
// example, the questions reworded in the run shown.
export const cardContext = computed(() => {
  const list = lines.value;
  const shown = lastRun.value;
  if (!list || !shown) return null;
  const index = list.findIndex((line) => line.id === shown.id);
  if (index < 0) return null;
  const before = previousOf(list, index, shown.model, exampleOf(list[index]));
  const previous = before >= 0 ? list[before] : null;
  return {
    number: numberOfLine(list[index].id),
    previous,
    previousNumber: previous ? numberOfLine(previous.id) : 0,
    reworded: previous ? touched(requestChanges(previous.request, list[index].request)) : new Set(),
  };
});

// The latest Run All on a model in the history: {id, time, examples, results}.
export function batchFromHistory(model) {
  const all = entries.value ?? [];
  const entry = all.findLast((item) => item.batch && item.lines[0].model === model);
  if (!entry) return null;
  return {
    id: entry.batch,
    time: entry.lines[0].time,
    number: entry.number,
    examples: entry.lines.map(exampleOf),
    results: Object.fromEntries(entry.lines.map((line) => [exampleOf(line), line])),
  };
}

// The entry picked in the History tab (by key), per set; the latest when unset.
const picked = signal({});
export const selected = computed(() => {
  const list = entries.value;
  if (!list?.length) return null;
  const key = picked.value[setName.value];
  return list.find((entry) => entry.key === key) ?? list.at(-1);
});

export function pick(key) {
  picked.value = { ...picked.value, [setName.value]: key };
}

// Open the History at the entry holding the line `id`.
export function openRun(id) {
  const entry = entryOfLine.value.get(id);
  if (entry) pick(entry.key);
  showTab("playground", "history");
}

// Show an example in the editor and the Answers tab.
export function showExample(name) {
  const target = draft.value && exampleByName(draft.value, name);
  if (!target) {
    toast(`${name} is no longer an example of this set`);
    return;
  }
  updateDraft((value) => selectExample(value, target.uid));
  showTab("playground", "answers");
}

// ---------- the tab ----------

const many = computed(() => (draft.value?.examples.length ?? 1) > 1);

function entrySummary(list, entry) {
  const all = entries.value;
  if (entry.batch) {
    const before = all[entry.number - 2];
    const questions = (line) => ({ questions: line.request.questions });
    const changes = before ? requestChanges(questions(before.lines[0]), questions(entry.lines[0])) : [];
    const what = before ? (changes.length ? summary(changes, false) : "Same questions") : "First run";
    return { what: `Run All · ${what}`, meta: `${entry.lines[0].model} · ${plural(entry.lines.length, "example")}` };
  }
  const [line] = entry.lines;
  const index = list.findIndex((item) => item.id === line.id);
  const before = previousOf(list, index, null, exampleOf(line));
  const what = summary(before >= 0 ? requestChanges(list[before].request, line.request) : [], before < 0);
  const name = many.value || line.example ? ` · ${exampleOf(line)}` : "";
  return { what, meta: `${line.model}${name} · ${duration(line.latency_ms)}` };
}

function EntriesList({ list }) {
  const current = selected.value;
  return html`
    <div class="hist-list" role="listbox" aria-label="Runs">
      ${[...entries.value].reverse().map((entry) => {
        const { what, meta } = entrySummary(list, entry);
        return html`
          <button class=${`hrow ${entry.batch ? "batch" : ""}`} role="option" key=${entry.key}
                  aria-selected=${entry.key === current.key} onClick=${() => pick(entry.key)}>
            <span class="n num">#${entry.number}</span>
            <span class="what"><b>${what}</b><span class="num">${meta}</span></span>
            <span class="t num">${clock(entry.lines[0].time)}</span>
          </button>
        `;
      })}
    </div>
  `;
}

const Diff = ({ before, after }) => html`
  ${wordDiff(before, after).map((part) =>
    part.kind === "same" ? part.text : part.kind === "del" ? html`<del>${part.text}</del>` : html`<ins>${part.text}</ins>`,
  )}
`;

function restoreButton(questions, number) {
  const restorable = built.value && JSON.stringify(built.value.body.questions) !== JSON.stringify(questions);
  if (!restorable) return null;
  const restore = () => {
    updateDraft((value) => ({ ...value, questions: fromBody({ questions }).questions }));
    toast(`Restored the questions of run #${number}; save to keep them`);
  };
  return html`<button class="btn" title="Put this run's questions back in the editor" onClick=${restore}>Restore These Questions</button>`;
}

function ChangeLines({ changes }) {
  return changes.map(
    (change) => html`
      <div class="dline" key=${change.where}>
        <span class="where">${change.where}</span>
        <span class="txt">${change.note ?? html`<${Diff} before=${change.before} after=${change.after} />`}</span>
      </div>
    `,
  );
}

function FirstQuestions({ request }) {
  return Object.entries(request.questions ?? {}).map(
    ([id, question]) => html`
      <div class="dline" key=${id}>
        <span class="where">${id} · ${question.type}</span>
        <span class="txt">${typeof question.instructions === "string" ? question.instructions : "(no instructions)"}</span>
      </div>
    `,
  );
}

function Changes({ list, entry }) {
  const [line] = entry.lines;
  const head = (title) => html`
    <div class="diff-head">
      <h4>${title}</h4>
      ${restoreButton(line.request.questions, entry.number)}
    </div>
  `;
  if (entry.batch) {
    const before = entries.value[entry.number - 2];
    const questions = (item) => ({ questions: item.request.questions });
    if (!before) return html`<div class="diff">${head("First run, asked with")}<${FirstQuestions} request=${line.request} /></div>`;
    const changes = requestChanges(questions(before.lines[0]), questions(line));
    return html`
      <div class="diff">
        ${head(`Questions changed since #${before.number}`)}
        ${changes.length === 0 && html`<span class="dim">The same questions as #${before.number}.</span>`}
        <${ChangeLines} changes=${changes} />
      </div>
    `;
  }
  const index = list.findIndex((item) => item.id === line.id);
  const name = exampleOf(line);
  const before = previousOf(list, index, null, name);
  if (before < 0)
    return html`
      <div class="diff">
        ${head(many.value ? `First run of ${name}, asked with` : "First run, asked with")}
        <${FirstQuestions} request=${line.request} />
      </div>
    `;
  const previous = list[before];
  const number = numberOfLine(previous.id);
  const changes = requestChanges(previous.request, line.request);
  return html`
    <div class="diff">
      ${head(`Changed since #${number}${previous.model === line.model ? "" : ` (${previous.model})`}`)}
      ${changes.length === 0 && html`<span class="dim">The same request as #${number}.</span>`}
      <${ChangeLines} changes=${changes} />
    </div>
  `;
}

function Sparkline({ values, max, mark }) {
  const [width, height, pad] = [200, 34, 4];
  const points = values.map((value, index) => ({ value, index })).filter((point) => point.value != null);
  if (!points.length) return html`<svg viewBox="0 0 200 34"></svg>`;
  const x = (index) => pad + (width - 2 * pad) * (values.length === 1 ? 0.5 : index / (values.length - 1));
  const y = (value) => height - pad - (height - 2 * pad) * (value / max);
  const line = points.map((point) => `${x(point.index)},${y(point.value)}`).join(" ");
  return html`
    <svg viewBox=${`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
      <line x1=${pad} x2=${width - pad} y1=${y(0)} y2=${y(0)} stroke="var(--sep)" />
      <line x1=${pad} x2=${width - pad} y1=${y(max)} y2=${y(max)} stroke="var(--sep)" stroke-dasharray="2 3" />
      <polyline points=${line} fill="none" stroke="var(--accent)" stroke-width="1.6" vector-effect="non-scaling-stroke" />
      ${points.map(
        (point) => html`<circle cx=${x(point.index)} cy=${y(point.value)} r=${point.index === mark ? 3.6 : 2.2}
          fill=${point.index === mark ? "var(--accent)" : "var(--group)"} stroke="var(--accent)" stroke-width="1.4"
          vector-effect="non-scaling-stroke" />`,
      )}
    </svg>
  `;
}

const SHADES = (count) =>
  Array.from({ length: count }, (_, level) =>
    level === count - 1 ? "var(--accent)" : `color-mix(in srgb, var(--accent) ${Math.round(18 + (60 * level) / Math.max(1, count - 1))}%, var(--track))`,
  );

// How one example's answers moved over its runs on one model.
function Moves({ list, line }) {
  const name = exampleOf(line);
  const same = list.filter((entry) => entry.model === line.model && exampleOf(entry) === name);
  const mark = same.findIndex((entry) => entry.id === line.id);
  const answers = Object.entries(line.response?.answers ?? {});
  const scoreIds = answers.filter(([, answer]) => answer.type === "score").map(([id]) => id);
  const recent = same.slice(Math.max(0, mark - 7), mark + 1);
  return html`
    <div class="moves">
      <h4>How the answers moved on ${line.model}${many.value ? ` for ${name}` : ""}, ${plural(same.length, "run")}</h4>
      ${answers.map(([id, answer]) => {
        const values = same.map((entry) => answerValue(entry.response?.answers?.[id], answer));
        const previous = mark > 0 ? values[mark - 1] : null;
        const delta = previous == null ? null : values[mark] - previous;
        const kind = delta == null ? "flat" : delta > 0.005 ? "up" : delta < -0.005 ? "down" : "flat";
        const before = mark > 0 ? numberOfLine(same[mark - 1].id) : null;
        return html`
          <div class="move" key=${id}>
            <span class="qid">${id}<small>${valueLabel(answer)}</small></span>
            <${Sparkline} values=${values} max=${answerMax(answer)} mark=${mark} />
            <span class=${`delta num ${kind}`}>${two(shownValue(answer, values[mark]))}<small>${
              delta == null ? "first run" : `${delta >= 0 ? "+" : "−"}${two(Math.abs(delta))} vs #${before}`
            }</small></span>
          </div>
        `;
      })}
      ${scoreIds.map((id) => {
        const levels = Object.keys(line.response.answers[id].probabilities).length;
        const shades = SHADES(levels);
        return html`
          <h4 key=${`${id}-head`}>${id} by level</h4>
          <div class="stack-rows" key=${`${id}-rows`}>
            ${recent.map((entry) => {
              const answer = entry.response?.answers?.[id];
              if (answer?.type !== "score") return null;
              return html`
                <div class=${`stack-row ${entry.id === line.id ? "sel" : ""}`} key=${entry.id}>
                  <span class="num">#${numberOfLine(entry.id)}</span>
                  <span class="stack">${Object.values(answer.probabilities).map(
                    (p, level) => html`<i style=${`width:${(p * 100).toFixed(1)}%; background:${shades[level] ?? "var(--fg-3)"}`}
                                          title=${`level ${level + 1}: ${two(p)}`}></i>`,
                  )}</span>
                  <span class="num">${two(answer.score + 1)}</span>
                </div>
              `;
            })}
          </div>
        `;
      })}
    </div>
  `;
}

// A Run All entry: its grid (click a row to show that example).
function BatchAnswers({ entry }) {
  const questions = Object.keys(entry.lines[0].request.questions ?? {});
  const rows = entry.lines.map((line) => ({ name: exampleOf(line), run: line }));
  return html`
    <div class="moves">
      <h4>Answers on ${entry.lines[0].model}, ${plural(rows.length, "example")}</h4>
      <${Grid} rows=${rows} questions=${questions} selected=${example.value?.name} onPick=${showExample} />
    </div>
  `;
}

function clearHistory() {
  const set = setName.value;
  const count = lines.value?.length ?? 0;
  openSheet({
    title: `Clear the history of ${set}?`,
    purpose: `Deletes history/${set}.jsonl (${plural(count, "run")}) and the images no other set's history uses. The set itself stays.`,
    actions: [
      {
        label: "Clear History",
        primary: true,
        onSelect: async () => {
          closeSheet();
          try {
            await api(`/history/${encodeURIComponent(set)}`, { method: "DELETE" });
            batch(() => {
              histories.value = { ...histories.value, [set]: { runs: [] } };
              forgetRuns(set);
            });
            toast(`Cleared the history of ${set}`);
          } catch (error) {
            toast(`Not cleared: ${error.message}`);
          }
        },
      },
    ],
  });
}

export function History() {
  const entry = histories.value[setName.value];
  if (!setName.value) return html`<div class="empty"><b>No question set open</b></div>`;
  if (entry?.error) return html`<div class="empty"><b>${entry.error}</b></div>`;
  const list = entry?.runs;
  if (!list) return null;
  if (list.length === 0)
    return html`
      <div class="empty">
        <b>No runs of this set yet</b>
        <span>Every run is kept in the workspace's <code>history/</code> folder until you clear it.</span>
      </div>
    `;
  const shown = selected.value;
  return html`
    <${EntriesList} list=${list} />
    <${Changes} list=${list} entry=${shown} />
    ${shown.batch
      ? html`<${BatchAnswers} entry=${shown} />`
      : html`<${Moves} list=${list} line=${shown.lines[0]} />`}
    <div class="hist-foot">
      <span class="dim"><code>history/${setName.value}.jsonl</code> · ${plural(entries.value.length, "entry", "entries")} · ${plural(list.length, "run")}</span>
      <button class="btn plain danger" onClick=${clearHistory}>Clear History…</button>
    </div>
  `;
}
