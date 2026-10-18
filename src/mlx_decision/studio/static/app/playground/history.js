// The set's runs from its history file: the History tab, and what the answer
// cards say about the run before.
import { batch, computed, effect, signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import { two } from "../charts/answers.js";
import { closeSheet, openSheet } from "../components/sheet.js";
import { showTab } from "../components/inspector.js";
import { clock, duration, plural } from "../format.js";
import { workspaceChange } from "../stream.js";
import { toast } from "../store/ui.js";
import { answerMax, answerValue, requestChanges, summary, touched, valueLabel, wordDiff } from "./changes.js";
import { fromBody } from "./draft.js";
import { built, forgetRuns, lastRun, modelName, rememberRun, runs, setName, updateDraft } from "./session.js";

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

// Answers after a reload: the latest run of this set on this model, from the file.
effect(() => {
  const list = lines.value;
  const set = setName.peek();
  const name = modelName.value;
  if (!list || !name || lastRun.peek()) return;
  const latest = list.findLast((line) => line.model === name);
  if (!latest) return;
  api(`/history/${encodeURIComponent(set)}/${encodeURIComponent(latest.id)}`)
    .then((line) => {
      if (!runs.peek()[`${set}\u0000${name}`]) rememberRun(set, name, line);
    })
    .catch(() => {});
});

// Run numbers: #1 for the first line of the file.
const numberOf = (list, id) => list.findIndex((line) => line.id === id) + 1;
const previousOn = (list, index, model) => {
  for (let i = index - 1; i >= 0; i -= 1) if (list[i].model === model) return i;
  return -1;
};

// For the answer cards: the run shown, the run before on the same model,
// the questions reworded in the run shown.
export const cardContext = computed(() => {
  const list = lines.value;
  const shown = lastRun.value;
  if (!list || !shown) return null;
  const index = list.findIndex((line) => line.id === shown.id);
  if (index < 0) return null;
  const before = previousOn(list, index, shown.model);
  const previous = before >= 0 ? list[before] : null;
  return {
    number: index + 1,
    previous,
    previousNumber: before + 1,
    reworded: previous ? touched(requestChanges(previous.request, list[index].request)) : new Set(),
  };
});

// The run picked in the History tab (by id), per set; the latest when unset.
const picked = signal({});
export const selected = computed(() => {
  const list = lines.value;
  if (!list?.length) return null;
  const id = picked.value[setName.value];
  return list.find((line) => line.id === id) ?? list.at(-1);
});

export function pick(id) {
  picked.value = { ...picked.value, [setName.value]: id };
}

export function openRun(id) {
  pick(id);
  showTab("playground", "history");
}

// ---------- the tab ----------

function RunsList({ list }) {
  const current = selected.value;
  return html`
    <div class="hist-list" role="listbox" aria-label="Runs">
      ${[...list].reverse().map((line) => {
        const number = numberOf(list, line.id);
        const before = list[number - 2];
        const what = summary(before ? requestChanges(before.request, line.request) : [], number === 1);
        return html`
          <button class="hrow" role="option" key=${line.id} aria-selected=${line.id === current.id}
                  onClick=${() => pick(line.id)}>
            <span class="n num">#${number}</span>
            <span class="what"><b>${what}</b><span class="num">${line.model} · ${duration(line.latency_ms)}</span></span>
            <span class="t num">${clock(line.time)}</span>
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

function Changes({ list, line }) {
  const number = numberOf(list, line.id);
  const before = list[number - 2];
  const restorable = built.value && JSON.stringify(built.value.body.questions) !== JSON.stringify(line.request.questions);
  const restore = () => {
    updateDraft((draft) => ({ ...draft, questions: fromBody({ questions: line.request.questions }).questions }));
    toast(`Restored the questions of run #${number}; save to keep them`);
  };
  const head = (title) => html`
    <div class="diff-head">
      <h4>${title}</h4>
      ${restorable && html`<button class="btn" title="Put this run's questions back in the editor" onClick=${restore}>Restore These Questions</button>`}
    </div>
  `;
  if (!before) {
    const questions = Object.entries(line.request.questions ?? {});
    return html`
      <div class="diff">
        ${head("First run, asked with")}
        ${questions.map(
          ([id, question]) => html`
            <div class="dline" key=${id}>
              <span class="where">${id} · ${question.type}</span>
              <span class="txt">${typeof question.instructions === "string" ? question.instructions : "(no instructions)"}</span>
            </div>
          `,
        )}
      </div>
    `;
  }
  const changes = requestChanges(before.request, line.request);
  return html`
    <div class="diff">
      ${head(`Changed since run #${number - 1}${before.model === line.model ? "" : ` (${before.model})`}`)}
      ${changes.length === 0 && html`<span class="dim">The same request as run #${number - 1}.</span>`}
      ${changes.map(
        (change) => html`
          <div class="dline" key=${change.where}>
            <span class="where">${change.where}</span>
            <span class="txt">${change.note ?? html`<${Diff} before=${change.before} after=${change.after} />`}</span>
          </div>
        `,
      )}
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

function Moves({ list, line }) {
  const same = list.filter((entry) => entry.model === line.model);
  const mark = same.findIndex((entry) => entry.id === line.id);
  const answers = Object.entries(line.response?.answers ?? {});
  const scoreIds = answers.filter(([, answer]) => answer.type === "score").map(([id]) => id);
  const recent = same.slice(Math.max(0, mark - 7), mark + 1);
  return html`
    <div class="moves">
      <h4>How the answers moved on ${line.model}, ${plural(same.length, "run")}</h4>
      ${answers.map(([id, answer]) => {
        const values = same.map((entry) => answerValue(entry.response?.answers?.[id], answer));
        const previous = mark > 0 ? values[mark - 1] : null;
        const delta = previous == null ? null : values[mark] - previous;
        const kind = delta == null ? "flat" : delta > 0.005 ? "up" : delta < -0.005 ? "down" : "flat";
        const before = mark > 0 ? numberOf(list, same[mark - 1].id) : null;
        return html`
          <div class="move" key=${id}>
            <span class="qid">${id}<small>${valueLabel(answer)}</small></span>
            <${Sparkline} values=${values} max=${answerMax(answer)} mark=${mark} />
            <span class=${`delta num ${kind}`}>${two(values[mark])}<small>${
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
                  <span class="num">#${numberOf(list, entry.id)}</span>
                  <span class="stack">${Object.values(answer.probabilities).map(
                    (p, level) => html`<i style=${`width:${(p * 100).toFixed(1)}%; background:${shades[level] ?? "var(--fg-3)"}`}
                                          title=${`level ${level}: ${two(p)}`}></i>`,
                  )}</span>
                  <span class="num">${two(answer.score)}</span>
                </div>
              `;
            })}
          </div>
        `;
      })}
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
  const line = selected.value;
  return html`
    <${RunsList} list=${list} />
    <${Changes} list=${list} line=${line} />
    <${Moves} list=${list} line=${line} />
    <div class="hist-foot">
      <span class="dim"><code>history/${setName.value}.jsonl</code> · ${plural(list.length, "run")}</span>
      <button class="btn plain danger" onClick=${clearHistory}>Clear History…</button>
    </div>
  `;
}
