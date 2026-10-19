// The inspector's Answers: the last run of this set's selected example on the
// model, one card per question, stale where the question changed since; with
// several examples, also the last Run All as a grid (D161).
import { computed } from "@preact/signals";
import { html } from "htm/preact";
import { ChoiceBars, NoulBar, ScoreColumns, two } from "../charts/answers.js";
import * as icons from "../components/icons.js";
import { clock, duration } from "../format.js";
import { loadedNames } from "../store/status.js";
import { SHORTCUTS } from "../keys.js";
import { answerValue } from "./changes.js";
import { TYPE_LABELS, changedSince, toBody } from "./draft.js";
import { focusLater } from "./editor.js";
import { CompareView } from "./compare.js";
import { Grid } from "./grid.js";
import { batchFromHistory, cardContext, openRun, showExample } from "./history.js";
import {
  built,
  answersView as view,
  canRun,
  canRunAll,
  draft,
  example,
  lastBatch,
  lastRun,
  linked,
  modelName,
  run,
  runAll,
  running,
  several,
  stale,
} from "./session.js";

// How the answer moved since the run before on the same model, and whether
// its question was reworded in this run.
function Chips({ id, answer, edited }) {
  const context = cardContext.value;
  const chips = [];
  if (edited) chips.push(html`<span class="chip edit" key="edit">Edited</span>`);
  const previous = context?.previous?.response?.answers?.[id];
  if (previous) {
    const delta = answerValue(answer) - answerValue(previous, answer);
    if (Math.abs(delta) >= 0.005)
      chips.push(html`<span class=${`chip ${delta > 0 ? "up" : "down"} num`} key="delta"
        title=${`Change since run #${context.previousNumber}`}>${delta > 0 ? "+" : "−"}${two(Math.abs(delta))} since #${context.previousNumber}</span>`);
  }
  if (context?.reworded.has(id))
    chips.push(html`<button class="chip link" key="reworded" title="See what changed"
      onClick=${() => openRun(lastRun.value.id)}>Reworded in #${context.number}</button>`);
  return html`<div class="chips">${chips}</div>`;
}

// Hovering links a card and its question, unless a question is being edited.
export const hoverLink = (id) => {
  if (!document.activeElement?.closest?.(".question")) linked.value = id;
};

function Card({ question, answer, isStale, edited }) {
  const show = () => {
    linked.value = question.id;
    focusLater(`#q-${question.uid} .ins`);
  };
  const body =
    answer.type === "choice"
      ? html`<${ChoiceBars} answer=${answer} />`
      : answer.type === "score"
        ? html`<${ScoreColumns} answer=${answer} />`
        : html`<${NoulBar} answer=${answer} />`;
  return html`
    <article
      class=${`acard ${isStale ? "stale" : ""} ${linked.value === question.id ? "linked" : ""}`}
      data-q=${question.id}
      onMouseEnter=${() => hoverLink(question.id)}
    >
      <div class="acard-head">
        <button class="qid" title="Show the question" onClick=${show}>${question.id}</button>
        <span class="qtype">${TYPE_LABELS[answer.type] ?? answer.type}</span>
        ${answer.confidence != null &&
        html`<span class="right"><span class="conf num">Confidence <b>${two(answer.confidence)}</b></span></span>`}
      </div>
      <${Chips} id=${question.id} answer=${answer} edited=${edited} />
      ${body}
    </article>
  `;
}

function Placeholder({ question }) {
  return html`
    <article class=${`acard ${linked.value === question.id ? "linked" : ""}`} data-q=${question.id}
             onMouseEnter=${() => hoverLink(question.id)}>
      <div class="acard-head">
        <button class="qid" onClick=${() => focusLater(`#q-${question.uid} .ins`)}>${question.id}</button>
        <span class="qtype">${TYPE_LABELS[question.type]}</span>
      </div>
      <span class="dim">Not in this run. Run to answer it.</span>
    </article>
  `;
}

function StaleBanner() {
  const changed = stale.value;
  const run = lastRun.value;
  if (!changed.all && changed.ids.size === 0) return null;
  const what = changed.all
    ? html`You changed the state or its images since`
    : html`You edited <b>${[...changed.ids].join(", ")}</b> since`;
  return html`
    <div class="stale-banner">
      <${icons.StatusIcon} kind="warn" />
      <span class="msg">These answers are from ${clock(run.time)}. ${what}.</span>
      <button class="btn primary" disabled=${!canRun.value} onClick=${run_}>${several.value ? "Run This Example" : "Run"}</button>
    </div>
  `;
}
const run_ = () => run();

// The Run All shown for the model: this tab's latest, else the history's;
// with each example's row and what changed since (questions edited,
// examples edited or not in it). Null without one.
export const shownBatch = computed(() => {
  const current = draft.value;
  const name = modelName.value;
  if (!current || !name || current.examples.length < 2) return null;
  const shown = lastBatch.value ?? batchFromHistory(name);
  if (!shown) return null;
  const flight = running.value;
  const rows = shown.examples.map((exampleName) => {
    const result = shown.results[exampleName];
    const target = current.examples.find((item) => item.name === exampleName);
    if (!result) return { name: exampleName, pending: !!flight?.all };
    if (result.error) return { name: exampleName, error: result.error };
    const changed = target ? changedSince(toBody(current, target.uid).body, result.request) : { all: true, ids: new Set() };
    return { name: exampleName, run: result, stale: !target || changed.all || changed.ids.size > 0 };
  });
  const asked = rows.find((row) => row.run)?.run.request.questions;
  const questionsChanged = !!built.value && !!asked && JSON.stringify(asked) !== JSON.stringify(built.value.body.questions);
  const missing = current.examples.filter((item) => !shown.examples.includes(item.name)).map((item) => item.name);
  const done = rows.every((row) => row.run || row.error);
  const changed = questionsChanged || missing.length > 0 || rows.some((row) => row.stale);
  return { shown, rows, questionsChanged, missing, done, changed };
});

// One example: One Model / Compare; several: This Example / All Examples / Compare.
function ViewSwitch() {
  const many = several.value;
  const shown = view.value === "compare" ? "compare" : many ? view.value : "one";
  const choices = many
    ? [["one", "This Example"], ["all", "All Examples"], ["compare", "Compare"]]
    : [["one", "One Model"], ["compare", "Compare"]];
  return html`
    <div class="seg small answers-view" role="tablist" aria-label="Answers for">
      ${choices.map(
        ([id, label]) => html`<button role="tab" key=${id} aria-pressed=${shown === id}
          onClick=${() => (view.value = id)}>${label}</button>`,
      )}
    </div>
  `;
}

function AllExamples() {
  const current = draft.value;
  const name = modelName.value;
  const batch = shownBatch.value;
  if (!batch) {
    const flight = running.value;
    if (flight?.all)
      return html`<div class="empty"><b>Asking ${flight.model} about ${flight.example}…</b>${
        flight.loads ? html`<span>It loads first; that can take a while the first time.</span>` : null
      }</div>`;
    return html`
      <div class="empty">
        <b>No Run All on ${name} yet</b>
        <span>Run All (${SHORTCUTS.run}) asks each example in turn and shows the answers here, an example a row.</span>
        <div class="empty-actions"><button class="btn primary" disabled=${!canRunAll.value} onClick=${() => runAll()}>Run All</button></div>
      </div>
    `;
  }
  const { shown, rows, questionsChanged, missing } = batch;
  const questions = current.questions.map((question) => question.id);
  return html`
    <div class="run-head">
      <h3>${shown.number ? `Run All #${shown.number}` : "Run All"}<small class="num">${clock(shown.time)} · ${name}</small></h3>
    </div>
    ${(questionsChanged || missing.length > 0) &&
    html`<div class="stale-banner">
      <${icons.StatusIcon} kind="warn" />
      <span class="msg">${questionsChanged ? "Questions edited since this Run All. " : ""}${
        missing.length ? html`Not in it: <b>${missing.join(", ")}</b>. ` : ""
      }</span>
      <button class="btn primary" disabled=${!canRunAll.value} onClick=${() => runAll()}>Run All</button>
    </div>`}
    <${Grid} rows=${rows} questions=${questions} selected=${example.value?.name}
      onPick=${(picked) => {
        showExample(picked);
        view.value = "one";
      }} />
  `;
}

export function Answers() {
  const current = draft.value;
  const name = modelName.value;
  if (!current || !name)
    return html`<div class="empty"><b>Nothing to run yet</b><span>Pick a question set and a model.</span></div>`;
  const shown =
    view.value === "compare"
      ? html`<${CompareView} />`
      : view.value === "all" && current.examples.length > 1
        ? html`<${AllExamples} />`
        : html`<${OneExample} />`;
  return html`
    <${ViewSwitch} />
    ${shown}
  `;
}

function OneExample() {
  const current = draft.value;
  const result = lastRun.value;
  const name = modelName.value;
  const flight = running.value;
  const exampleName = current.examples.length > 1 ? example.value.name : null;
  if (!result) {
    if (flight && flight.example === example.value.name)
      return html`<div class="empty"><b>Asking ${flight.model}…</b>${
        flight.loads ? html`<span>It loads first; that can take a while the first time.</span>` : null
      }</div>`;
    const loads = !loadedNames.value.has(name);
    return html`
      <div class="empty">
        <b>No answers from ${name}${exampleName ? ` for ${exampleName}` : ""} yet</b>
        <span>${
          several.value
            ? `Run This Example (${SHORTCUTS.runExample}) answers its questions here, one card per question; Run All (${SHORTCUTS.run}) asks every example.`
            : `Run (${SHORTCUTS.run}) answers this set's questions here, one card per question.`
        }${loads ? " It loads the model first." : ""}</span>
      </div>
    `;
  }
  const changed = stale.value;
  const answers = result.response.answers ?? {};
  return html`
    <div class="run-head">
      <h3>${cardContext.value ? `Run #${cardContext.value.number}` : "Answers"}<small class="num">${clock(result.time)} · ${result.model}${exampleName ? ` · ${exampleName}` : ""}</small></h3>
    </div>
    <div class="run-stats num">
      <span><b>${duration(result.latency_ms)}</b></span>
      <span><b>${result.input_tokens}</b> input tokens</span>
      <span class=${result.truncated ? "warn" : "good"}>${result.truncated ? "The state was cut to fit" : "Nothing cut"}</span>
      <span>${result.load_ms != null ? `loaded in ${duration(result.load_ms)}` : "model already loaded"}</span>
    </div>
    <${StaleBanner} />
    <div class=${`answers ${flight ? "busy" : ""}`}>
      ${current.questions.map((question) => {
        const answer = answers[question.id];
        if (!answer || !(question.id in (result.request.questions ?? {})))
          return html`<${Placeholder} key=${question.uid} question=${question} />`;
        const edited = changed.ids.has(question.id);
        return html`<${Card} key=${question.uid} question=${question} answer=${answer}
                             isStale=${changed.all || edited} edited=${edited} />`;
      })}
    </div>
  `;
}
