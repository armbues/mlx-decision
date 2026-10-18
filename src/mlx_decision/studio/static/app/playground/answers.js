// The inspector's Answers: the last run on this set and model, one card per
// question, stale where the question changed since.
import { html } from "htm/preact";
import { ChoiceBars, NoulBar, ScoreColumns, two } from "../charts/answers.js";
import * as icons from "../components/icons.js";
import { clock, duration } from "../format.js";
import { loadedNames } from "../store/status.js";
import { answerValue } from "./changes.js";
import { TYPE_LABELS } from "./draft.js";
import { focusLater } from "./editor.js";
import { cardContext, openRun } from "./history.js";
import { canRun, draft, lastRun, linked, modelName, run, running, stale } from "./session.js";

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
      <button class="btn primary" disabled=${!canRun.value} onClick=${run_}>Run</button>
    </div>
  `;
}
const run_ = () => run();

export function Answers() {
  const current = draft.value;
  const result = lastRun.value;
  const name = modelName.value;
  const flight = running.value;
  if (!current || !name)
    return html`<div class="empty"><b>Nothing to run yet</b><span>Pick a question set and a model.</span></div>`;
  if (!result) {
    if (flight)
      return html`<div class="empty"><b>Asking ${flight.model}…</b>${
        flight.loads ? html`<span>It loads first; that can take a while the first time.</span>` : null
      }</div>`;
    const loads = !loadedNames.value.has(name);
    return html`
      <div class="empty">
        <b>No answers from ${name} yet</b>
        <span>Run (⌘↩) answers this set's questions here, one card per question.${
          loads ? " It loads the model first." : ""
        }</span>
      </div>
    `;
  }
  const changed = stale.value;
  const answers = result.response.answers ?? {};
  return html`
    <div class="run-head">
      <h3>${cardContext.value ? `Run #${cardContext.value.number}` : "Answers"}<small class="num">${clock(result.time)} · ${result.model}</small></h3>
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
