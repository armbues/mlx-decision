// A fit or an evaluation in the centre of the Training page. A fit: while it runs, the stage stepper,
// the step line with the time left (measured: the events' own estimate), the
// loss chart for the kinds that train in epochs and the log in the CLI's
// words; once done, its report (before and after on the kept-aside examples,
// the kept epoch, where it was saved, the command that does the same, Open in
// Playground). An evaluation: each model's load and scoring, then a table of
// both models per question with the differences. Stopped or failed: what
// happened, and the log.
import { html } from "htm/preact";
import { useEffect, useRef, useState } from "preact/hooks";
import { api } from "../api.js";
import { plural, shellWord } from "../format.js";
import { chooseModel } from "../playground/session.js";
import { href } from "../router.js";
import { fitById, opened, openFit } from "../store/fits.js";
import { stopJob } from "../store/jobs.js";
import { models } from "../store/status.js";
import { toast } from "../store/ui.js";

const STEPS = [
  ["check", "Check"],
  ["before", "Score before"],
  ["train", "Train"],
  ["calibrate", "Calibrate"],
  ["after", "Score after"],
  ["write", "Write"],
];
const LABELS = { ...Object.fromEntries(STEPS), load: "Load", score: "Score" };
const METRICS = [
  ["accuracy", "Accuracy", true],
  ["macro_f1", "Macro-F1", true],
  ["ece", "Calibration error (ECE)", false],
  ["mae", "Mean error (MAE, levels)", false],
];

// "42 s", "3 min 5 s", "1 h 12 min", as the CLI writes it.
export function seconds(value) {
  const total = Math.round(value);
  if (total < 60) return `${total} s`;
  const minutes = Math.floor(total / 60);
  if (minutes < 60) return `${minutes} min ${total % 60} s`;
  return `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

const percent = (value) => (value == null ? "–" : `${(value * 100).toFixed(1)}%`);

// The kinds the fit runs (its check event says), and the step a stage belongs to.
function kindsOf(events) {
  return events.find((event) => event?.stage === "check" && event.metrics?.kinds)?.metrics.kinds ?? null;
}
function stepOf(stage, trainer) {
  if (stage === "features") return trainer ? "train" : "calibrate";
  return LABELS[stage] ? stage : null;
}

// ---------- while it runs ----------

function Stepper({ events, job }) {
  const kinds = kindsOf(events);
  const trainer = kinds && kinds[0] !== "calibrate" ? kinds[0] : null;
  const steps = STEPS.filter(([id]) => id !== "train" || trainer || !kinds);
  const done = job.state === "ended" && job.outcome === "done";
  let current = null;
  for (const event of events) {
    const step = event && stepOf(event.stage, trainer);
    if (step) current = step;
  }
  const at = done ? steps.length : Math.max(0, steps.findIndex(([id]) => id === current));
  const failed = job.state === "ended" && !done;
  return html`
    <ol class="stepper" aria-label="Stages">
      ${steps.map(([id, label], index) => {
        const state = index < at ? "done" : index === at ? (failed ? "stopped" : "now") : "next";
        return html`<li key=${id} class=${state} aria-current=${state === "now" ? "step" : undefined}>
          <span class="dot" aria-hidden="true"></span><span>${label}</span></li>`;
      })}
    </ol>
  `;
}

function StepLine({ events }) {
  const last = [...events].reverse().find((event) => event && event.step != null && event.total);
  if (!last) return html`<p class="step-line">Starting…</p>`;
  const fraction = Math.min(1, last.step / last.total);
  const label = last.stage === "features" ? "Reading the examples" : (LABELS[last.stage] ?? last.stage);
  const parts = [
    `${last.step.toLocaleString("en")} of ${last.total.toLocaleString("en")}`,
    `${Math.floor(fraction * 100)}%`,
  ];
  if (last.eta_s != null && last.step < last.total) parts.push(`${seconds(last.eta_s)} left`);
  return html`
    <div class="step-line">
      <span><b>${label}</b> · ${parts.join(" · ")}</span>
      <div class="bar"><i style=${`width:${(fraction * 100).toFixed(1)}%`}></i></div>
    </div>
  `;
}

// An evaluation's steps: each model's load and scoring, in turn.
function EvalStepper({ events, job }) {
  const names = [job.details?.model, job.details?.compare].filter(Boolean);
  const steps = names.flatMap((name) => [`Load ${name}`, `Score ${name}`]);
  const done = job.state === "ended" && job.outcome === "done";
  let at = 0;
  let loads = 0;
  for (const event of events) {
    if (event?.stage === "load") loads += 1;
    if (event?.stage === "load" || event?.stage === "score") at = Math.max(0, (loads - 1) * 2 + (event.stage === "score" ? 1 : 0));
  }
  if (done) at = steps.length;
  const failed = job.state === "ended" && !done;
  return html`
    <ol class="stepper" aria-label="Stages">
      ${steps.map((label, index) => {
        const state = index < at ? "done" : index === at ? (failed ? "stopped" : "now") : "next";
        return html`<li key=${label} class=${state} aria-current=${state === "now" ? "step" : undefined}>
          <span class="dot" aria-hidden="true"></span><span>${label}</span></li>`;
      })}
    </ol>
  `;
}

// ---------- the loss chart ----------

const CHART = { width: 640, height: 180, left: 44, right: 16, top: 12, bottom: 26 };

// The loss per step, and the dev loss per epoch at the step its epoch ended.
function lossSeries(events) {
  const steps = [];
  const epochs = [];
  let lastStep = 0;
  for (const event of events) {
    if (!event || event.stage !== "train") continue;
    if (event.metrics?.loss != null && event.step != null) {
      steps.push([event.step, event.metrics.loss]);
      lastStep = event.step;
    } else if (event.metrics?.epoch != null) {
      epochs.push({ ...event.metrics, step: lastStep });
    }
  }
  return { steps, epochs };
}

function ticks(low, high) {
  const span = high - low || 1;
  const step = 10 ** Math.floor(Math.log10(span / 3));
  const nice = [1, 2, 5, 10].map((f) => f * step).find((s) => span / s <= 5) ?? step * 10;
  const first = Math.ceil(low / nice) * nice;
  const result = [];
  for (let value = first; value <= high + 1e-9; value += nice) result.push(Number(value.toFixed(6)));
  return result;
}

// ``done``: the fit ended done, so the epoch with the lowest dev loss is the kept one.
function LossChart({ events, done }) {
  const { steps, epochs } = lossSeries(events);
  const [hover, setHover] = useState(null); // index into steps
  if (steps.length === 0) return null;
  const kept = done && epochs.length ? epochs.reduce((best, row) => (row.dev_loss < best.dev_loss ? row : best)).epoch : null;
  const total = steps[steps.length - 1][0];
  const values = [...steps.map(([, loss]) => loss), ...epochs.map((row) => row.dev_loss)].filter((v) => v != null);
  const low = Math.min(0, ...values);
  const high = Math.max(...values) * 1.05 || 1;
  const { width, height, left, right, top, bottom } = CHART;
  const x = (step) => left + ((step - 0) / Math.max(1, total)) * (width - left - right);
  const y = (value) => top + (1 - (value - low) / (high - low)) * (height - top - bottom);
  const line = steps.map(([step, loss]) => `${x(step).toFixed(1)},${y(loss).toFixed(1)}`).join(" ");
  const onMove = (event) => {
    const box = event.currentTarget.getBoundingClientRect();
    const at = ((event.clientX - box.left) / box.width) * width;
    let best = 0;
    for (let i = 1; i < steps.length; i += 1) if (Math.abs(x(steps[i][0]) - at) < Math.abs(x(steps[best][0]) - at)) best = i;
    setHover(best);
  };
  const point = hover != null ? steps[hover] : null;
  const epochAt = point ? epochs.find((row) => row.step >= point[0]) : null;
  return html`
    <section class="gsec loss-sec"><div class="gsec-head"><h2>Loss</h2>
        <span class="legend"><span class="key line"></span>training loss, per step<span class="key dot"></span>dev loss, per epoch</span></div>
      <div class="group loss-group">
        <div class="loss-chart">
          <svg viewBox=${`0 0 ${width} ${height}`} role="img"
               aria-label=${`Training loss over ${total} steps${epochs.length ? `, dev loss after ${plural(epochs.length, "epoch")}` : ""}`}
               onPointerMove=${onMove} onPointerLeave=${() => setHover(null)}>
            ${ticks(low, high).map(
              (value) => html`<g key=${value}><line class="grid" x1=${left} x2=${width - right} y1=${y(value)} y2=${y(value)} />
                <text class="tick" x=${left - 6} y=${y(value) + 3.5} text-anchor="end">${value}</text></g>`,
            )}
            <text class="tick" x=${left} y=${height - 8}>step 0</text>
            <text class="tick" x=${width - right} y=${height - 8} text-anchor="end">${total.toLocaleString("en")}</text>
            <polyline class="loss-line" points=${line} />
            ${epochs.map(
              (row) => html`<g key=${row.epoch}>
                <circle class="dev-dot" cx=${x(row.step)} cy=${y(row.dev_loss)} r="4.5" />
                ${row.epoch === kept && html`<text class="kept" x=${x(row.step)} y=${y(row.dev_loss) - 10} text-anchor="middle">kept</text>`}
              </g>`,
            )}
            ${point && html`<line class="cross" x1=${x(point[0])} x2=${x(point[0])} y1=${top} y2=${height - bottom} />
              <circle class="hover-dot" cx=${x(point[0])} cy=${y(point[1])} r="4" />`}
          </svg>
          ${point &&
          html`<div class="chart-tip" style=${`left:${((x(point[0]) / width) * 100).toFixed(1)}%`}>
            <div><b class="num">${point[1].toFixed(4)}</b><span><i class="key line"></i>loss at step ${point[0].toLocaleString("en")}</span></div>
            ${epochAt && html`<div><b class="num">${epochAt.dev_loss.toFixed(4)}</b><span><i class="key dot"></i>dev loss, epoch ${epochAt.epoch}</span></div>`}
          </div>`}
        </div>
        ${epochs.length > 0 &&
        html`<table class="epochs">
          <thead><tr><th>Epoch</th><th class="r">Loss</th><th class="r">Dev loss</th><th class="r">Took</th><th></th></tr></thead>
          <tbody>${epochs.map(
            (row) => html`<tr key=${row.epoch} class=${row.epoch === kept ? "kept" : ""}>
              <td class="num">${row.epoch}</td><td class="r num">${row.loss}</td><td class="r num">${row.dev_loss}</td>
              <td class="r num">${row.seconds != null ? seconds(row.seconds) : ""}</td>
              <td>${row.epoch === kept ? "kept" : row.stopped ? "stopped early: dev loss rose" : ""}</td></tr>`,
          )}</tbody>
        </table>`}
      </div>
    </section>
  `;
}

// ---------- the log ----------

function planLines(metrics) {
  const lines = [
    `data: ${metrics.examples.toLocaleString("en")} examples: ${metrics.holdout.toLocaleString("en")} holdout, ` +
      `${metrics.fit.toLocaleString("en")} fit, ${metrics.dev.toLocaleString("en")} dev`,
  ];
  for (const [question, counts] of Object.entries(metrics.labels ?? {}))
    lines.push(`labels: ${question}: ${Object.entries(counts).map(([label, n]) => `${label} ${n.toLocaleString("en")}`).join(", ")}`);
  lines.push(`method: ${metrics.kinds.join(", then ")}`);
  let memory = `memory: about ${metrics.estimate.peak_gb} GB at the peak`;
  if (metrics.estimate.disk_gb) memory += `, ${metrics.estimate.disk_gb} GB on disk while it runs`;
  lines.push(memory);
  return lines;
}

function Log({ events, job }) {
  const box = useRef(null);
  const lines = [];
  for (const event of events) {
    if (!event) continue;
    if (event.message) lines.push(`${event.stage}: ${event.message}`);
    if (event.stage === "check" && event.metrics?.estimate) lines.push(...planLines(event.metrics));
  }
  if (job.state === "ended" && job.outcome !== "done" && job.message && !lines.some((line) => line.includes(job.message)))
    lines.push(job.message);
  useEffect(() => {
    if (box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [lines.length]);
  return html`
    <details class="gsec log-sec" open=${job.state !== "ended" || job.outcome !== "done"}>
      <summary class="gsec-head"><h2>Log</h2><span class="tag">as ${job.kind === "eval" ? "training eval" : "training fit"} prints it</span></summary>
      <div class="group"><pre class="fit-log" ref=${box}>${lines.join("\n") || "Starting…"}</pre></div>
    </details>
  `;
}

// ---------- the report ----------

function Change({ before, after, up }) {
  if (before == null || after == null) return html`<span></span>`;
  const delta = (after - before) * 100;
  if (Math.abs(delta) < 0.05) return html`<span class="change same">±0</span>`;
  const better = up ? delta > 0 : delta < 0;
  return html`<span class=${`change ${better ? "better" : "worse"}`}>${delta > 0 ? "+" : "−"}${Math.abs(delta).toFixed(1)} pts
    <span class="sr">${better ? "better" : "worse"}</span></span>`;
}

function QuestionReport({ id, entry, where }) {
  const keys = METRICS.filter(([key]) => entry.before[key] != null || entry.after[key] != null);
  return html`
    <div class="q-report" key=${id}>
      <h3><code>${id}</code> · ${plural(entry.before.n, "example")} ${where}</h3>
      <table class="report-table">
        <thead><tr><th></th><th class="r">Before</th><th class="r">After</th><th class="r">Change</th></tr></thead>
        <tbody>${keys.map(
          ([key, label, up]) => html`<tr key=${key}><th>${label}</th>
            <td class="r num">${key === "mae" ? entry.before[key]?.toFixed(3) : percent(entry.before[key])}</td>
            <td class="r num"><b>${key === "mae" ? entry.after[key]?.toFixed(3) : percent(entry.after[key])}</b></td>
            <td class="r num">${key === "mae" ? "" : html`<${Change} before=${entry.before[key]} after=${entry.after[key]} up=${up} />`}</td></tr>`,
        )}
        ${Boolean(entry.before.refused || entry.after.refused) &&
        html`<tr><th>Refused</th><td class="r num">${entry.before.refused}</td><td class="r num">${entry.after.refused}</td><td></td></tr>`}
        </tbody>
      </table>
    </div>
  `;
}

async function revealFit(id) {
  try {
    await api(`/jobs/${encodeURIComponent(id)}/reveal`, { method: "POST" });
  } catch (error) {
    toast(`Could not show it in Finder: ${error.message}`);
  }
}

function Report({ job }) {
  const [state, setState] = useState(null); // {path, report} or {error}
  useEffect(() => {
    let live = true;
    api(`/jobs/${encodeURIComponent(job.id)}/report`).then(
      (answer) => live && setState(answer),
      (error) => live && setState({ error: error.message }),
    );
    return () => {
      live = false;
    };
  }, [job.id]);
  if (!state) return null;
  if (state.error) return html`<p class="warn-text">${state.error}</p>`;
  const { report, path } = state;
  const name = job.details?.name;
  const listed = (models.value ?? []).some((model) => model.name === name);
  const validation = report.options?.validation;
  const where = validation ? `in ${String(validation).split("/").pop().replace(/\.jsonl$/, "")}` : "kept aside";
  const training = report.training ?? {};
  const command = (job.command ?? []).map(shellWord).join(" ");
  return html`
    <section class="gsec"><div class="gsec-head"><h2>Report</h2><span class="tag">scored ${where}, before and after the fit</span></div>
      <div class="group report-group">
        ${Object.entries(report.questions ?? {}).map(([id, entry]) => html`<${QuestionReport} id=${id} entry=${entry} where=${where} />`)}
        ${training.best_epoch != null &&
        html`<p class="report-note">Kept epoch ${training.best_epoch} of ${training.epochs?.length ?? "?"}${training.stopped_early ? " (stopped early: dev loss rose)" : ""}.</p>`}
        ${(report.warnings ?? []).map((warning) => html`<p class="warn-text">${warning}</p>`)}
      </div>
    </section>
    <section class="gsec"><div class="gsec-head"><h2>The fitted model</h2></div>
      <div class="group"><dl>
        <div class="kv"><dt>Name</dt><dd><code>${name}</code>${report.seconds?.total != null && html`<small>Took ${seconds(report.seconds.total)}</small>`}</dd></div>
        <div class="kv"><dt>Saved in</dt><dd><code class="path" title=${path}>${path}</code>
          <button class="btn sm" onClick=${() => revealFit(job.id)}>Show in Finder</button></dd></div>
        <div class="kv"><dt>Same in a terminal</dt><dd><code class="command">${command}</code></dd></div>
      </dl></div>
    </section>
    <div class="start-bar">
      <a class=${`btn primary${listed ? "" : " off"}`} href=${listed ? href("playground") : undefined}
         aria-disabled=${listed ? undefined : "true"} title=${listed ? `Ask ${name} in the Playground` : `${name} is not in the list yet`}
         onClick=${() => listed && chooseModel(name)}>Open in Playground</a>
      <a class="btn" href=${href("models", name)}>Show in Models</a>
    </div>
  `;
}

// ---------- an evaluation's report ----------

const value = (key, entry) => (entry?.[key] == null ? "–" : key === "mae" ? entry[key].toFixed(3) : percent(entry[key]));

function Confusion({ question, name }) {
  const labels = question.labels ?? Object.keys(question.confusion ?? {});
  return html`
    <div class="confusion">
      <h4>${name}</h4>
      <table class="confusion-table" aria-label=${`${name}: gold label by row, predicted by column`}>
        <thead><tr><th>gold ↓ predicted →</th>${labels.map((label) => html`<th class="r"><code>${label}</code></th>`)}</tr></thead>
        <tbody>${labels.map(
          (gold) => html`<tr key=${gold}><th><code>${gold}</code></th>${labels.map(
            (predicted) => html`<td class=${`r num${gold === predicted ? " hit" : ""}`}>${question.confusion?.[gold]?.[predicted] ?? 0}</td>`,
          )}</tr>`,
        )}</tbody>
      </table>
    </div>
  `;
}

function EvalReport({ job }) {
  const [state, setState] = useState(null);
  useEffect(() => {
    let live = true;
    api(`/jobs/${encodeURIComponent(job.id)}/report`).then(
      (answer) => live && setState(answer),
      (error) => live && setState({ error: error.message }),
    );
    return () => {
      live = false;
    };
  }, [job.id]);
  if (!state) return null;
  if (state.error) return html`<p class="warn-text">${state.error}</p>`;
  const { data, models: scored } = state.result;
  const [first, second] = scored;
  const ids = [...new Set(scored.flatMap((model) => Object.keys(model.questions)))];
  const command = (job.command ?? []).map(shellWord).join(" ");
  return html`
    <section class="gsec"><div class="gsec-head"><h2>Report</h2><span class="tag">${plural(data.examples, "example")} in ${job.details?.data}</span></div>
      <div class="group report-group">
        ${ids.map((id) => {
          const a = first.questions[id];
          const b = second?.questions[id];
          const keys = METRICS.filter(([key]) => a?.[key] != null || b?.[key] != null);
          return html`<div class="q-report" key=${id}>
            <h3><code>${id}</code> · ${plural((a ?? b).n, "example")} labelled</h3>
            <table class="report-table eval-table">
              <thead><tr><th></th><th class="r"><code>${first.name}</code></th>${second && html`<th class="r"><code>${second.name}</code></th><th class="r">Difference</th>`}</tr></thead>
              <tbody>
                ${keys.map(
                  ([key, label, up]) => html`<tr key=${key}><th>${label}</th>
                    <td class="r num"><b>${value(key, a)}</b></td>
                    ${second && html`<td class="r num">${value(key, b)}</td>
                      <td class="r num">${key === "mae" ? "" : html`<${Change} before=${b?.[key]} after=${a?.[key]} up=${up} />`}</td>`}
                  </tr>`,
                )}
                ${Boolean(a?.refused || b?.refused) &&
                html`<tr><th>Refused</th><td class="r num">${a?.refused ?? 0}</td>${second && html`<td class="r num">${b?.refused ?? 0}</td><td></td>`}</tr>`}
              </tbody>
            </table>
            <details class="confusions"><summary>Confusion matrices</summary>
              <div class="confusion-pair">
                ${a && html`<${Confusion} question=${a} name=${first.name} />`}
                ${b && html`<${Confusion} question=${b} name=${second.name} />`}
              </div>
            </details>
          </div>`;
        })}
        ${second && html`<p class="report-note">Difference: ${first.name} against ${second.name}, in points; green where ${first.name} does better.</p>`}
      </div>
    </section>
    <section class="gsec"><div class="gsec-head"><h2>Same in a terminal</h2></div>
      <div class="group"><dl>
        <div class="kv"><dt>Command</dt><dd><code class="command">${command}</code></dd></div>
        <div class="kv"><dt>Took</dt><dd>${scored.map((model) => `${model.name} ${seconds(model.seconds)}`).join(" · ")}</dd></div>
      </dl></div>
    </section>
  `;
}

// ---------- the view ----------

const OUTCOMES = {
  cancelled: "Stopped: nothing was written",
  error: "Failed",
  failed: "Failed",
  interrupted: "Interrupted: the Studio stopped while it ran",
};

export function FitView({ id }) {
  useEffect(() => {
    openFit(id);
  }, [id]);
  const job = fitById(id);
  if (!job) return html`<div class="empty"><b>Nothing named ${id}</b><span>It may be from another workspace. Pick a fit or an evaluation on the left.</span></div>`;
  const evaluation = job.kind === "eval";
  const current = opened.value.id === id ? opened.value : { events: [] };
  const events = current.events.filter(Boolean);
  const details = job.details ?? {};
  const running = job.state !== "ended";
  const done = job.outcome === "done";
  const sub = evaluation
    ? [details.compare ? `${details.model} against ${details.compare}` : details.model, details.data && `on ${details.data}`].filter(Boolean).join(" · ")
    : [details.model, details.data && `on ${details.data}`, details.report && `reported on ${details.report}`, details.method]
        .filter(Boolean)
        .join(" · ");
  return html`
    <div class="pane-head">
      <div><h1>${job.title}</h1><p>${sub}</p></div>
      ${running && job.state === "waiting" &&
      html`<div class="actions"><button class="btn" disabled=${job.stop_requested} onClick=${() => stopJob(job.id)}>${job.stop_requested ? "Stopping…" : "Stop"}</button></div>`}
    </div>
    <div class="group fit-progress">
      ${evaluation ? html`<${EvalStepper} events=${events} job=${job} />` : html`<${Stepper} events=${events} job=${job} />`}
      ${running && html`<${StepLine} events=${events} />`}
      ${!running && !done && html`<p class="run-outcome warn-text">${OUTCOMES[job.outcome] ?? job.outcome}${job.message && job.outcome !== "cancelled" ? `: ${job.message.trim().split("\n").pop()}` : ""}</p>`}
    </div>
    ${(details.warnings ?? []).map((warning) => html`<p class="warn-text run-warning">${warning}</p>`)}
    ${done && (evaluation ? html`<${EvalReport} job=${job} />` : html`<${Report} job=${job} />`)}
    ${!evaluation && html`<${LossChart} events=${events} done=${done} />`}
    <${Log} events=${events} job=${job} />
  `;
}
