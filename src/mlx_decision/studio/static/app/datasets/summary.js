// A dataset's questions with their labels and counts: the Datasets page's
// Questions (a label picked filters the examples) and the Training page's
// inspector, for the dataset the fit form has picked.
import { html } from "htm/preact";
import { useEffect } from "preact/hooks";
import { fileSize, plural, shellWord } from "../format.js";
import { described, describe, modifiedOf } from "../store/data.js";
import { revealData } from "./files.js";

// The summary of ``name``, read again when the file changes; null while it reads.
export function useSummary(name) {
  const modified = modifiedOf(name);
  useEffect(() => {
    if (name) describe(name);
  }, [name, modified]);
  return name ? (described.value[name] ?? null) : null;
}

// The question's labels in its own option order, each with its count and a bar
// (the most common label fills it). With ``onPick`` each label is a button
// that picks it; ``active`` is the label picked.
export function Labels({ question, total, onPick = null, active = null }) {
  const most = Math.max(1, ...question.labels.map(([, count]) => count));
  const counts = new Map(question.labels);
  const options = question.options ?? question.labels.map(([label]) => label);
  const row = (label) => {
    const count = counts.get(label) ?? 0;
    const inner = html`<code>${label}</code>
      <span class="bar"><i style=${`width:${((count / most) * 100).toFixed(1)}%`}></i></span>
      <span class="num">${count.toLocaleString("en")}</span>`;
    const classes = [count ? "" : "none", label === active ? "on" : ""].filter(Boolean).join(" ");
    return html`<li key=${label} class=${classes}>
      ${onPick
        ? html`<button aria-pressed=${label === active ? "true" : "false"} disabled=${!count}
            title=${count ? `Show the examples labelled ${label}` : "No example has this label"}
            onClick=${() => onPick(label)}>${inner}</button>`
        : inner}
    </li>`;
  };
  return html`
    <ul class=${`label-counts${onPick ? " can-pick" : ""}`}>${options.map(row)}</ul>
    <small>${question.labelled.toLocaleString("en")} of ${plural(total, "example")} labelled</small>
  `;
}

// The inspector's summary of a dataset: its questions with their labels, and the file.
export function DatasetSummary({ name }) {
  const summary = useSummary(name);
  if (!name) return null;
  if (!summary) return null;
  if (summary.error) return html`<p class="warn-text">${summary.error}</p>`;
  return html`
    <div class="idet data-inspector">
      <div class="idet-head"><div><h2 class="mono">${name}</h2><p>${plural(summary.examples, "example")} · ${plural(summary.questions.length, "question")}</p></div></div>
      ${summary.questions.map(
        (question) => html`<section class="igroup" key=${question.id}><h4>${question.id} · ${question.type}</h4><dl>
          ${question.instructions && html`<div class="kv"><dt>Asks</dt><dd>${question.instructions}</dd></div>`}
          <div class="kv"><dt>Labels</dt><dd><${Labels} question=${question} total=${summary.examples} /></dd></div>
        </dl></section>`,
      )}
      <${FileGroup} summary=${summary} />
    </div>
  `;
}

// The path for a command line: with ~ when the shell needs no quotes for it.
const pathWord = (summary) => (/^[A-Za-z0-9_@%+=:,./~-]+$/.test(summary.display) ? summary.display : shellWord(summary.path));

// The file: its path (with Show in Finder), size and the commands that read it.
export function FileGroup({ summary }) {
  const path = pathWord(summary);
  return html`
    <section class="igroup file-group"><h4>File</h4><dl>
      <div class="kv"><dt>Path</dt><dd><code title=${summary.path}>${summary.display}</code>
        <button class="btn sm" onClick=${() => revealData(summary.name)}>Show in Finder</button></dd></div>
      <div class="kv"><dt>Size</dt><dd>${fileSize(summary.bytes)}</dd></div>
      <div class="kv"><dt>Used by</dt><dd>
        <code>mlx-decision training fit -m MODEL --data ${path}</code>
        <code>mlx-decision training eval -m FITTED --data ${path}</code>
      </dd></div>
    </dl></section>
  `;
}
