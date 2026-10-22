// A dataset's questions with their labels and counts: the Datasets page's
// Questions (a label picked filters the examples) and the Training page's
// inspector, for the dataset the fit form has picked.
import { html } from "htm/preact";
import { useEffect } from "preact/hooks";
import { NOT_THIS_MAC, fileSize, onThisMac, plural, shellWord } from "../format.js";
import { described, describe, modifiedOf, summaryKey } from "../store/data.js";
import { revealData } from "./files.js";

// The summary of ``name``'s ``split`` (null: its default split), read again
// when the dataset changes; null while it reads.
export function useSummary(name, split = null) {
  const modified = modifiedOf(name);
  useEffect(() => {
    if (name) describe(name, split);
  }, [name, split, modified]);
  return name ? (described.value[summaryKey(name, split)] ?? null) : null;
}

// A label column's value with its name when the dataset knows one: "3 · Dividend".
export function valueWords(summary, column, value) {
  const name = summary?.columns?.[column]?.values?.[value]?.name;
  return name ? `${value} · ${name}` : value;
}

// The values of each label column with their counts in this split, as Labels
// shows a question's labels (a value picked filters the examples).
export function Values({ summary, column, onPick = null, active = null }) {
  const values = summary.values?.[column] ?? [];
  const most = Math.max(1, ...values.map(([, count]) => count));
  return html`
    <ul class=${`label-counts${onPick ? " can-pick" : ""}`}>
      ${values.map(([value, count]) => {
        const inner = html`<code title=${valueWords(summary, column, value)}>${valueWords(summary, column, value)}</code>
          <span class="bar"><i style=${`width:${((count / most) * 100).toFixed(1)}%`}></i></span>
          <span class="num">${count.toLocaleString("en")}</span>`;
        const classes = [count ? "" : "none", value === active ? "on" : ""].filter(Boolean).join(" ");
        return html`<li key=${value} class=${classes}>${onPick
          ? html`<button aria-pressed=${value === active ? "true" : "false"} disabled=${!count}
              title=${count ? `Show the rows whose ${column} is ${value}` : "No row has this value"}
              onClick=${() => onPick(value)}>${inner}</button>`
          : inner}</li>`;
      })}
    </ul>
  `;
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
            title=${count ? `Show the rows labelled ${label}` : "No row has this label"}
            onClick=${() => onPick(label)}>${inner}</button>`
        : inner}
    </li>`;
  };
  return html`
    <ul class=${`label-counts${onPick ? " can-pick" : ""}`}>${options.map(row)}</ul>
    <small>${question.labelled.toLocaleString("en")} of ${plural(total, "row")} labelled</small>
  `;
}

// The inspector's summary of a dataset's split: its questions with their labels, and the file.
export function DatasetSummary({ name, split = null }) {
  const summary = useSummary(name, split);
  if (!name) return null;
  if (!summary) return null;
  if (summary.error) return html`<p class="warn-text">${summary.error}</p>`;
  return html`
    <div class="idet data-inspector">
      <div class="idet-head"><div><h2 class="mono">${name}</h2><p>${summary.kind === "folder" ? `${summary.split} split · ` : ""}${plural(summary.examples, "row")} · ${plural(summary.questions.length, "question")}</p></div></div>
      ${summary.missing?.length > 0 && html`<p class="warn-text">Not ready for a fit: ${summary.missing.join("; ")}</p>`}
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

// A split's labels on one line, per question: "topic: analyst 20 · fed 20 · other 20".
export function LabelLine({ name, split = null }) {
  const summary = useSummary(name, split);
  if (!summary || summary.error) return null;
  const asked = summary.questions.filter((question) => question.labelled > 0);
  if (!asked.length) return html`<span class="hint">No labels in this split</span>`;
  return html`<span class="hint label-line">${asked.map((question, index) => {
    const shown = question.labels.slice(0, 6).map(([label, count]) => `${label} ${count.toLocaleString("en")}`);
    const more = question.labels.length > 6 ? ` · ${question.labels.length - 6} more` : "";
    return html`${index ? html`<br />` : ""}${asked.length > 1 ? html`<code>${question.id}</code>: ` : ""}${shown.join(" · ")}${more}`;
  })}</span>`;
}

// The path for a command line: with ~ when the shell needs no quotes for it.
const pathWord = (summary) => (/^[A-Za-z0-9_@%+=:,./~-]+$/.test(summary.display) ? summary.display : shellWord(summary.path));

// The dataset's folder (or file): its path (with Show in Finder), the split's
// file and size, and the commands that read it.
export function FileGroup({ summary }) {
  const path = pathWord(summary);
  const folder = summary.kind === "folder";
  return html`
    <section class="igroup file-group"><h4>${folder ? "Folder" : "File"}</h4><dl>
      <div class="kv"><dt>Path</dt><dd><code title=${summary.path}>${summary.display}${folder ? "/" : ""}</code>
        <button class="btn sm" disabled=${!onThisMac} title=${onThisMac ? null : NOT_THIS_MAC} onClick=${() => revealData(summary.name)}>Show in Finder</button></dd></div>
      ${folder && html`<div class="kv"><dt>Split</dt><dd><code>${summary.split}.jsonl</code></dd></div>`}
      <div class="kv"><dt>Size</dt><dd>${fileSize(summary.bytes)}</dd></div>
      <div class="kv"><dt>Used by</dt><dd>
        <code>mlx-decision training fit -m MODEL --data ${path}</code>
        <code>mlx-decision training eval -m FITTED --data ${path}</code>
      </dd></div>
    </dl></section>
  `;
}
