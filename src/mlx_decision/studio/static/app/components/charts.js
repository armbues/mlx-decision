// Charts shared by the fit report, Evaluate and batches: the confusion matrix
// (a table) and the reliability diagram (an SVG), plus the axis ticks the
// line charts use.

import { html } from "htm/preact";
import { useState } from "preact/hooks";

// Round tick values from low to high, about five of them.
export function ticks(low, high) {
  const span = high - low || 1;
  const step = 10 ** Math.floor(Math.log10(span / 3));
  const nice = [1, 2, 5, 10].map((f) => f * step).find((s) => span / s <= 5) ?? step * 10;
  const first = Math.ceil(low / nice) * nice;
  const result = [];
  for (let value = first; value <= high + 1e-9; value += nice) result.push(Number(value.toFixed(6)));
  return result;
}

// Gold labels by row, predicted by column, from a report's ``confusion``
// ({gold: {predicted: count}}) and ``labels`` (the order). Columns the
// table has beyond the labels (``refused``) come last. ``names`` shows a
// label under another name; ``onCell(gold, predicted)`` makes the cells
// buttons (a batch filters its rows by them).
export function Confusion({ question, name, names = {}, onCell = null }) {
  const labels = question.labels ?? Object.keys(question.confusion ?? {});
  const extra = Object.keys(Object.values(question.confusion ?? {})[0] ?? {}).filter((key) => !labels.includes(key));
  const columns = [...labels, ...extra];
  const shown = (label) => names[label] ?? label;
  const cell = (gold, predicted) => {
    const count = question.confusion?.[gold]?.[predicted] ?? 0;
    const hit = gold === predicted;
    const label = `${count} with ${shown(gold)} answered ${shown(predicted)}`;
    return html`<td class=${`r num${hit ? " hit" : ""}${count ? "" : " zero"}`}>${
      onCell && count
        ? html`<button class="cm-cell" title=${label} aria-label=${label} onClick=${() => onCell(gold, predicted)}>${count}</button>`
        : count
    }</td>`;
  };
  return html`
    <div class="confusion">
      ${name && html`<h4>${name}</h4>`}
      <table class="confusion-table" aria-label=${`${name ? `${name}: ` : ""}gold label by row, predicted by column`}>
        <thead><tr><th>gold ↓ predicted →</th>${columns.map((label) => html`<th class="r"><code>${shown(label)}</code></th>`)}</tr></thead>
        <tbody>${labels.map(
          (gold) => html`<tr key=${gold}><th><code>${shown(gold)}</code></th>${columns.map((predicted) => cell(gold, predicted))}</tr>`,
        )}</tbody>
      </table>
    </div>
  `;
}

const DIAGRAM = { size: 220, left: 34, right: 10, top: 10, bottom: 30 };

// Accuracy against confidence in equal bins (``bins``: [{low, high, n,
// accuracy, confidence}]): a bar per bin with answers, the diagonal of a
// model whose confidence is its accuracy, and each bin's count on hover.
export function Reliability({ bins, name = "" }) {
  const [hover, setHover] = useState(null);
  const { size, left, right, top, bottom } = DIAGRAM;
  const inner = size - left - right;
  const tall = size - top - bottom;
  const x = (value) => left + value * inner;
  const y = (value) => top + (1 - value) * tall;
  const used = bins.filter((bin) => bin.n > 0);
  const total = used.reduce((sum, bin) => sum + bin.n, 0);
  const picked = hover != null ? bins[hover] : null;
  return html`
    <div class="reliability">
      ${name && html`<h4>${name}</h4>`}
      <svg viewBox=${`0 0 ${size} ${size}`} role="img"
           aria-label=${`Reliability${name ? ` of ${name}` : ""}: accuracy by confidence over ${total} answers`}>
        ${[0, 0.5, 1].map(
          (value) => html`<g key=${value}>
            <line class="grid" x1=${left} x2=${size - right} y1=${y(value)} y2=${y(value)} />
            <text class="tick" x=${left - 5} y=${y(value) + 3.5} text-anchor="end">${value * 100}%</text>
            <text class="tick" x=${x(value)} y=${size - bottom + 13} text-anchor="middle">${value * 100}%</text>
          </g>`,
        )}
        <text class="tick axis" x=${left + inner / 2} y=${size - 4} text-anchor="middle">confidence</text>
        <line class="ideal" x1=${x(0)} y1=${y(0)} x2=${x(1)} y2=${y(1)} />
        ${bins.map((bin, index) =>
          bin.n > 0
            ? html`<rect key=${index} class=${`rel-bar${hover === index ? " on" : ""}`}
                x=${x(bin.low) + 1} width=${Math.max(1, x(bin.high) - x(bin.low) - 2)}
                y=${y(bin.accuracy)} height=${Math.max(0.5, y(0) - y(bin.accuracy))}
                onPointerEnter=${() => setHover(index)} onPointerLeave=${() => setHover(null)} />`
            : null,
        )}
      </svg>
      <div class="rel-note">${
        picked
          ? html`<b class="num">${Math.round(picked.low * 100)}–${Math.round(picked.high * 100)}%</b> ${picked.n} answered, ${Math.round(picked.accuracy * 100)}% right (mean confidence ${Math.round(picked.confidence * 100)}%)`
          : html`Bars at the diagonal: as right as sure. Below it: too sure.`
      }</div>
    </div>
  `;
}
