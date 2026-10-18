// The answer charts: probability bars (choice), columns per level drawn to
// scale with the expected level marked (score), one yes bar (noul).
import { html } from "htm/preact";

export const two = (value) => value.toFixed(2);
export const percent = (value) => `${Math.round(value * 100)}%`;

export function ChoiceBars({ answer }) {
  const entries = Object.entries(answer.probabilities);
  return html`
    <div class="verdict">
      <span class="big">${answer.choice}</span>
      <span class="small num">${percent(answer.probabilities[answer.choice] ?? answer.confidence)}</span>
    </div>
    <div class="pbars">
      ${entries.map(
        ([key, p]) => html`
          <div class=${`pbar ${key === answer.choice ? "top" : ""}`} key=${key}>
            <span class="k" title=${key}>${key}</span>
            <span class="ptrack"><i style=${`width:${(p * 100).toFixed(1)}%`}></i></span>
            <span class="v num">${two(p)}</span>
          </div>
        `,
      )}
    </div>
  `;
}

export function ScoreColumns({ answer, width = 340 }) {
  const levels = Object.values(answer.probabilities);
  const count = levels.length;
  const [padLeft, padRight, height, base, maxHeight] = [10, 10, 122, 78, 60];
  const step = (width - padLeft - padRight) / count;
  const barWidth = Math.min(56, step * 0.56);
  const x = (level) => padLeft + step * (level + 0.5);
  const top = levels.indexOf(Math.max(...levels));
  const expected = count > 1 ? x(0) + (x(count - 1) - x(0)) * (answer.score / (count - 1)) : x(0);
  // Levels are shown from 1, as people rate; the API's value counts from 0.
  const shown = two(answer.score + 1);
  return html`
    <div class="verdict">
      <span class="big num">${shown}</span>
      <span class="small">expected level, 1 to ${count}</span>
      <span class="small api-value num" title="The score field in the API's answer counts levels from 0">API value ${two(answer.score)} (0 to ${count - 1})</span>
    </div>
    <svg class="score-svg" viewBox=${`0 0 ${width} ${height}`} role="img" aria-label=${`Expected level ${shown} of ${count}`}>
      ${levels.map((p, level) => {
        const h = p * maxHeight;
        const fill = level === top ? "var(--accent)" : "var(--fg-3)";
        return html`
          <rect x=${x(level) - barWidth / 2} y=${base - h} width=${barWidth} height=${h} rx="3" fill=${fill}
                opacity=${level === top ? 1 : 0.55} />
          <text x=${x(level)} y=${base - h - 5} text-anchor="middle" class="val">${two(p)}</text>
          <line x1=${x(level)} x2=${x(level)} y1=${base + 6} y2=${base + 12} stroke="var(--fg-3)" />
          <text x=${x(level)} y=${base + 23} text-anchor="middle">${level + 1}</text>
        `;
      })}
      <line x1=${x(0)} x2=${x(count - 1)} y1=${base + 9} y2=${base + 9} stroke="var(--sep-strong)" />
      <line x1=${expected} x2=${expected} y1=${base + 9} y2=${base + 30} stroke="var(--fg)" stroke-width="1.2" />
      <path d=${`M${expected - 5} ${base + 2} L${expected + 5} ${base + 2} L${expected} ${base + 9} Z`} fill="var(--fg)" />
      <rect x=${expected - 50} y=${base + 29} width="100" height="17" rx="8.5" fill="var(--fg)" />
      <text x=${expected} y=${base + 41.5} text-anchor="middle" class="val" style="fill: var(--group)">
        expected ${shown}
      </text>
    </svg>
    <div class="score-legend">
      ${Object.entries(answer.legend ?? {}).map(
        ([level, text]) => html`<div key=${level}><b class="num">${Number(level) + 1}</b> ${
          typeof text === "string" ? text : JSON.stringify(text)
        }</div>`,
      )}
    </div>
  `;
}

export function NoulBar({ answer }) {
  const yes = answer.noul;
  return html`
    <div class="verdict">
      <span class="big">${yes >= 0.5 ? "Yes" : "No"}</span>
      <span class="small num">${two(yes)} probability of yes</span>
    </div>
    <div class="noul">
      <div class="noul-track"><span class="yes" style=${`width:${(yes * 100).toFixed(1)}%`}></span></div>
      <div class="noul-labels num">
        <span>Yes <b>${percent(yes)}</b></span><span>No <b>${percent(1 - yes)}</b></span>
      </div>
    </div>
  `;
}
