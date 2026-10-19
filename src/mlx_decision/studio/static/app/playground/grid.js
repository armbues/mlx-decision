// Run All's grid: a row per example, a column per question, each cell the
// verdict and its value. Used by the Answers tab and the History.
import { html } from "htm/preact";
import { two } from "../charts/answers.js";

// A short verdict and its value: "technical .80", "2.4 of 3" (levels from 1),
// "yes .91".
function cell(answer) {
  if (!answer) return null;
  if (answer.type === "choice")
    return { verdict: answer.choice, value: two(answer.probabilities?.[answer.choice] ?? 0) };
  if (answer.type === "score") {
    const levels = Math.max(2, Object.keys(answer.probabilities ?? {}).length);
    return { verdict: two(answer.score + 1), value: `of ${levels}` };
  }
  return { verdict: answer.noul >= 0.5 ? "yes" : "no", value: two(answer.noul) };
}

// `rows`: [{name, run?, error?, pending?, stale?}]; `questions`: the ids to
// show; `selected`: the example whose row is marked; `onPick(name)`.
export function Grid({ rows, questions, selected = null, onPick }) {
  return html`
    <div class="grid-wrap">
      <table class="ex-grid">
        <thead>
          <tr><th>Example</th>${questions.map((id) => html`<th key=${id}>${id}</th>`)}</tr>
        </thead>
        <tbody>
          ${rows.map(
            (row) => html`
              <tr key=${row.name} class=${row.name === selected ? "sel" : ""} onClick=${() => onPick?.(row.name)}
                  title=${row.stale ? "This example changed since; Run All to update" : `Show ${row.name}'s answers`}>
                <th class=${row.stale ? "stale" : ""}>${row.name}</th>
                ${row.error
                  ? html`<td class="err" colspan=${questions.length}>${row.error}</td>`
                  : questions.map((id) => {
                      if (row.pending) return html`<td class="wait" key=${id}>…</td>`;
                      const shown = cell(row.run?.response?.answers?.[id]);
                      return html`<td key=${id} class=${row.stale ? "stale" : ""}>${
                        shown ? html`<span class="v">${shown.verdict}</span><span class="p num">${shown.value}</span>` : "—"
                      }</td>`;
                    })}
              </tr>
            `,
          )}
        </tbody>
      </table>
    </div>
  `;
}
