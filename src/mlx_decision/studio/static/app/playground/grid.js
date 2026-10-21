// Run All's grid: a row per example, a column per question, each cell the
// verdict and its value. Used by the Answers tab and the History. On a
// dataset, each cell also holds the row's expected answer, and each question
// how many answers agree with it (a baseline before any fit).
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

// Whether an answer is the expected one: a choice's option, yes / no, a level
// from 1 (a score's expected level, rounded).
function agrees(answer, expected) {
  if (answer.type === "choice") return answer.choice === expected;
  if (answer.type === "score") return Math.round(answer.score) + 1 === Number(expected);
  return (answer.noul >= 0.5 ? "yes" : "no") === expected;
}

// Per question: {agree, of} over the rows answered that expect something.
function agreement(rows, questions, expected) {
  const counts = Object.fromEntries(questions.map((id) => [id, { agree: 0, of: 0 }]));
  for (const row of rows) {
    if (!row.run || row.stale) continue;
    for (const id of questions) {
      const want = expected?.[row.name]?.[id];
      const answer = row.run.response?.answers?.[id];
      if (want == null || !answer) continue;
      counts[id].of += 1;
      if (agrees(answer, want)) counts[id].agree += 1;
    }
  }
  return counts;
}

// `rows`: [{name, run?, error?, pending?, stale?}]; `questions`: the ids to
// show; `selected`: the example whose row is marked; `onPick(name)`;
// `expected`: {row name: {question id: answer}} on a dataset, else null.
export function Grid({ rows, questions, selected = null, onPick, expected = null }) {
  const counts = expected ? agreement(rows, questions, expected) : null;
  return html`
    <div class="grid-wrap">
      <table class="ex-grid">
        <thead>
          <tr><th>Example</th>${questions.map((id) => {
            const count = counts?.[id];
            return html`<th key=${id}>${id}${count?.of ? html`<small class="agree num" title="Answers that are the rows' expected ones">${count.agree} of ${count.of} agree · ${Math.round((count.agree / count.of) * 100)}%</small>` : ""}</th>`;
          })}</tr>
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
                      const answer = row.run?.response?.answers?.[id];
                      const shown = cell(answer);
                      const want = expected?.[row.name]?.[id];
                      const mark = want != null && answer && !row.stale ? (agrees(answer, want) ? "agree" : "disagree") : "";
                      return html`<td key=${id} class=${[row.stale ? "stale" : "", mark].filter(Boolean).join(" ")}>${
                        shown ? html`<span class="v">${shown.verdict}</span><span class="p num">${shown.value}</span>` : "—"
                      }${want != null && html`<span class="x">expects ${want}</span>`}</td>`;
                    })}
              </tr>
            `,
          )}
        </tbody>
      </table>
    </div>
  `;
}
