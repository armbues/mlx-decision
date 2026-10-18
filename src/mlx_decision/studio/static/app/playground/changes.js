// What changed between two runs' requests, and the numbers that show how
// an answer moved.

const text = (value) => (value == null ? "" : typeof value === "string" ? value : JSON.stringify(value));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// Longest-common-subsequence word diff: [{kind: "same"|"del"|"ins", text}],
// whitespace kept with the words. Very long texts fall back to one
// deletion and one insertion.
export function wordDiff(before, after) {
  const x = text(before).split(/(\s+)/).filter((part) => part !== "");
  const y = text(after).split(/(\s+)/).filter((part) => part !== "");
  if (x.length * y.length > 2_000_000)
    return [
      { kind: "del", text: text(before) },
      { kind: "ins", text: text(after) },
    ];
  const m = x.length;
  const n = y.length;
  const longest = Array.from({ length: m + 1 }, () => new Uint32Array(n + 1));
  for (let i = m - 1; i >= 0; i -= 1)
    for (let j = n - 1; j >= 0; j -= 1)
      longest[i][j] = x[i] === y[j] ? longest[i + 1][j + 1] + 1 : Math.max(longest[i + 1][j], longest[i][j + 1]);
  const parts = [];
  const push = (kind, part) => {
    const last = parts.at(-1);
    if (last && last.kind === kind) last.text += part;
    else parts.push({ kind, text: part });
  };
  let i = 0;
  let j = 0;
  while (i < m && j < n) {
    if (x[i] === y[j]) {
      push("same", x[i]);
      i += 1;
      j += 1;
    } else if (longest[i + 1][j] >= longest[i][j + 1]) push("del", x[i++]);
    else push("ins", y[j++]);
  }
  while (i < m) push("del", x[i++]);
  while (j < n) push("ins", y[j++]);
  return parts;
}

// The changes from request `before` to request `after`, each {where, before,
// after} (before or after missing for something added or removed), or
// {where, note} for what is only named as changed.
export function requestChanges(before, after) {
  const changes = [];
  if (!same(before.state, after.state))
    changes.push({ where: "state", before: before.state, after: after.state });
  if (!same(before.images ?? [], after.images ?? []))
    changes.push({ where: "images", note: `${(after.images ?? []).length} images, changed` });
  const old = before.questions ?? {};
  const now = after.questions ?? {};
  for (const [id, question] of Object.entries(now)) {
    const previous = old[id];
    if (!previous) {
      changes.push({ where: id, note: `added, ${question.type}` });
      continue;
    }
    if (previous.type !== question.type) {
      changes.push({ where: `${id}.type`, before: previous.type, after: question.type });
      continue;
    }
    if (!same(previous.instructions, question.instructions))
      changes.push({ where: `${id}.instructions`, before: previous.instructions, after: question.instructions });
    criteriaChanges(id, question.type, previous.criteria, question.criteria, changes);
  }
  for (const id of Object.keys(old)) if (!(id in now)) changes.push({ where: id, note: "removed" });
  return changes;
}

function criteriaChanges(id, type, before, after, changes) {
  if (same(before, after)) return;
  const isMap = (value) => value && typeof value === "object" && !Array.isArray(value);
  if (type === "score" && Array.isArray(before) && Array.isArray(after)) {
    for (let level = 0; level < Math.max(before.length, after.length); level += 1) {
      if (level >= after.length) changes.push({ where: `${id}.level ${level + 1}`, note: "removed" });
      else if (level >= before.length) changes.push({ where: `${id}.level ${level + 1}`, after: after[level] });
      else if (!same(before[level], after[level]))
        changes.push({ where: `${id}.level ${level + 1}`, before: before[level], after: after[level] });
    }
    return;
  }
  if (isMap(before) && isMap(after)) {
    for (const [key, value] of Object.entries(after)) {
      if (!(key in before)) changes.push({ where: `${id}.${key}`, note: `added${value ? `: ${text(value)}` : ""}` });
      else if (!same(before[key], value)) changes.push({ where: `${id}.${key}`, before: before[key], after: value });
    }
    for (const key of Object.keys(before)) if (!(key in after)) changes.push({ where: `${id}.${key}`, note: "removed" });
    const order = (map) => Object.keys(map).filter((key) => key in before && key in after).join();
    if (order(before) !== order(after)) changes.push({ where: `${id}.criteria`, note: "reordered" });
    return;
  }
  changes.push({ where: `${id}.criteria`, before, after });
}

// The question ids a change list touches.
export const touched = (changes) =>
  new Set(changes.map((change) => change.where.split(".")[0]).filter((id) => id !== "state" && id !== "images"));

// One line for the runs list.
export function summary(changes, isFirst) {
  if (isFirst) return "First run";
  if (changes.length === 0) return "Same request";
  const ids = [...touched(changes)];
  const parts = [];
  if (changes.some((change) => change.where === "state")) parts.push("state");
  if (changes.some((change) => change.where === "images")) parts.push("images");
  if (ids.length) parts.push(ids.join(", "));
  return `Changed ${parts.join(", ")}`;
}

// The number that shows how an answer moved, on the scale of `reference`
// (an answer to the same question): a choice's probability of the option
// `reference` chose, a score's expected level, a noul's probability of yes.
export function answerValue(answer, reference = answer) {
  if (!answer || !reference || answer.type !== reference.type) return null;
  if (answer.type === "choice") return answer.probabilities?.[reference.choice] ?? null;
  if (answer.type === "score") return answer.score;
  if (answer.type === "noul") return answer.noul;
  return null;
}

// The top of the scale: 1, or a score's highest level (0-based, as answerValue).
export const answerMax = (answer) =>
  answer?.type === "score" ? Math.max(1, Object.keys(answer.probabilities ?? {}).length - 1) : 1;

// What the value means, for a label: "technical, p", "score, 1 to 3", "p(yes)".
export function valueLabel(answer) {
  if (answer.type === "choice") return `${answer.choice}, p`;
  if (answer.type === "score") return `score, 1 to ${answerMax(answer) + 1}`;
  return "p(yes)";
}

// An answer's value as the page shows it: a score's level counted from 1.
export const shownValue = (answer, value) => (value == null ? null : answer?.type === "score" ? value + 1 : value);
