// A question set as the editor holds it, and back to a request body.
//
// Every text the editor shows is a field {text, original}: original is the
// value the set had, kept while the text still shows it, so values the editor
// cannot show as typed text (null, numbers, objects) come back unchanged.
// Questions, options and levels carry a uid, so rows keep their identity
// while ids and keys are being edited.

let lastUid = 0;
const uid = () => `u${++lastUid}`;

export const TYPES = ["choice", "score", "noul"];
export const TYPE_LABELS = { choice: "Choice", score: "Score", noul: "Noul" };
export const TYPE_DESCRIPTIONS = {
  choice: "Pick one option from a set you describe",
  score: "Place it on ordered levels, low to high",
  noul: "Yes or no, answered as a probability",
};

const show = (value) =>
  value == null ? "" : typeof value === "string" ? value : JSON.stringify(value);

export const field = (value) => ({ text: show(value), original: value });
export const typed = (text) => ({ text, original: undefined });

// The field's value for the request: the original while unchanged, else the
// text, an empty text giving `empty`.
export function fieldValue(item, empty = null) {
  if (item.original !== undefined && item.text === show(item.original)) return item.original;
  return item.text === "" ? empty : item.text;
}

const option = (key, description) => ({ uid: uid(), key, description });
const level = (description) => ({ uid: uid(), description });

function questionFrom(id, value) {
  const { type, instructions, criteria, ...rest } = value && typeof value === "object" ? value : {};
  const question = {
    uid: uid(),
    id,
    type: TYPES.includes(type) ? type : "choice",
    instructions: field(instructions),
    options: [],
    levels: [],
    yes: field(null),
    no: field(null),
    rest,
  };
  if (question.type === "choice" && criteria && typeof criteria === "object" && !Array.isArray(criteria))
    question.options = Object.entries(criteria).map(([key, description]) => option(key, field(description)));
  else if (question.type === "choice" && Array.isArray(criteria))
    question.options = criteria.map((key) => option(show(key), field(null)));
  if (question.type === "score" && Array.isArray(criteria))
    question.levels = criteria.map((description) => level(field(description)));
  if (question.type === "noul" && criteria && typeof criteria === "object") {
    question.yes = field(criteria.true ?? null);
    question.no = field(criteria.false ?? null);
  }
  return question;
}

// ---------- examples ----------
//
// A set holds named examples, each a state with its images. The file keeps
// them in `examples`, the first also as the top-level state and images (so
// `run -q` and the API read the file as before); a plain request body is one
// example named "Example".

export const FIRST_EXAMPLE = "Example";

function exampleFrom(name, state, images) {
  const json = state !== undefined && typeof state !== "string";
  return {
    uid: uid(),
    name,
    stateMode: json ? "json" : "text",
    stateText: json ? JSON.stringify(state, null, 2) : (state ?? ""),
    images: Array.isArray(images) ? images : [],
  };
}

function examplesFrom(body) {
  const listed = Array.isArray(body.examples)
    ? body.examples.filter((item) => item && typeof item === "object" && !Array.isArray(item))
    : [];
  if (!listed.length) return [exampleFrom(FIRST_EXAMPLE, body.state, body.images)];
  const taken = new Set();
  return listed.map((item, index) => {
    const base = typeof item.name === "string" && item.name.trim() ? item.name.trim() : `Example ${index + 1}`;
    let name = base;
    for (let n = 2; taken.has(name); n += 1) name = `${base} ${n}`;
    taken.add(name);
    return exampleFrom(name, item.state, item.images);
  });
}

// A dataset's rows as examples (the Playground on a dataset): named by their
// id, read-only, each with its label values and labels.
export function rowsFrom(rows) {
  if (!rows.length) return [{ ...exampleFrom(FIRST_EXAMPLE, ""), row: { values: {}, labels: {}, empty: true } }];
  return rows.map((row) => ({
    ...exampleFrom(row.id, row.state, []),
    row: { index: row.index, values: row.values ?? {}, labels: row.labels ?? {} },
  }));
}

export function fromBody(body) {
  const value = body && typeof body === "object" ? body : {};
  const { state, questions, images, examples: listed, ...rest } = value;
  const examples = examplesFrom(value);
  return {
    rest,
    examples,
    current: examples[0].uid,
    questions:
      questions && typeof questions === "object" && !Array.isArray(questions)
        ? Object.entries(questions).map(([id, item]) => questionFrom(id, item))
        : [],
  };
}

// The selected example (the first when the selection is gone).
export const currentExample = (draft) =>
  draft.examples.find((item) => item.uid === draft.current) ?? draft.examples[0];

export const exampleByName = (draft, name) => draft.examples.find((item) => item.name === name) ?? null;

// An example's state for the request, or a problem when its JSON does not parse.
function exampleState(example) {
  if (example.stateMode !== "json") return { state: example.stateText, problem: null };
  try {
    return { state: JSON.parse(example.stateText), problem: null };
  } catch (error) {
    const where = { state: true, example: example.uid };
    return { state: example.stateText, problem: { where, message: `The state is not valid JSON: ${error.message}` } };
  }
}

function questionValue(question) {
  const value = { ...question.rest, type: question.type };
  const instructions = fieldValue(question.instructions);
  if (instructions !== null) value.instructions = instructions;
  if (question.type === "choice")
    value.criteria = Object.fromEntries(
      question.options.map((item) => [item.key, fieldValue(item.description)]),
    );
  else if (question.type === "score")
    value.criteria = question.levels.map((item) => fieldValue(item.description, ""));
  else {
    const yes = fieldValue(question.yes);
    const no = fieldValue(question.no);
    if (yes !== null || no !== null) value.criteria = { true: yes, false: no };
  }
  return value;
}

// The questions as the request has them, and their problems.
function questionsOf(draft) {
  const problems = [];
  const questions = {};
  const seen = new Set();
  for (const question of draft.questions) {
    const id = question.id;
    if (id.trim() === "")
      problems.push({ where: { q: question.uid, part: "id" }, message: "Give the question an id" });
    else if (seen.has(id))
      problems.push({ where: { q: question.uid, part: "id" }, message: `Two questions are named ${id}` });
    seen.add(id);
    if (question.type === "choice") {
      const keys = new Set();
      for (const item of question.options) {
        if (item.key.trim() === "")
          problems.push({ where: { q: question.uid, part: item.uid }, message: "Give the option a key" });
        else if (keys.has(item.key))
          problems.push({ where: { q: question.uid, part: item.uid }, message: `Two options are named ${item.key}` });
        keys.add(item.key);
      }
    }
    questions[id] = questionValue(question);
  }
  return { questions, problems };
}

// The request one example makes (the selected one by default), and the
// problems that keep it from being one: each {where, message}, where naming
// the question (uid) and field, or the state.
export function toBody(draft, exampleUid = draft.current) {
  const example = draft.examples.find((item) => item.uid === exampleUid) ?? draft.examples[0];
  const { state, problem } = exampleState(example);
  const { questions, problems } = questionsOf(draft);
  const body = { ...draft.rest, state, questions };
  if (example.images.length) body.images = example.images;
  return { body, problems: problem ? [problem, ...problems] : problems };
}

// The set's file: the first example as the top-level state and images, and
// every example in `examples` (left out for one example named "Example").
// Problems are every example's and the questions'.
export function toFile(draft) {
  const { questions, problems } = questionsOf(draft);
  const states = draft.examples.map((example) => {
    const { state, problem } = exampleState(example);
    if (problem) problems.unshift(problem);
    return state;
  });
  const names = new Set();
  for (const example of draft.examples) {
    if (!example.name.trim())
      problems.push({ where: { example: example.uid, name: true }, message: "Give the example a name" });
    else if (names.has(example.name))
      problems.push({ where: { example: example.uid, name: true }, message: `Two examples are named ${example.name}` });
    names.add(example.name);
  }
  const [first] = draft.examples;
  const body = { ...draft.rest, state: states[0], questions };
  if (first.images.length) body.images = first.images;
  if (draft.examples.length > 1 || first.name !== FIRST_EXAMPLE)
    body.examples = draft.examples.map((example, index) => {
      const item = { name: example.name, state: states[index] };
      if (example.images.length) item.images = example.images;
      return item;
    });
  return { body, problems };
}

// Editing examples: each returns a new draft.

export const editExample = (draft, exampleUid, fields) => ({
  ...draft,
  examples: draft.examples.map((item) => (item.uid === exampleUid ? { ...item, ...fields } : item)),
});

export const selectExample = (draft, exampleUid) => ({ ...draft, current: exampleUid });

export function freeExampleName(draft, base = "Example") {
  const taken = new Set(draft.examples.map((item) => item.name));
  for (let n = draft.examples.length + 1; ; n += 1) if (!taken.has(`${base} ${n}`)) return `${base} ${n}`;
}

export function addExample(draft) {
  const example = exampleFrom(freeExampleName(draft), "", []);
  return { ...draft, examples: [...draft.examples, example], current: example.uid };
}

export function duplicateExample(draft, exampleUid) {
  const index = draft.examples.findIndex((item) => item.uid === exampleUid);
  const source = draft.examples[index];
  const taken = new Set(draft.examples.map((item) => item.name));
  let name = `${source.name} copy`;
  for (let n = 2; taken.has(name); n += 1) name = `${source.name} copy ${n}`;
  const copy = { ...source, uid: uid(), name };
  const examples = [...draft.examples];
  examples.splice(index + 1, 0, copy);
  return { ...draft, examples, current: copy.uid };
}

// Remove an example; the last one stays. The neighbour is selected.
export function removeExample(draft, exampleUid) {
  if (draft.examples.length < 2) return draft;
  const index = draft.examples.findIndex((item) => item.uid === exampleUid);
  const examples = draft.examples.filter((item) => item.uid !== exampleUid);
  const current = draft.current === exampleUid ? examples[Math.min(index, examples.length - 1)].uid : draft.current;
  return { ...draft, examples, current };
}

export function moveExample(draft, exampleUid, by) {
  const examples = [...draft.examples];
  const from = examples.findIndex((item) => item.uid === exampleUid);
  const to = from + by;
  if (from < 0 || to < 0 || to >= examples.length) return draft;
  [examples[from], examples[to]] = [examples[to], examples[from]];
  return { ...draft, examples };
}

// Where a server error's param points: the state or an image (of the selected
// example), a question and its field, or the model.
export function locate(draft, param) {
  if (!param) return { general: true };
  if (param === "state" || param.startsWith("state.")) return { state: true, example: draft.current };
  if (param === "images" || param.startsWith("images")) return { images: true, example: draft.current };
  if (param === "model") return { model: true };
  if (!param.startsWith("questions.")) return { general: true };
  const path = param.slice("questions.".length);
  const question = [...draft.questions]
    .sort((a, b) => b.id.length - a.id.length)
    .find((item) => path === item.id || path.startsWith(`${item.id}.`));
  if (!question) return { general: true };
  let parts = path.slice(question.id.length + 1).split(".").filter(Boolean);
  if (TYPES.includes(parts[0])) parts = parts.slice(1); // the validator's type tag
  const [name, index] = parts;
  if (name === "instructions") return { q: question.uid, part: "instructions" };
  if (name === "type") return { q: question.uid, part: "type" };
  if (name === "criteria" && index !== undefined) {
    if (question.type === "choice") {
      const item = question.options.find((entry) => entry.key === index);
      if (item) return { q: question.uid, part: item.uid };
    }
    if (question.type === "score" && question.levels[Number(index)])
      return { q: question.uid, part: question.levels[Number(index)].uid };
    if (question.type === "noul") return { q: question.uid, part: index === "true" ? "yes" : "no" };
  }
  return { q: question.uid, part: name === "criteria" ? "criteria" : "question" };
}

// Editing: each returns a new draft.

const mapQuestion = (draft, questionUid, change) => ({
  ...draft,
  questions: draft.questions.map((item) => (item.uid === questionUid ? change(item) : item)),
});

export const editQuestion = (draft, questionUid, fields) =>
  mapQuestion(draft, questionUid, (item) => ({ ...item, ...fields }));

export function freeId(draft, base) {
  const taken = new Set(draft.questions.map((item) => item.id));
  if (!taken.has(base)) return base;
  let n = 2;
  while (taken.has(`${base}_${n}`)) n += 1;
  return `${base}_${n}`;
}

export function newQuestion(type, id) {
  const question = questionFrom(id, { type });
  if (type === "choice") question.options = [option("option_a", typed("")), option("option_b", typed(""))];
  if (type === "score") question.levels = [level(typed("")), level(typed(""))];
  return question;
}

export function addQuestion(draft, type) {
  const question = newQuestion(type, freeId(draft, type));
  return [{ ...draft, questions: [...draft.questions, question] }, question];
}

export function removeQuestion(draft, questionUid) {
  return { ...draft, questions: draft.questions.filter((item) => item.uid !== questionUid) };
}

export function duplicateQuestion(draft, questionUid) {
  const index = draft.questions.findIndex((item) => item.uid === questionUid);
  const source = draft.questions[index];
  const copy = {
    ...source,
    uid: uid(),
    id: freeId(draft, `${source.id}_copy`),
    options: source.options.map((item) => ({ ...item, uid: uid() })),
    levels: source.levels.map((item) => ({ ...item, uid: uid() })),
  };
  const questions = [...draft.questions];
  questions.splice(index + 1, 0, copy);
  return [{ ...draft, questions }, copy];
}

export function moveQuestion(draft, questionUid, by) {
  const questions = [...draft.questions];
  const from = questions.findIndex((item) => item.uid === questionUid);
  const to = from + by;
  if (to < 0 || to >= questions.length) return draft;
  [questions[from], questions[to]] = [questions[to], questions[from]];
  return { ...draft, questions };
}

const keyFor = (text, taken, fallback) => {
  let key = text.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 24);
  key = key || fallback;
  let candidate = key;
  for (let n = 2; taken.has(candidate); n += 1) candidate = `${key}_${n}`;
  taken.add(candidate);
  return candidate;
};

// Change a question's type, keeping what carries over: instructions always;
// options and levels become each other; yes and no become two of them.
export function changeType(draft, questionUid, type) {
  return mapQuestion(draft, questionUid, (question) => {
    if (question.type === type) return question;
    const texts =
      question.type === "choice"
        ? question.options.map((item) => (item.description.text !== "" ? item.description : typed(item.key)))
        : question.type === "score"
          ? question.levels.map((item) => item.description)
          : [question.no, question.yes];
    const changed = { ...question, type, options: [], levels: [], yes: typed(""), no: typed("") };
    if (type === "choice") {
      const taken = new Set();
      changed.options =
        question.type === "noul"
          ? [option("yes", question.yes), option("no", question.no)]
          : texts.map((text, index) => option(keyFor(text.text, taken, `option_${index + 1}`), text));
    } else if (type === "score") {
      changed.levels = texts.map((text) => level(text));
      while (changed.levels.length < 2) changed.levels.push(level(typed("")));
    } else if (question.type === "choice" && question.options.length === 2) {
      changed.yes = question.options[0].description;
      changed.no = question.options[1].description;
    }
    return changed;
  });
}

// Options and levels: the question's list under `list` ("options" or "levels").
export function editItem(draft, questionUid, list, itemUid, fields) {
  return mapQuestion(draft, questionUid, (question) => ({
    ...question,
    [list]: question[list].map((item) => (item.uid === itemUid ? { ...item, ...fields } : item)),
  }));
}

export function addItem(draft, questionUid, list) {
  let added;
  const next = mapQuestion(draft, questionUid, (question) => {
    if (list === "options") {
      const taken = new Set(question.options.map((item) => item.key));
      added = option(keyFor("", taken, `option_${question.options.length + 1}`), typed(""));
    } else added = level(typed(""));
    return { ...question, [list]: [...question[list], added] };
  });
  return [next, added];
}

export function removeItem(draft, questionUid, list, itemUid) {
  return mapQuestion(draft, questionUid, (question) => ({
    ...question,
    [list]: question[list].filter((item) => item.uid !== itemUid),
  }));
}

export function moveItem(draft, questionUid, list, itemUid, toIndex) {
  return mapQuestion(draft, questionUid, (question) => {
    const items = [...question[list]];
    const from = items.findIndex((item) => item.uid === itemUid);
    if (from < 0 || toIndex < 0 || toIndex >= items.length || toIndex === from) return question;
    const [moved] = items.splice(from, 1);
    items.splice(toIndex, 0, moved);
    return { ...question, [list]: items };
  });
}

// An example's state as the other mode shows it.
export function switchStateMode(example, mode) {
  if (example.stateMode === mode) return example;
  if (mode === "json") {
    const text = example.stateText;
    try {
      JSON.parse(text);
      return { ...example, stateMode: "json" };
    } catch {
      return { ...example, stateMode: "json", stateText: JSON.stringify(text) };
    }
  }
  try {
    const value = JSON.parse(example.stateText);
    return {
      ...example,
      stateMode: "text",
      stateText: typeof value === "string" ? value : JSON.stringify(value, null, 2),
    };
  } catch {
    return { ...example, stateMode: "text" };
  }
}

// Questions whose request differs from `request` (a run's), by id; all of
// them when the state or the images changed.
export function changedSince(body, request) {
  if (!request) return { all: false, ids: new Set() };
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  const all = !same(body.state, request.state) || !same(body.images ?? [], request.images ?? []);
  const ids = new Set();
  for (const [id, question] of Object.entries(body.questions)) {
    if (!same(question, request.questions?.[id])) ids.add(id);
  }
  return { all, ids };
}

// ---------- a dataset's answers map ----------
//
// A dataset's questions are answered by the values of one of its label
// columns. The draft keeps, per question uid, {column, map, left_out}: the map
// from each value to an option's uid (choice), true / false (noul) or a
// level's uid (score), so it follows options and levels renamed or moved.

export const LEFT_OUT = "\u0000out";

// The draft's map from the file's: option keys and level indices to uids;
// targets the question does not take are dropped.
export function answersFromFile(draft, answers) {
  const byUid = {};
  for (const question of draft.questions) {
    const answer = answers?.[question.id];
    if (!answer) continue;
    const map = {};
    for (const [value, target] of Object.entries(answer.map ?? {})) {
      const uid = targetUid(question, target);
      if (uid !== null) map[value] = uid;
    }
    byUid[question.uid] = { column: answer.column, map, left_out: [...(answer.left_out ?? [])] };
  }
  return byUid;
}

function targetUid(question, target) {
  if (question.type === "choice") return question.options.find((option) => option.key === target)?.uid ?? null;
  if (question.type === "noul") return target === true || target === "true" ? true : target === false || target === "false" ? false : null;
  const level = Number(target);
  return Number.isInteger(level) ? (question.levels[level]?.uid ?? null) : null;
}

// What a value of the map stands for in the file: an option's key, true /
// false, a level's index; null when the question no longer has it.
export function targetValue(question, uid) {
  if (question.type === "choice") return question.options.find((option) => option.uid === uid)?.key ?? null;
  if (question.type === "noul") return typeof uid === "boolean" ? uid : null;
  const index = question.levels.findIndex((level) => level.uid === uid);
  return index >= 0 ? index : null;
}

// The map as dataset.json keeps it, by the questions' current ids.
export function answersFor(draft) {
  const answers = {};
  for (const question of draft.questions) {
    const answer = draft.answers?.[question.uid];
    if (!answer?.column || !question.id.trim()) continue;
    const map = {};
    for (const [value, uid] of Object.entries(answer.map ?? {})) {
      const target = targetValue(question, uid);
      if (target !== null) map[value] = target;
    }
    answers[question.id] = { column: answer.column, map, left_out: [...(answer.left_out ?? [])] };
  }
  return answers;
}

const answerOf = (draft, questionUid) => draft.answers?.[questionUid] ?? { column: null, map: {}, left_out: [] };
const withAnswer = (draft, questionUid, answer) => ({
  ...draft,
  answers: { ...(draft.answers ?? {}), [questionUid]: answer },
});

// "Answered by": the column whose values answer the question (null: asked, not trained).
export function setAnswerColumn(draft, questionUid, column) {
  if (!column) {
    const { [questionUid]: _, ...rest } = draft.answers ?? {};
    return { ...draft, answers: rest };
  }
  return withAnswer(draft, questionUid, { column, map: {}, left_out: [] });
}

// One value's target: an option's or level's uid, true / false, LEFT_OUT, or
// null (not mapped).
export function mapValue(draft, questionUid, value, target) {
  const answer = answerOf(draft, questionUid);
  const map = { ...answer.map };
  delete map[value];
  const leftOut = answer.left_out.filter((item) => item !== value);
  if (target === LEFT_OUT) leftOut.push(value);
  else if (target !== null) map[value] = target;
  return withAnswer(draft, questionUid, { ...answer, map, left_out: leftOut });
}

// An id from a name: lower case, other characters than letters, digits, '.'
// and '-' as '_' (as the data command makes them); a repeated one gets _2, _3.
export function safeIds(names, fallback = "option") {
  const ids = [];
  for (const name of names) {
    const base =
      String(name)
        .toLowerCase()
        .replace(/[^\p{L}\p{N}_.-]+/gu, "_")
        .replace(/^_+|_+$/g, "") || fallback;
    let id = base;
    for (let n = 2; ids.includes(id); n += 1) id = `${base}_${n}`;
    ids.push(id);
  }
  return ids;
}

// Options from values: one option per value (``values``: [[value, name]]),
// its id made from the name, the name as its description, each value mapped
// to its option.
export function optionsFromValues(draft, questionUid, values) {
  const ids = safeIds(values.map(([value, name]) => name || value));
  const options = values.map(([value, name], index) => option(ids[index], typed(name || value)));
  const map = Object.fromEntries(values.map(([value], index) => [value, options[index].uid]));
  const next = mapQuestion(draft, questionUid, (question) => ({ ...question, options }));
  const answer = answerOf(draft, questionUid);
  return withAnswer(next, questionUid, { ...answer, map, left_out: [] });
}

// A row's expected answer for a question, as the Run All grid shows answers:
// a choice's option key, "yes" / "no", a level from 1; null when its value is
// not mapped or left out.
export function expectedAnswer(draft, question, values) {
  const answer = draft.answers?.[question.uid];
  const value = answer ? values?.[answer.column] : undefined;
  if (value === undefined || value === null || !(value in (answer.map ?? {}))) return null;
  const target = targetValue(question, answer.map[value]);
  if (target === null) return null;
  if (question.type === "noul") return target ? "yes" : "no";
  if (question.type === "score") return String(target + 1);
  return target;
}
