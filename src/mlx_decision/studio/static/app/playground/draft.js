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

export function fromBody(body) {
  const { state, questions, images, ...rest } = body && typeof body === "object" ? body : {};
  const json = state !== undefined && typeof state !== "string";
  return {
    rest,
    stateMode: json ? "json" : "text",
    stateText: json ? JSON.stringify(state, null, 2) : (state ?? ""),
    images: Array.isArray(images) ? images : [],
    questions:
      questions && typeof questions === "object" && !Array.isArray(questions)
        ? Object.entries(questions).map(([id, value]) => questionFrom(id, value))
        : [],
  };
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

// The request body, and the problems that keep it from being one: each
// {where, message}, where naming the question (uid) and field.
export function toBody(draft) {
  const problems = [];
  let state = draft.stateText;
  if (draft.stateMode === "json") {
    try {
      state = JSON.parse(draft.stateText);
    } catch (error) {
      problems.push({ where: { state: true }, message: `The state is not valid JSON: ${error.message}` });
    }
  }
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
  const body = { ...draft.rest, state, questions };
  if (draft.images.length) body.images = draft.images;
  return { body, problems };
}

// Where a server error's param points: the state, an image, a question and
// its field, or the model.
export function locate(draft, param) {
  if (!param) return { general: true };
  if (param === "state" || param.startsWith("state.")) return { state: true };
  if (param === "images" || param.startsWith("images")) return { images: true };
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

// The state as the other mode shows it.
export function switchStateMode(draft, mode) {
  if (draft.stateMode === mode) return draft;
  if (mode === "json") {
    const text = draft.stateText;
    try {
      JSON.parse(text);
      return { ...draft, stateMode: "json" };
    } catch {
      return { ...draft, stateMode: "json", stateText: JSON.stringify(text) };
    }
  }
  try {
    const value = JSON.parse(draft.stateText);
    return {
      ...draft,
      stateMode: "text",
      stateText: typeof value === "string" ? value : JSON.stringify(value, null, 2),
    };
  } catch {
    return { ...draft, stateMode: "text" };
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
