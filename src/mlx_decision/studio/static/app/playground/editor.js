// The Playground's centre: the state, one group per question, Add Question
// and the live check, each problem shown at the field it is about.
import { html } from "htm/preact";
import { useRef, useState } from "preact/hooks";
import * as icons from "../components/icons.js";
import { MenuButton, MenuHeader, MenuItem, MenuSeparator } from "../components/menu.js";
import { plural } from "../format.js";
import { toast } from "../store/ui.js";
import * as edit from "./draft.js";
import {
  built,
  draft,
  lastRun,
  linked,
  model,
  modelName,
  problems,
  stale,
  updateDraft,
} from "./session.js";

const IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp"];

// The first problem whose place matches `test`, or null.
const problemWhere = (test) => problems.value.find((problem) => test(problem.where)) ?? null;
const at = (questionUid, part) =>
  problemWhere((where) => where.q === questionUid && where.part === part);

const Problem = ({ problem }) =>
  problem && html`<div class="field-problem" role="alert">${problem.message}</div>`;

// Focus a field once the next render has drawn it.
export function focusLater(selector) {
  requestAnimationFrame(() => {
    const element = document.querySelector(selector);
    element?.focus();
    element?.scrollIntoView({ block: "center", behavior: "smooth" });
  });
}

const Minus = ({ label, onClick }) => html`
  <button class="minus" aria-label=${label} title=${label} onClick=${onClick}>
    <svg viewBox="0 0 8 8" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M1.5 4h5" /></svg>
  </button>
`;

const Grip = () => html`
  <svg class="grip" viewBox="0 0 10 14" fill="currentColor" aria-hidden="true">
    <circle cx="3" cy="3" r="1" /><circle cx="7" cy="3" r="1" /><circle cx="3" cy="7" r="1" />
    <circle cx="7" cy="7" r="1" /><circle cx="3" cy="11" r="1" /><circle cx="7" cy="11" r="1" />
  </svg>
`;

// ---------- the state ----------

const readImage = (file) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });

export async function addImages(files) {
  const images = [...files].filter((file) => IMAGE_TYPES.includes(file.type));
  if (images.length < files.length) toast("Only PNG, JPEG and WebP images can be added");
  if (!images.length) return;
  const urls = await Promise.all(images.map(readImage));
  updateDraft((current) => ({ ...current, images: [...current.images, ...urls] }));
}

function Images() {
  const current = draft.value;
  const reads = model.value?.capabilities?.supports_images ?? false;
  const picker = useRef(null);
  const [over, setOver] = useState(false);
  const problem = problemWhere((where) => where.images);
  const remove = (index) =>
    updateDraft((value) => ({ ...value, images: value.images.filter((_, i) => i !== index) }));
  const onDrop = (event) => {
    event.preventDefault();
    setOver(false);
    if (reads) addImages(event.dataTransfer.files);
  };
  return html`
    ${current.images.length > 0 &&
    html`<div class="thumbs">
      ${current.images.map(
        (url, index) => html`
          <figure class="thumb" key=${index}>
            ${typeof url === "string" && url.startsWith("data:image/")
              ? html`<img src=${url} alt=${`Image ${index + 1}`} />`
              : html`<span class="mono">not a data URL</span>`}
            <figcaption class="num">[image ${index + 1}]</figcaption>
            <${Minus} label=${`Remove image ${index + 1}`} onClick=${() => remove(index)} />
          </figure>
        `,
      )}
    </div>`}
    <div
      class=${`drop ${over ? "over" : ""} ${reads ? "" : "off"}`}
      onDragOver=${(event) => {
        event.preventDefault();
        setOver(reads);
      }}
      onDragLeave=${() => setOver(false)}
      onDrop=${onDrop}
    >
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.3" aria-hidden="true">
        <rect x="2.5" y="3.5" width="15" height="13" rx="2" /><circle cx="7" cy="8" r="1.5" />
        <path d="m2.5 14 4.5-4 3.5 3 2.5-2 4.5 4" />
      </svg>
      <span>
        ${reads
          ? html`Drop or paste images here <span class="dim">· PNG, JPEG, WebP</span>`
          : `${modelName.value ?? "This model"} reads text only. Pick a model that reads images to add them.`}
      </span>
      ${reads &&
      html`<button class="btn" onClick=${() => picker.current.click()}>Add Image…</button>
        <input
          ref=${picker}
          type="file"
          accept=${IMAGE_TYPES.join(",")}
          multiple
          hidden
          onChange=${(event) => {
            addImages(event.target.files);
            event.target.value = "";
          }}
        />`}
    </div>
    <${Problem} problem=${problem} />
  `;
}

const rowsFor = (text, least = 3, most = 14) =>
  Math.min(most, Math.max(least, text.split("\n").length + 1));

function State() {
  const current = draft.value;
  const json = current.stateMode === "json";
  const problem = problemWhere((where) => where.state);
  const run = lastRun.value;
  const setMode = (mode) => updateDraft((value) => edit.switchStateMode(value, mode));
  const onPaste = (event) => {
    const files = [...(event.clipboardData?.files ?? [])];
    if (files.length && model.value?.capabilities?.supports_images) {
      event.preventDefault();
      addImages(files);
    }
  };
  return html`
    <section class="gsec">
      <div class="gsec-head">
        <h2>State</h2>
        <span class="tag">What the model reads</span>
        <div class="right">
          <div class="seg small" role="group" aria-label="State format">
            <button aria-pressed=${!json} onClick=${() => setMode("text")}>Text</button>
            <button aria-pressed=${json} onClick=${() => setMode("json")}>JSON</button>
          </div>
        </div>
      </div>
      <div class=${`group ${problem ? "has-problem" : ""}`}>
        <div class="state-box">
          <textarea
            class=${`field state-text ${json ? "mono" : ""}`}
            aria-label=${json ? "State as JSON" : "State text"}
            rows=${rowsFor(current.stateText)}
            spellcheck=${!json}
            placeholder=${json ? '{"ticket": "..."}' : "What the model reads: a ticket, a review, a message…"}
            value=${current.stateText}
            onInput=${(event) => updateDraft((value) => ({ ...value, stateText: event.target.value }))}
            onPaste=${onPaste}
          ></textarea>
          ${json && html`<${Problem} problem=${problem} />`}
          <${Images} />
          <div class="state-foot">
            <span>${json ? "Sent as JSON; the model reads keys and values" : `${current.stateText.length.toLocaleString("en")} characters, plain text`}</span>
            ${run && html`<span class="num">Last run: ${run.input_tokens} input tokens${run.truncated ? ", the state was cut" : ""}</span>`}
          </div>
          ${!json && html`<${Problem} problem=${problem} />`}
        </div>
      </div>
    </section>
  `;
}

// ---------- questions ----------

function moveOnKeys(event, question, list, item, index) {
  if (!event.altKey || (event.key !== "ArrowUp" && event.key !== "ArrowDown")) return;
  event.preventDefault();
  const to = index + (event.key === "ArrowUp" ? -1 : 1);
  updateDraft((value) => edit.moveItem(value, question.uid, list, item.uid, to));
  const target = event.target.dataset.cell;
  focusLater(`[data-item="${item.uid}"] [data-cell="${target}"]`);
}

// Drag a row by its grip within its list.
function useDrag(question, list) {
  const [dragged, setDragged] = useState(null);
  return {
    grip: (item) => ({
      draggable: true,
      onDragStart: (event) => {
        setDragged(item.uid);
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData("text/plain", item.uid);
      },
      onDragEnd: () => setDragged(null),
    }),
    target: (index) => ({
      onDragOver: (event) => {
        if (dragged) event.preventDefault();
      },
      onDrop: (event) => {
        if (!dragged) return;
        event.preventDefault();
        updateDraft((value) => edit.moveItem(value, question.uid, list, dragged, index));
        setDragged(null);
      },
    }),
    dragged,
  };
}

function ChoiceRows({ question }) {
  const drag = useDrag(question, "options");
  const change = (item, fields) =>
    updateDraft((value) => edit.editItem(value, question.uid, "options", item.uid, fields));
  const add = () => {
    let added;
    updateDraft((value) => {
      const [next, item] = edit.addItem(value, question.uid, "options");
      added = item;
      return next;
    });
    focusLater(`[data-item="${added.uid}"] [data-cell="key"]`);
  };
  return html`
    ${question.options.map(
      (item, index) => html`
        <div class=${`row opt-row ${drag.dragged === item.uid ? "dragging" : ""}`} key=${item.uid}
             data-item=${item.uid} ...${drag.target(index)}>
          <span class="key-cell">
            <span class="grip-handle" title="Drag to reorder (or ⌥↑ / ⌥↓)" ...${drag.grip(item)}><${Grip} /></span>
            <input class=${`opt-key ${at(question.uid, item.uid) ? "bad" : ""}`} data-cell="key"
                   aria-label="Option key" spellcheck="false" value=${item.key}
                   onInput=${(event) => change(item, { key: event.target.value })}
                   onKeyDown=${(event) => moveOnKeys(event, question, "options", item, index)} />
          </span>
          <input class="field" data-cell="description" aria-label=${`Description of ${item.key}`}
                 placeholder="Describe this option" value=${item.description.text}
                 onInput=${(event) => change(item, { description: { ...item.description, text: event.target.value } })}
                 onKeyDown=${(event) => moveOnKeys(event, question, "options", item, index)} />
          <${Minus} label=${`Remove option ${item.key}`}
                    onClick=${() => updateDraft((value) => edit.removeItem(value, question.uid, "options", item.uid))} />
          <${Problem} problem=${at(question.uid, item.uid)} />
        </div>
      `,
    )}
    <div class="add-row"><button class="btn plain" onClick=${add}>+ Add Option</button></div>
  `;
}

function ScoreRows({ question }) {
  const drag = useDrag(question, "levels");
  const last = question.levels.length - 1;
  const add = () => {
    let added;
    updateDraft((value) => {
      const [next, item] = edit.addItem(value, question.uid, "levels");
      added = item;
      return next;
    });
    focusLater(`[data-item="${added.uid}"] [data-cell="description"]`);
  };
  return html`
    ${question.levels.map(
      (item, index) => html`
        <div class=${`row opt-row ${drag.dragged === item.uid ? "dragging" : ""}`} key=${item.uid}
             data-item=${item.uid} ...${drag.target(index)}>
          <span class="level-lbl">
            <span class="grip-handle" title="Drag to reorder (or ⌥↑ / ⌥↓)" ...${drag.grip(item)}><${Grip} /></span>
            <span class="level-n num">${index}</span>
            <span class="hint">${index === 0 ? "lowest" : index === last ? "highest" : ""}</span>
          </span>
          <input class="field" data-cell="description" aria-label=${`Level ${index}`}
                 placeholder="Describe this level" value=${item.description.text}
                 onInput=${(event) =>
                   updateDraft((value) =>
                     edit.editItem(value, question.uid, "levels", item.uid, {
                       description: { ...item.description, text: event.target.value },
                     }),
                   )}
                 onKeyDown=${(event) => moveOnKeys(event, question, "levels", item, index)} />
          <${Minus} label=${`Remove level ${index}`}
                    onClick=${() => updateDraft((value) => edit.removeItem(value, question.uid, "levels", item.uid))} />
          <${Problem} problem=${at(question.uid, item.uid)} />
        </div>
      `,
    )}
    <div class="add-row"><button class="btn plain" onClick=${add}>+ Add Level</button></div>
  `;
}

function NoulRows({ question }) {
  const row = (part, label, placeholder) => html`
    <div class="row">
      <label for=${`${part}-${question.uid}`}>${label}</label>
      <input id=${`${part}-${question.uid}`} class="field" placeholder=${placeholder}
             value=${question[part].text}
             onInput=${(event) =>
               updateDraft((value) =>
                 edit.editQuestion(value, question.uid, { [part]: { ...question[part], text: event.target.value } }),
               )} />
      <${Problem} problem=${at(question.uid, part)} />
    </div>
  `;
  return html`
    ${row("yes", "When yes", "Optional: what counts as yes")}
    ${row("no", "When no", "Optional: what counts as no")}
  `;
}

function QuestionMenu({ question, index, count }) {
  const act = (change, focus) => () => {
    let added;
    updateDraft((value) => {
      const result = change(value);
      if (Array.isArray(result)) {
        [value, added] = result;
        return value;
      }
      return result;
    });
    if (focus && added) focusLater(`#q-${added.uid} .qid-input`);
  };
  return html`
    <${MenuButton}
      id=${`question-${question.uid}`}
      className="icon-btn"
      ariaLabel=${`More for ${question.id}`}
      title="Move, duplicate, change type, remove"
      label=${html`<svg viewBox="0 0 20 20" fill="currentColor"><circle cx="5" cy="10" r="1.4" /><circle cx="10" cy="10" r="1.4" /><circle cx="15" cy="10" r="1.4" /></svg>`}
      menuStyle="top:30px; right:0; width:230px"
    >
      <${MenuItem} disabled=${index === 0} onSelect=${act((value) => edit.moveQuestion(value, question.uid, -1))}>Move Up<//>
      <${MenuItem} disabled=${index === count - 1} onSelect=${act((value) => edit.moveQuestion(value, question.uid, 1))}>Move Down<//>
      <${MenuItem} onSelect=${act((value) => edit.duplicateQuestion(value, question.uid), true)}>Duplicate<//>
      <${MenuSeparator} />
      <${MenuHeader}>Type<//>
      ${edit.TYPES.map(
        (type) => html`<${MenuItem} ticked=${question.type === type} sub=${edit.TYPE_DESCRIPTIONS[type]}
                         onSelect=${act((value) => edit.changeType(value, question.uid, type))}>${edit.TYPE_LABELS[type]}<//>`,
      )}
      <${MenuSeparator} />
      <${MenuItem} onSelect=${act((value) => edit.removeQuestion(value, question.uid))}>Remove Question<//>
    <//>
  `;
}

function Question({ question, index, count }) {
  const editedIds = stale.value.ids;
  const edited = !!lastRun.value && editedIds.has(question.id);
  const requires = model.value?.capabilities?.requires_instructions;
  const isLinked = linked.value === question.id;
  const typeProblem = at(question.uid, "type");
  const wholeProblem = at(question.uid, "question") ?? at(question.uid, "criteria");
  return html`
    <section
      class=${`gsec question ${isLinked ? "linked" : ""}`}
      id=${`q-${question.uid}`}
      data-q=${question.id}
      onFocusIn=${() => {
        linked.value = question.id;
      }}
      onMouseEnter=${() => {
        if (!document.activeElement?.closest?.(".question")) linked.value = question.id;
      }}
    >
      <div class="gsec-head">
        <h2>
          <input class=${`qid-input mono ${at(question.uid, "id") ? "bad" : ""}`} aria-label="Question id"
                 spellcheck="false" size=${Math.max(4, question.id.length + 1)} value=${question.id}
                 onInput=${(event) => updateDraft((value) => edit.editQuestion(value, question.uid, { id: event.target.value }))} />
        </h2>
        <span class=${`tag ${typeProblem ? "bad" : ""}`}>${edit.TYPE_LABELS[question.type]}</span>
        ${edited && html`<span class="stale-tag">Edited</span>`}
        <div class="right"><${QuestionMenu} question=${question} index=${index} count=${count} /></div>
      </div>
      <${Problem} problem=${at(question.uid, "id") ?? typeProblem} />
      <div class=${`group ${wholeProblem ? "has-problem" : ""}`}>
        <div class="row">
          <label for=${`ins-${question.uid}`}>Instructions</label>
          <input id=${`ins-${question.uid}`} class="field ins"
                 placeholder=${requires ? `Required by ${modelName.value}` : "What should the model decide?"}
                 value=${question.instructions.text}
                 onInput=${(event) =>
                   updateDraft((value) =>
                     edit.editQuestion(value, question.uid, {
                       instructions: { ...question.instructions, text: event.target.value },
                     }),
                   )} />
          <${Problem} problem=${at(question.uid, "instructions")} />
        </div>
        ${question.type === "choice" && html`<${ChoiceRows} question=${question} />`}
        ${question.type === "score" && html`<${ScoreRows} question=${question} />`}
        ${question.type === "noul" && html`<${NoulRows} question=${question} />`}
      </div>
      <${Problem} problem=${wholeProblem} />
    </section>
  `;
}

function AddQuestion() {
  const add = (type) => {
    let question;
    updateDraft((value) => {
      const [next, added] = edit.addQuestion(value, type);
      question = added;
      return next;
    });
    focusLater(`#q-${question.uid} .ins`);
  };
  return html`
    <${MenuButton}
      id="add-question"
      className="btn"
      label=${html`<${icons.Plus} /> Add Question`}
      menuStyle="bottom:32px; left:0; width:300px"
    >
      ${edit.TYPES.map(
        (type) => html`<${MenuItem} sub=${edit.TYPE_DESCRIPTIONS[type]} onSelect=${() => add(type)}>${edit.TYPE_LABELS[type]}<//>`,
      )}
    <//>
  `;
}

function CheckLine() {
  const open = problems.value;
  const current = built.value;
  const count = Object.keys(current.body.questions).length;
  if (open.length)
    return html`
      <div class="check">
        <span class="bad"><${icons.StatusIcon} kind="warn" />${open[0].message}${
          open.length > 1 ? ` (and ${open.length - 1} more)` : ""
        }</span>
      </div>
    `;
  if (!modelName.value) return html`<div class="check"></div>`;
  return html`
    <div class="check">
      <span class="ok"><${icons.StatusIcon} kind="ok" />${plural(count, "question")}, within ${modelName.value}'s limits</span>
      <span class="info"><${icons.StatusIcon} kind="info" />Input length is checked when it runs</span>
    </div>
  `;
}

export function Editor() {
  const current = draft.value;
  return html`
    <${State} />
    <div class="questions">
      ${current.questions.map(
        (question, index) =>
          html`<${Question} key=${question.uid} question=${question} index=${index} count=${current.questions.length} />`,
      )}
    </div>
    <div class="add-bar">
      <${AddQuestion} />
      <${CheckLine} />
    </div>
  `;
}
