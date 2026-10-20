// Convert: an 8-bit or 4-bit copy of a model (as `mlx-decision convert -q`),
// with its name and the size estimates; a job that holds the GPU, writing into
// the workspace's models/. The copy appears in Models when it is done.
import { computed, signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import { closeSheet, openSheet } from "../components/sheet.js";
import { size } from "../format.js";
import { gpuJob } from "../store/jobs.js";
import { loadedNames, models } from "../store/status.js";
import { toast } from "../store/ui.js";

const NAME = /^[A-Za-z0-9_-][A-Za-z0-9._-]{0,79}$/;

const planned = signal(null); // GET /models/convert
const bits = signal(8);
const name = signal("");
const named = signal(false); // the name was typed, so a change of bits keeps it
const starting = signal(false);

const short = (model) => model.split("/").pop();
const defaultName = (model, width) => `${short(model)}-q${width}`;

// Why the name cannot be used, or null.
const nameProblem = computed(() => {
  const value = name.value.trim();
  if (!value) return "Give the copy a name";
  if (!NAME.test(value)) return "Letters, digits, '.', '_' or '-' only, not starting with '.'";
  if ((models.value ?? []).some((model) => model.name === value)) return `A model named ${value} exists already`;
  return null;
});

const blocked = computed(() => {
  const job = gpuJob.value;
  return job ? `${job.title} is running; one job at a time can use the GPU` : null;
});

function setBits(width) {
  bits.value = width;
  if (!named.value) name.value = defaultName(planned.value.model, width);
}

async function start() {
  if (nameProblem.value || blocked.value || starting.value) return;
  starting.value = true;
  const model = planned.value.model;
  try {
    const { job } = await api("/jobs", {
      method: "POST",
      body: { kind: "convert", model, bits: bits.value, name: name.value.trim() },
    });
    closeSheet();
    toast(`${job.title}: the copy shows up in Models when it is done`);
  } catch (error) {
    toast(`Could not convert ${model}: ${error.message}`);
  } finally {
    starting.value = false;
  }
}

const FITS = {
  ok: ["ok", "Fits the memory budget"],
  over_budget: ["mid", "Over the memory budget"],
  too_large: ["no", "Too large for this Mac"],
};

// What holding the GPU means now, in one sentence.
function gpuNote(loaded) {
  const unloaded = loaded.length ? `${loaded.join(", ")} ${loaded.length === 1 ? "is" : "are"} unloaded, and ` : "";
  return `Converting needs the GPU to itself: ${unloaded}the Playground, Load and the API answer busy until it is done.`;
}

function Body() {
  const plan = planned.value;
  if (!plan) return null;
  const loaded = [...loadedNames.value];
  return html`
    <div class="opts">
      <h4>Store as</h4>
      <div class="qopts two" role="radiogroup" aria-label="Precision">
        ${plan.options.map((option) => {
          const [kind, words] = FITS[option.fit];
          return html`
            <button class="qopt" role="radio" aria-checked=${option.bits === bits.value ? "true" : "false"}
                    data-bits=${option.bits} onClick=${() => setBits(option.bits)}>
              <b>${option.bits}-bit</b>
              <span>${option.bits === 8 ? "close to the original" : "smallest, a little less exact"}</span>
              <span class="num">about ${size(option.bytes)} on disk</span>
              <span class="num">about ${size(option.memory_bytes)} to run</span>
              <span class=${`fit ${kind}`}>${words}</span>
            </button>
          `;
        })}
      </div>
      <span class="note">${plan.model}'s weights take ${size(plan.source_bytes)} now. Sizes are estimates.</span>
    </div>
    <label class="name-field">
      <span>Name</span>
      <input class="sheet-input mono" aria-label="Name" spellcheck=${false} value=${name.value}
             onInput=${(event) => {
               name.value = event.target.value;
               named.value = true;
             }} />
    </label>
    ${nameProblem.value && html`<p class="warn-text">${nameProblem.value}</p>`}
    <div class="notice"><span>${gpuNote(loaded)}${blocked.value && html` <b>${blocked.value}.</b>`}</span></div>
  `;
}

export async function openConvert(model) {
  let plan;
  try {
    plan = await api(`/models/convert?model=${encodeURIComponent(model)}`);
  } catch (error) {
    toast(`Could not plan the conversion: ${error.message}`);
    return;
  }
  if (!plan.convertible) {
    toast(`${model} cannot be converted: ${plan.reason}`);
    return;
  }
  planned.value = plan;
  named.value = false;
  setBits(8);
  openSheet({
    title: `Convert ${model}`,
    purpose: "A quantized copy that loads and runs with less memory, as mlx-decision convert -q makes it. It goes into the workspace's models/ folder and shows up in Models when it is done.",
    Body,
    note: "Stop is in the Jobs list.",
    actions: () => [
      {
        label: "Convert",
        primary: true,
        disabled: Boolean(nameProblem.value || blocked.value || starting.value),
        onSelect: start,
      },
    ],
  });
}
