// Models: what the Studio serves and what each one reads.
import { computed } from "@preact/signals";
import { html } from "htm/preact";
import { plural } from "../format.js";
import { href, route } from "../router.js";
import { loadedNames, models } from "../store/status.js";

const selected = computed(() => {
  const name = route.value.page === "models" ? route.value.item : null;
  return (models.value ?? []).find((model) => model.name === name) ?? null;
});

const status = computed(() => {
  const list = models.value;
  if (!list) return null;
  if (list.length === 0)
    return { icon: "info", text: html`No models yet · <code>mlx-decision download</code>` };
  return {
    icon: "ok",
    text: html`<b>${plural(list.length, "model")}</b> · ${loadedNames.value.size} loaded`,
  };
});

const action = computed(() => ({ label: "Download Model…", icon: false, disabled: true }));

function Centre() {
  const list = models.value;
  if (!list) return html`<div class="centre-inner"></div>`;
  return html`
    <div class="centre-inner">
      <div class="pane-head"><div><h1>Models</h1></div></div>
      ${list.length === 0
        ? html`<div class="empty">
            <b>No models yet</b>
            <span>Download one with <code>mlx-decision download</code>; it appears here.</span>
          </div>`
        : html`<div class="group">
            ${list.map(
              (model) => html`
                <a class="row model-row" key=${model.name} href=${href("models", model.name)}
                   aria-current=${selected.value?.name === model.name ? "true" : undefined}>
                  <span class="mono">${model.name}</span>
                  <span class="model-meta">
                    <span>${model.family}${model.fitted ? ` · fitted on ${model.base}` : ""}</span>
                    ${loadedNames.value.has(model.name) && html`<span class="pill loaded">Loaded</span>`}
                  </span>
                </a>
              `,
            )}
          </div>`}
    </div>
  `;
}

const yesNo = (value) => (value ? "yes" : "no");

function Details() {
  const model = selected.value;
  if (!model)
    return html`<div class="empty"><b>No model selected</b><span>Pick one to see what it reads.</span></div>`;
  const caps = model.capabilities;
  return html`
    <div class="details">
      <h3 class="mono">${model.name}</h3>
      ${model.error && html`<p class="warn-text">${model.error}</p>`}
      ${caps &&
      html`<dl>
        <dt>Family</dt><dd>${model.family}</dd>
        <dt>Question types</dt><dd>${caps.question_types.join(", ")}</dd>
        <dt>Input limit</dt>
        <dd>${caps.max_input_tokens ? `${caps.max_input_tokens.toLocaleString("en")} tokens` : "none"}${
          caps.max_input_tokens ? (caps.truncates_input ? ", longer states are cut" : ", longer is refused") : ""
        }</dd>
        <dt>Choice options</dt><dd>${caps.max_choice_options ?? "no limit"}</dd>
        <dt>Score levels</dt><dd>${caps.max_score_levels ?? "no limit"}</dd>
        <dt>Instructions</dt><dd>${caps.requires_instructions ? "required" : "optional"}</dd>
        <dt>Images</dt><dd>${yesNo(caps.supports_images)}</dd>
      </dl>`}
      <p class="note">In the API: <code>"model": "${model.name}"</code> · CLI: <code>-m ${model.name}</code></p>
    </div>
  `;
}

export default {
  id: "models",
  title: "Models",
  action,
  status,
  Centre,
  tabs: [{ id: "details", label: "Model", Body: Details }],
};
