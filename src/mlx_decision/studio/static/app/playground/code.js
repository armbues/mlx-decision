// Get Code (PG-8): the selected example's request as curl, Jev SDK (Python),
// `mlx-decision run` and an MCP client's config, with the Studio's real
// address and the CLI's real path, so the page teaches the API and the CLI.
import { signal } from "@preact/signals";
import { html } from "htm/preact";
import * as icons from "../components/icons.js";
import { openSheet } from "../components/sheet.js";
import { plural, shellWord } from "../format.js";
import { remembered } from "../store/prefs.js";
import { models, status } from "../store/status.js";
import {
  datasetOfDocument, built, draft, edited, example, modelName, setName } from "./session.js";

const TABS = [
  ["curl", "curl"],
  ["python", "Python"],
  ["cli", "Command Line"],
  ["mcp", "MCP Client"],
];
const tab = remembered("playground.codeTab", "curl");
const copied = signal(false);

const KEY_VARIABLE = "MLX_DECISION_API_KEY";
const SDK_TYPES = { choice: "Choice", score: "Score", noul: "Noul" };

// An image as the sheet shows it: the start of its data URL and its size.
// Copy takes the whole one.
function shortImage(url) {
  if (typeof url !== "string" || !url.startsWith("data:") || url.length < 120) return url;
  const kilobytes = Math.round((url.length * 3) / 4 / 1024);
  return `${url.slice(0, url.indexOf(",") + 13)}…(${kilobytes} KB)`;
}

// A value as Python source: JSON's strings are Python's; short values on one line.
function py(value, indent = 0) {
  if (value === null || value === undefined) return "None";
  if (value === true) return "True";
  if (value === false) return "False";
  if (typeof value !== "object") return JSON.stringify(value);
  const pad = " ".repeat(indent);
  const entries = Array.isArray(value)
    ? value.map((item) => py(item, indent + 4))
    : Object.entries(value).map(([key, item]) => `${JSON.stringify(key)}: ${py(item, indent + 4)}`);
  const [open, close] = Array.isArray(value) ? ["[", "]"] : ["{", "}"];
  if (!entries.length) return open + close;
  const flat = `${open}${entries.join(", ")}${close}`;
  if (flat.length + indent < 76 && !flat.includes("\n")) return flat;
  return `${open}\n${entries.map((entry) => `${pad}    ${entry},`).join("\n")}\n${pad}${close}`;
}

// One question as the SDK's class (a dict for a type it has no class for).
function sdkQuestion(question, indent) {
  const type = SDK_TYPES[question.type];
  if (!type) return { code: py(question, indent), type: null };
  const pad = " ".repeat(indent + 4);
  const args = [];
  if (question.instructions != null) args.push(`instructions=${py(question.instructions, indent + 4)}`);
  if (question.criteria != null) args.push(`criteria=${py(question.criteria, indent + 4)}`);
  const flat = `${type}(${args.join(", ")})`;
  if (flat.length + indent < 76 && !flat.includes("\n")) return { code: flat, type };
  return { code: `${type}(\n${args.map((arg) => `${pad}${arg},`).join("\n")}\n${" ".repeat(indent)})`, type };
}

// What the snippets are built from; null when there is nothing to show.
function context() {
  const current = built.value;
  const name = modelName.value;
  const served = status.value;
  if (!current || !name || !served) return null;
  const model = (models.value ?? []).find((item) => item.name === name);
  const examples = draft.value.examples;
  const index = examples.findIndex((item) => item.uid === draft.value.current);
  const dataset = datasetOfDocument(setName.value);
  return {
    body: current.body,
    problems: current.problems,
    model: name,
    cli: model?.cli ?? { short: ["-m", name], full: ["-m", name] },
    address: served.address,
    api: served.api?.url ?? `${served.address}/v1/systemone`,
    keyRequired: !!served.api?.key_required,
    cliPath: served.cli_path ?? "mlx-decision",
    file: dataset
      ? `${served.data_folder}/${dataset}/dataset.json` // its questions; the state given inline
      : `${served.sets_folder}/${setName.value}.json`,
    set: setName.value,
    dataset,
    example: example.value,
    first: index <= 0 && !dataset,
    firstImages: examples[0]?.images ?? [],
    many: examples.length > 1,
  };
}

// The request with the model first; images shortened unless `whole`.
function request(c, whole) {
  const body = { model: c.model, ...c.body };
  if (!whole && body.images) body.images = body.images.map(shortImage);
  return body;
}

const SNIPPETS = {
  curl(c, whole) {
    const lines = [`curl ${c.api}`, `  -H "Content-Type: application/json"`];
    if (c.keyRequired) lines.push(`  -H "Authorization: Bearer $${KEY_VARIABLE}"`);
    lines.push(`  --data-binary @- <<'JSON'`);
    return `${lines.join(" \\\n")}\n${JSON.stringify(request(c, whole), null, 2)}\nJSON`;
  },

  python(c, whole) {
    const body = request(c, whole);
    const types = new Set();
    const questions = Object.entries(body.questions).map(([id, question]) => {
      const { code, type } = sdkQuestion(question, 8);
      if (type) types.add(type);
      return `        ${JSON.stringify(id)}: ${code},`;
    });
    const imports = [...[...types].sort(), "TypeSafeClient"].join(", ");
    const head = c.keyRequired
      ? `import os\n\nfrom typesafe_sdk import ${imports}\n\nclient = TypeSafeClient(base_url=${JSON.stringify(c.address)}, api_key=os.environ[${JSON.stringify(KEY_VARIABLE)}])`
      : `from typesafe_sdk import ${imports}\n\n# The SDK needs a key; this Studio asks for none.\nclient = TypeSafeClient(base_url=${JSON.stringify(c.address)}, api_key="unused")`;
    const args = [`    model=${JSON.stringify(c.model)},`, `    state=${py(body.state, 4)},`, `    questions={\n${questions.join("\n")}\n    },`];
    if (body.images?.length) args.push(`    extra_body={"images": ${py(body.images, 4)}},`);
    return `${head}\n\nresult = client.system_one(\n${args.join("\n")}\n)\nprint(result)`;
  },

  cli(c, whole) {
    const command = ["mlx-decision", "run", ...c.cli.short, "-q", c.file].map(shellWord).join(" ");
    if (c.first) return command;
    // Another example: its state from a flag; with images (its own or the
    // file's first example's), the whole request as a --states line.
    const images = c.body.images ?? [];
    if (images.length || c.firstImages.length) {
      const line = { state: c.body.state, images: whole ? images : images.map(shortImage) };
      return `${command} --states - <<'JSON'\n${JSON.stringify(line)}\nJSON`;
    }
    const state = c.body.state;
    if (typeof state === "string") return `${command} \\\n  -s ${shellWord(state)}`;
    return `${command} \\\n  --state-json -s ${shellWord(JSON.stringify(state))}`;
  },

  mcp(c) {
    const config = { mcpServers: { decision: { command: c.cliPath, args: ["mcp", ...c.cli.full] } } };
    return JSON.stringify(config, null, 2).replace(/\[\n\s+([^\]]*?)\n\s+\]/s, (_, items) => `[${items.split(/,\n\s+/).join(", ")}]`);
  },
};

function Note({ c }) {
  const which = tab.value;
  if (which === "curl")
    return html`Calls the Studio's Jev-compatible API at <code>${c.address}</code> while it runs; any Jev client works against it.${
      c.keyRequired ? html` The Studio asks for its API key: set <code>${KEY_VARIABLE}</code> to it first.` : ""
    }`;
  if (which === "python")
    return html`The Jev Python SDK (<code>pip install typesafe-sdk</code>) against this Studio's API; <code>model=</code> picks the model.`;
  if (which === "cli" && c.dataset)
    return html`Reads the questions from <code>data/${c.dataset}/dataset.json</code>; the state of the row <b>${c.example.name}</b> is given here.${
      edited.value ? html` <span class="warn-text">The file does not have your edits yet: save first.</span>` : ""
    }${" "}The command is in the Studio's environment: <code>${c.cliPath}</code>.`;
  if (which === "cli")
    return html`${
      c.first
        ? html`Reads <code>sets/${c.set}.json</code>, its state and its questions.`
        : html`Reads the questions from <code>sets/${c.set}.json</code>; the state of <b>${c.example.name}</b> is given here${c.body.images?.length || c.firstImages.length ? ", with its images, and the answer is printed as JSON" : ""}.`
    }${edited.value ? html` <span class="warn-text">The file does not have your edits yet: save first.</span>` : ""}${" "}The
      command is in the Studio's environment: <code>${c.cliPath}</code>.`;
  const words = [c.cliPath, "mcp", ...c.cli.full].map(shellWord).join(" ");
  return html`For Claude Desktop's <code>claude_desktop_config.json</code> (needs the <code>mcp</code> extra); it runs its own
    copy of the model. Claude Code:${" "}<code>claude mcp add decision -- ${words}</code>`;
}

function CodeBody() {
  const c = context();
  if (!c) return html`<span class="dim">Pick a question set and a model first.</span>`;
  const shown = SNIPPETS[tab.value](c, false);
  const copy = () => {
    const done = () => {
      copied.value = true;
      setTimeout(() => (copied.value = false), 1200);
    };
    navigator.clipboard?.writeText(SNIPPETS[tab.value](c, true)).then(done, () => {});
  };
  return html`
    <div class="snip-bar">
      <div class="seg small" role="tablist" aria-label="Snippet">
        ${TABS.map(
          ([id, label]) => html`<button role="tab" key=${id} aria-pressed=${tab.value === id}
            onClick=${() => (tab.value = id)}>${label}</button>`,
        )}
      </div>
      <button class="btn copy" onClick=${copy}>${copied.value ? "Copied" : "Copy"}</button>
    </div>
    ${c.problems.length > 0 && html`<p class="note warn-text">The request has a problem: ${c.problems[0].message}</p>`}
    <pre class="code" data-snippet=${tab.value}>${shown}</pre>
    <p class="note snip-note"><${Note} c=${c} /></p>
  `;
}

export function openCode() {
  const c = context();
  const count = c ? plural(Object.keys(c.body.questions).length, "question") : "";
  openSheet({
    title: "Get Code",
    purpose: c
      ? `${c.dataset ?? c.set} on ${c.model}: ${count} about ${c.many ? c.example.name : "the state"}.`
      : "",
    Body: CodeBody,
    done: "Done",
    wide: true,
  });
}

export function CodeButton() {
  const ready = !!built.value && !!modelName.value;
  return html`
    <button class="btn tool get-code" disabled=${!ready} title="Use this request in code or another tool" onClick=${openCode}>
      <${icons.Code} /><span class="lbl">Get Code</span>
    </button>
  `;
}
