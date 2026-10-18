// The Studio's page: for now a shell that shows what it is serving.
import { render } from "preact";
import { html } from "htm/preact";
import { signal } from "@preact/signals";

const status = signal(null);
const problem = signal(null);

async function load() {
  try {
    const response = await fetch("/studio/api/status", { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`status ${response.status}`);
    status.value = await response.json();
  } catch (error) {
    problem.value = String(error);
  }
}

function Shell() {
  const current = status.value;
  return html`
    <main class="placeholder">
      <h1>mlx-decision Studio</h1>
      ${problem.value && html`<p class="problem">${problem.value}</p>`}
      ${current &&
      html`<p>
          Version ${current.version} · ${current.models.length} models · API at
          <code>${current.api.url}</code>
        </p>`}
    </main>
  `;
}

render(html`<${Shell} />`, document.getElementById("studio"));
load();
