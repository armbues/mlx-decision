// The window: sidebar, toolbar, centre and inspector around the current page.
import { html } from "htm/preact";
import { useEffect } from "preact/hooks";
import { PAGES } from "../pages/index.js";
import { route } from "../router.js";
import { inspectorShown, narrow, sidebarShown } from "../store/ui.js";
import { BusyBanner } from "./banner.js";
import { Inspector } from "./inspector.js";
import { Lightbox } from "./lightbox.js";
import { Sheet } from "./sheet.js";
import { Sidebar } from "./sidebar.js";
import { Toasts } from "./toasts.js";
import { Toolbar } from "./toolbar.js";

export function App() {
  const current = route.value.page;
  const page = PAGES[current];
  useEffect(() => {
    document.title = `${page.title} · mlx-decision Studio`;
  }, [page]);
  const classes = [
    "app",
    `page-${page.id}`,
    sidebarShown.value ? "" : "no-sidebar",
    narrow.value ? "narrow" : "",
    inspectorShown.value ? "inspector-open" : "no-inspector",
    page.wide?.value ? "compare" : "", // a wider inspector, for a table of models
  ];
  const Centre = page.Centre;
  return html`
    <div class=${classes.filter(Boolean).join(" ")}>
      <${Sidebar} page=${page} current=${current} />
      <${Toolbar} page=${page} />
      <main class="centre">
        <div class="banner-wrap"><${BusyBanner} page=${page.id} /></div>
        <${Centre} />
      </main>
      <${Inspector} page=${page} />
    </div>
    <${Sheet} />
    <${Lightbox} />
    <${Toasts} />
  `;
}
