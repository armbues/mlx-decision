// The Studio's icons: small inline SVGs drawn in the text colour.
import { html } from "htm/preact";

const stroke = (width = 1.3) => ({
  fill: "none",
  stroke: "currentColor",
  "stroke-width": width,
  "stroke-linecap": "round",
  "stroke-linejoin": "round",
});

export const Brand = () => html`
  <svg class="brand-icon" viewBox="0 0 26 26" aria-hidden="true">
    <defs>
      <linearGradient id="brand-gradient" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="var(--icon-a)" />
        <stop offset="1" stop-color="var(--icon-b)" />
      </linearGradient>
    </defs>
    <rect x="0.5" y="0.5" width="25" height="25" rx="6.5" fill="url(#brand-gradient)" />
    <rect x="6" y="7.2" width="7" height="2.6" rx="1.3" fill="#ffffff" opacity="0.38" />
    <rect x="6" y="11.7" width="14" height="2.6" rx="1.3" fill="var(--accent)" />
    <rect x="6" y="16.2" width="4.5" height="2.6" rx="1.3" fill="#ffffff" opacity="0.38" />
  </svg>
`;

export const Playground = () => html`
  <svg viewBox="0 0 16 16" ...${stroke()} aria-hidden="true">
    <path d="M2.5 4.5h7M2.5 8h5M2.5 11.5h6" />
    <circle cx="12" cy="4.5" r="1.5" /><circle cx="10" cy="8" r="1.5" /><circle cx="11" cy="11.5" r="1.5" />
  </svg>
`;

export const Models = () => html`
  <svg viewBox="0 0 16 16" ...${stroke()} aria-hidden="true">
    <path d="M8 1.8 13.5 5v6L8 14.2 2.5 11V5z" /><path d="M2.5 5 8 8.2 13.5 5M8 8.2v6" />
  </svg>
`;

// A table of rows: the Datasets page.
export const Datasets = () => html`
  <svg viewBox="0 0 16 16" ...${stroke()} aria-hidden="true">
    <rect x="2.5" y="2.5" width="11" height="11" rx="2" /><path d="M2.5 6.2h11M2.5 9.8h11M6.2 6.2v7.3" />
  </svg>
`;

export const Training = () => html`
  <svg viewBox="0 0 16 16" ...${stroke()} aria-hidden="true">
    <path d="M2 13.5h12" /><path d="M3 10.5 6.5 7l2.5 2.5L13.5 4" /><path d="M10.5 4h3v3" />
  </svg>
`;

export const Settings = () => html`
  <svg viewBox="0 0 16 16" ...${stroke()} aria-hidden="true">
    <path d="M2.5 4.5h6M11.5 4.5h2M2.5 11.5h2M7.5 11.5h6" /><circle cx="10" cy="4.5" r="1.6" /><circle cx="6" cy="11.5" r="1.6" />
  </svg>
`;

export const Batches = () => html`
  <svg viewBox="0 0 16 16" ...${stroke()} aria-hidden="true">
    <path d="M2 3.5h7.5M2 8h7.5M2 12.5h7.5" /><path d="M11.8 5.6 14.6 8l-2.8 2.4z" />
  </svg>
`;

export const Download = () => html`
  <svg viewBox="0 0 16 16" ...${stroke(1.4)} aria-hidden="true">
    <circle cx="8" cy="8" r="6.5" /><path d="M8 4.5v6M5.5 8.2 8 10.7l2.5-2.5" />
  </svg>
`;

export const Close = () => html`
  <svg viewBox="0 0 10 10" ...${stroke(1.4)} aria-hidden="true"><path d="M2.5 2.5l5 5M7.5 2.5l-5 5" /></svg>
`;

export const File = ({ className = "file" }) => html`
  <svg class=${className} viewBox="0 0 14 16" ...${stroke(1.2)} aria-hidden="true">
    <path d="M2.5 1.5h6l3 3v10h-9z" /><path d="M8.5 1.5v3h3" />
  </svg>
`;

// A picture: the model reads images (the Models page's Reads column).
export const Picture = () => html`
  <svg class="picture" viewBox="0 0 16 16" ...${stroke(1.2)} role="img" aria-label="Reads images">
    <title>Reads images</title>
    <rect x="2" y="3" width="12" height="10" rx="2" />
    <circle cx="6" cy="6.6" r="1.2" />
    <path d="M2.6 11.6l3.4-3.1 2.5 2.3 2-1.7 2.9 2.5" />
  </svg>
`;

export const Plus = () => html`
  <svg viewBox="0 0 12 12" width="11" height="11" ...${stroke(1.5)} aria-hidden="true">
    <path d="M6 1.5v9M1.5 6h9" />
  </svg>
`;

// Two ticked lines: New Evaluation (scoring answers against their labels).
export const Checklist = () => html`
  <svg viewBox="0 0 16 16" ...${stroke(1.6)} aria-hidden="true">
    <path d="m1.8 4.4 1.6 1.6 3-3.4M1.8 10.4l1.6 1.6 3-3.4M8.8 4.6h5.4M8.8 10.6h5.4" />
  </svg>
`;

export const Trash = () => html`
  <svg viewBox="0 0 14 14" width="13" height="13" ...${stroke(1.2)} aria-hidden="true">
    <path d="M2.5 3.8h9M5.6 3.8V2.6h2.8v1.2M3.6 3.8l.6 7.6c0 .5.4.9.9.9h3.8c.5 0 .9-.4.9-.9l.6-7.6M5.9 6v4.2M8.1 6v4.2" />
  </svg>
`;

export const Play = () => html`
  <svg viewBox="0 0 12 12" fill="currentColor" aria-hidden="true"><path d="M2.5 1.3v9.4L10.2 6z" /></svg>
`;

export const Stop = () => html`
  <svg viewBox="0 0 12 12" fill="currentColor" aria-hidden="true">
    <rect x="2" y="2" width="8" height="8" rx="1.5" />
  </svg>
`;

export const SidebarToggle = () => html`
  <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true">
    <rect x="2.5" y="3.5" width="15" height="13" rx="2.5" /><path d="M7.5 3.5v13" />
    <path d="M4.5 6.5h1.2M4.5 8.5h1.2" stroke-linecap="round" />
  </svg>
`;

export const InspectorToggle = () => html`
  <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true">
    <rect x="2.5" y="3.5" width="15" height="13" rx="2.5" /><path d="M12.5 3.5v13" />
    <path d="M14.3 6.5h1.2M14.3 8.5h1.2" stroke-linecap="round" />
  </svg>
`;

export const Code = () => html`
  <svg viewBox="0 0 16 16" ...${stroke(1.5)} aria-hidden="true"><path d="m5.5 4-4 4 4 4M10.5 4l4 4-4 4" /></svg>
`;

export const Theme = () => html`
  <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true">
    <circle cx="10" cy="10" r="6.5" /><path d="M10 3.5a6.5 6.5 0 0 1 0 13z" fill="currentColor" />
  </svg>
`;

export const Crumb = () => html`
  <svg class="crumb" viewBox="0 0 7 11" ...${stroke(1.4)} aria-hidden="true"><path d="m1.5 1.5 4 4-4 4" /></svg>
`;

export const Chevrons = () => html`
  <svg class="chev" viewBox="0 0 9 13" ...${stroke(1.4)} aria-hidden="true">
    <path d="M1.5 4.5 4.5 1.5l3 3M1.5 8.5l3 3 3-3" />
  </svg>
`;

export const Tick = () => html`
  <svg class="tick" viewBox="0 0 12 12" ...${stroke(1.8)} aria-hidden="true"><path d="m2.5 6.5 2.3 2.3 4.7-5.3" /></svg>
`;

// Status field and banner icons: ok, warn, info, spin.
export const StatusIcon = ({ kind }) => {
  const className = `ic ${kind}`;
  if (kind === "ok")
    return html`<svg class=${className} viewBox="0 0 14 14" aria-hidden="true">
      <circle cx="7" cy="7" r="6.5" fill="currentColor" />
      <path d="m4.2 7.2 1.9 1.9 3.8-4.2" fill="none" stroke="#fff" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />
    </svg>`;
  if (kind === "warn")
    return html`<svg class=${className} viewBox="0 0 14 14" aria-hidden="true">
      <path d="M7 1.2 13.2 12.3H.8z" fill="currentColor" />
      <path d="M7 5.2v3.3" stroke="#fff" stroke-width="1.4" stroke-linecap="round" />
      <circle cx="7" cy="10.4" r="0.8" fill="#fff" />
    </svg>`;
  if (kind === "spin")
    return html`<svg class=${className} viewBox="0 0 14 14" aria-hidden="true">
      <path d="M7 1.5A5.5 5.5 0 1 1 1.5 7" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" />
    </svg>`;
  return html`<svg class=${className} viewBox="0 0 14 14" aria-hidden="true">
    <circle cx="7" cy="7" r="6" fill="none" stroke="currentColor" stroke-width="1.2" />
    <path d="M7 6.2v3.6" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" />
    <circle cx="7" cy="4.3" r="0.8" fill="currentColor" />
  </svg>`;
};
