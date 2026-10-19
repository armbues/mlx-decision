// The image preview: one of a list of images, large, over the page. Esc, a
// click beside it or the close button ends it; ← and → step through the list.
import { signal } from "@preact/signals";
import { html } from "htm/preact";
import { useEffect, useRef, useState } from "preact/hooks";

// {images: [data URL], index}
export const preview = signal(null);

export function openPreview(images, index) {
  preview.value = { images, index };
}

export function closePreview() {
  preview.value = null;
}

function step(by) {
  const current = preview.value;
  if (!current || current.images.length < 2) return;
  const count = current.images.length;
  preview.value = { ...current, index: (current.index + by + count) % count };
}

const bytes = (n) => (n < 1024 ** 2 ? `${Math.max(1, Math.round(n / 1024))} KB` : `${(n / 1024 ** 2).toFixed(1)} MB`);

// "PNG · 7 KB" from a data URL, without decoding it.
function describe(url) {
  const match = /^data:image\/([A-Za-z0-9.+-]+)[^,]*;base64,/.exec(url);
  if (!match) return "";
  const size = Math.floor(((url.length - match[0].length) * 3) / 4);
  return `${match[1].toUpperCase()} · ${bytes(size)}`;
}

export function Lightbox() {
  const current = preview.value;
  const panel = useRef(null);
  // The shown image's pixel size, once loaded: {url, width, height}.
  const [size, setSize] = useState(null);
  useEffect(() => {
    if (current) panel.current?.focus();
  }, [!!current]);
  if (!current) return null;
  const { images, index } = current;
  const url = images[index];
  const several = images.length > 1;
  const onKeyDown = (event) => {
    if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
      event.preventDefault();
      step(event.key === "ArrowRight" ? 1 : -1);
    }
  };
  return html`
    <div class="lightbox" role="dialog" aria-modal="true" aria-label=${`Image ${index + 1}`}
         tabindex="-1" ref=${panel} onKeyDown=${onKeyDown}
         onClick=${(event) => event.target === event.currentTarget && closePreview()}>
      <figure onClick=${(event) => event.target === event.currentTarget && closePreview()}>
        <img src=${url} alt=${`Image ${index + 1}`}
             onLoad=${(event) => setSize({ url, width: event.target.naturalWidth, height: event.target.naturalHeight })} />
        <figcaption class="num">
          <span>[image ${index + 1}]${several ? ` of ${images.length}` : ""}</span>
          ${size?.url === url && html`<span>${size.width} × ${size.height}</span>`}
          <span>${describe(url)}</span>
        </figcaption>
      </figure>
      ${several &&
      html`<button class="lb-step prev" aria-label="Previous image" onClick=${() => step(-1)}>‹</button>
        <button class="lb-step next" aria-label="Next image" onClick=${() => step(1)}>›</button>`}
      <button class="lb-close" aria-label="Close" title="Close (Esc)" onClick=${closePreview}>×</button>
    </div>
  `;
}
