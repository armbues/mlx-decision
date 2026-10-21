// The page and its item from the address: #/playground/<set>, #/models/<name>,
// #/datasets/<name>, #/batches/<name>, #/training, #/settings.
import { signal } from "@preact/signals";

export const PAGES = ["playground", "models", "datasets", "batches", "training", "settings"];

function parse() {
  const [page, item] = location.hash.replace(/^#\/?/, "").split("/");
  return {
    page: PAGES.includes(page) ? page : "playground",
    item: item ? decodeURIComponent(item) : null,
  };
}

export const route = signal(parse());
addEventListener("hashchange", () => {
  route.value = parse();
});

export function href(page, item = null) {
  return `#/${page}${item ? `/${encodeURIComponent(item)}` : ""}`;
}

export function navigate(page, item = null) {
  location.hash = href(page, item);
}
