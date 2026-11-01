// Family marks and colours (the Studio's design brief), and how a loaded group is named
// and coloured wherever memory is shown (the Models page, the sidebar).
import { models } from "./store/status.js";

const FAMILIES = {
  clef: ["Cf", "#e8730c"],
  pplx: ["px", "#1f8a8a"],
  d3: ["d3", "#5b5bd6"],
  baselabs: ["BL", "#8a5a33"],
  h2o: ["H2", "#d4a20b"],
  d1lfm3: ["dL", "#1d5c9e"],
  d1: ["d1", "#2f7dd1"],
  laya: ["La", "#2e9d5b"],
  julia: ["Ju", "#c2417a"],
};
const OTHER = "#8e8e93";

// [two-letter mark, colour]; other families get their first letters in grey.
export const markOf = (family) => FAMILIES[family] ?? [(family ?? "").slice(0, 2), OTHER];

// A group by the models a request named; a base loaded only for its fits says so.
export const groupLabel = (group) =>
  group.members.includes(group.root) ? group.members.join(", ") : `${group.members.join(", ")} (with ${group.root})`;

// The listed model a group stands for (its root, else its first member).
function modelOf(group) {
  const list = models.value ?? [];
  return list.find((model) => model.name === group.root) ?? list.find((model) => model.name === group.members[0]);
}

// Shades for the second, third, ... loaded group of one family: darker and
// lighter in turn, all dark enough for white labels on them.
const SHADES = ["", "black 30%", "white 28%", "black 50%"];

// A group's colour among ``groups`` (the loaded ones): its family's colour,
// shaded by its place among that family's loaded groups in the list's order
// (not by use, so a model keeps its colour while it stays loaded).
export function groupColour(group, groups = [group]) {
  const list = models.value ?? [];
  const position = (g) => list.indexOf(modelOf(g));
  const family = modelOf(group)?.family;
  const same = groups.filter((other) => modelOf(other)?.family === family).sort((a, b) => position(a) - position(b));
  const colour = markOf(family)[1];
  const shade = SHADES[Math.max(0, same.findIndex((other) => other.root === group.root)) % SHADES.length];
  return shade ? `color-mix(in srgb, ${colour}, ${shade})` : colour;
}
