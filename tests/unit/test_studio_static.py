"""The Studio's static files: every import in the page resolves to a shipped file."""

import json
import re

from mlx_decision.studio.page import STATIC

IMPORT = re.compile(
    r"""(?:^|[;\s])(?:import|export)\s*(?:[^'";]*?\sfrom\s*)?["']([^"']+)["']""", re.M
)
DYNAMIC = re.compile(r"""import\(\s*["']([^"']+)["']\s*\)""")


def import_map() -> dict[str, str]:
    page = (STATIC / "index.html").read_text()
    [text] = re.findall(r'<script type="importmap">(.*?)</script>', page, re.S)
    return json.loads(text)["imports"]


def specifiers(path):
    text = path.read_text()
    return set(IMPORT.findall(text)) | set(DYNAMIC.findall(text))


def resolve(specifier: str, importer, mapping: dict[str, str]):
    if specifier.startswith("."):
        return (importer.parent / specifier).resolve()
    for name, target in sorted(mapping.items(), key=lambda item: -len(item[0])):
        if specifier == name or (name.endswith("/") and specifier.startswith(name)):
            mapped = target + specifier[len(name) :]
            return STATIC / mapped.removeprefix("__STATIC__/")
    return None


def test_every_import_resolves_to_a_shipped_file():
    mapping = import_map()
    modules = sorted((STATIC / "app").rglob("*.js")) + sorted((STATIC / "vendor").glob("*.js"))
    assert len(modules) > 10
    for module in modules:
        for specifier in specifiers(module):
            target = resolve(specifier, module, mapping)
            assert target is not None, f"{module.name}: {specifier} is not in the import map"
            assert target.is_file(), f"{module.name}: {specifier} -> {target} does not exist"


def test_the_import_map_targets_exist():
    for name, target in import_map().items():
        path = STATIC / target.removeprefix("__STATIC__/")
        assert path.is_dir() if name.endswith("/") else path.is_file(), name


def test_every_app_module_is_reached_from_main():
    mapping = import_map()
    seen, todo = set(), [STATIC / "app" / "main.js"]
    while todo:
        module = todo.pop().resolve()
        if module in seen or STATIC / "vendor" in module.parents:
            continue
        seen.add(module)
        todo += [resolve(s, module, mapping) for s in specifiers(module)]
    shipped = {path.resolve() for path in (STATIC / "app").rglob("*.js")}
    assert shipped <= seen, sorted(str(p.relative_to(STATIC)) for p in shipped - seen)
