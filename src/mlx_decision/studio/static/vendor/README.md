# Vendored JavaScript

The Studio's page imports these ES modules through the import map in
`../index.html`, so it needs no build step and loads nothing from
another host. Each file is the package's own build from the npm
registry, unchanged, except that the two htm files start with the
copyright and licence header of htm's `src/` (the published builds lack
it). The tarballs' SHA-512 matched the registry's `integrity` field.

| File | Package | Version | File in the package | Licence |
|---|---|---|---|---|
| `preact.js` | preact | 10.29.8 | `dist/preact.module.js` | MIT, Copyright (c) 2015-present Jason Miller |
| `preact-hooks.js` | preact | 10.29.8 | `hooks/dist/hooks.module.js` | MIT, as above |
| `htm.js` | htm | 3.1.1 | `dist/htm.module.js` | Apache-2.0, Copyright 2018 Google Inc. |
| `htm-preact.js` | htm | 3.1.1 | `preact/index.module.js` | Apache-2.0, as above |
| `signals.js` | @preact/signals | 2.11.2 | `dist/signals.module.js` | MIT, Copyright (c) 2022-present Preact Team |
| `signals-core.js` | @preact/signals-core | 1.14.4 | `dist/signals-core.module.js` | MIT, as above |

The licence texts are in the repository's `LICENSES/` folder
(`preact-MIT.txt`, `preact-signals-MIT.txt`, `htm-Apache-2.0.txt`) and
in the wheel's licence files.

To update one: download `https://registry.npmjs.org/<package>/-/<name>-<version>.tgz`,
check its SHA-512 against the registry's `integrity`, copy the file named
above over the old one (for htm, with the header again) and change the
table.
