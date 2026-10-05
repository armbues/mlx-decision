# Releasing

How a version of mlx-decision gets to PyPI. Commands run from the
repository root in an environment set up with `make setup` (it installs
`build` and `twine`). The version lives in one place:
`__version__` in `src/mlx_decision/__init__.py`.

## Once: accounts and tokens

PyPI and TestPyPI are separate sites with separate accounts.

1. Create an account on [pypi.org](https://pypi.org) and on
   [test.pypi.org](https://test.pypi.org), verify the email address and
   turn on two-factor authentication (required for uploads).
2. On each site, create an API token under Account settings -> API tokens.
   Before the project exists, the token has to cover the whole account;
   after the first upload, replace it with a token limited to
   `mlx-decision` and delete the account-wide one.
3. Put the tokens into `~/.pypirc` (and `chmod 600 ~/.pypirc`):

   ```ini
   [distutils]
   index-servers =
       pypi
       testpypi

   [pypi]
   username = __token__
   password = pypi-...

   [testpypi]
   repository = https://test.pypi.org/legacy/
   username = __token__
   password = pypi-...
   ```

## 1. Prepare

- Everything is committed and pushed; `make lint` passes.
- `make test` passes with the weights, so the parity tests run:
  `MLX_DECISION_MODELS=/path/to/models make test` (a folder holding
  `clef-flash`).
- `CHANGELOG.md`: the new version's heading gets today's date
  (`## [0.3.0] - 2026-10-07`), and `__version__` is the version to release.
  Commit both.

## 2. Dry run on TestPyPI

A version number can be uploaded to each site only once, even after it is
deleted, so the dry run uses a release candidate.

1. Set `__version__ = "0.3.0rc1"` in `src/mlx_decision/__init__.py`; do not
   commit it.
2. `make publish-test` builds, checks the package and asks before
   uploading. Run it in your own shell: `conda run` does not pass the
   question through.
3. Install it into a fresh environment: only the package itself from
   TestPyPI, its dependencies from PyPI (anyone can upload packages under
   the dependencies' names to TestPyPI, so it is not used as an index for
   them):

   ```bash
   python3.12 -m venv /tmp/mlx-decision-rc
   /tmp/mlx-decision-rc/bin/pip download --no-deps -d /tmp/mlx-decision-rc/dist \
     --index-url https://test.pypi.org/simple/ "mlx-decision==0.3.0rc1"
   /tmp/mlx-decision-rc/bin/pip install \
     "$(ls /tmp/mlx-decision-rc/dist/*.whl)[server,images]"
   ```

4. Check it: `/tmp/mlx-decision-rc/bin/mlx-decision --version`, then the
   README's `run` example with `-m` pointing at a local model folder (or the
   repo id, which downloads it). Look at the project page on
   test.pypi.org: description, links, licence.
5. Undo the version change: `git checkout src/mlx_decision/__init__.py`.
   If something needed fixing, fix and commit it, and repeat with `rc2`.

## 3. Release

1. Make the GitHub repository public, if it is not yet: the PyPI page
   links to it.
2. `make publish` builds the committed version, refuses to run with
   uncommitted changes, and asks before uploading. An upload cannot be
   taken back.
3. Check [pypi.org/project/mlx-decision](https://pypi.org/project/mlx-decision/)
   and install it into a fresh environment with `pip install mlx-decision`.
4. Tag the release and push the tag:

   ```bash
   git tag -a v0.3.0 -m "mlx-decision 0.3.0"
   git push origin v0.3.0
   ```

5. Optionally, a GitHub release with the changelog entry as its notes:
   `gh release create v0.3.0 --title "mlx-decision 0.3.0" --notes "..."`.

## 4. After

- Replace account-wide tokens with project-scoped ones (see above).
- Set `__version__` to the next development version (for example
  `0.4.0.dev0`) and add an `## [Unreleased]` section to `CHANGELOG.md`.
