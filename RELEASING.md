# Releasing HilbertBench

The publish steps, in order. Read this once before the first release —
a few of these are irreversible.

## What "release" means (three separate things)

| Step | Makes it… | Reversible? |
|------|-----------|-------------|
| Git tag + GitHub release | a citable code snapshot | the tag, yes; delete and re-tag |
| Zenodo archive | citable with a DOI (for the paper) | no — a DOI is permanent |
| **PyPI upload** | **installable worldwide via `pip`** | **no — a version is immutable** |

A tag alone does **not** put the package on PyPI. The PyPI upload is the
step that makes `pip install hilbertbench` work for anyone, and it is
the one you cannot undo: once `1.0.0` is uploaded you can never replace
it, only release a new version. So everything is verified *before* the
upload.

## Pre-flight (do every time)

```bash
# 1. version in pyproject.toml matches the tag you intend (vX.Y.Z <-> "X.Y.Z")
grep '^version' pyproject.toml

# 2. the whole suite passes
python -m pytest tests/ -q

# 3. the docs build
mkdocs build -f mkdocs.yml --strict
mkdocs build -f mkdocs-rtd.yml --strict

# 4. the test catalog is current
python tools/gen_test_catalog.py   # then check git diff is empty
```

## Build and check the distribution

```bash
pip install --upgrade build twine        # or: pip install -e ".[dev]"
rm -rf dist/ build/ *.egg-info
python -m build                          # makes dist/*.whl and dist/*.tar.gz
twine check dist/*                       # validates metadata + README rendering
```

`twine check` must pass clean — it catches a malformed README or
missing metadata before PyPI does.

## Test the install in a clean environment

This is the step that proves `pip install hilbertbench` actually works
for a stranger. Use a throwaway virtualenv so nothing from the dev
environment leaks in:

```bash
python -m venv /tmp/hb-test && source /tmp/hb-test/bin/activate
pip install dist/hilbertbench-1.0.0-py3-none-any.whl
python -c "
import hilbertbench
from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
from hilbertbench.recorder.tape import HilbertTape
print('import OK, version', __import__('importlib.metadata', fromlist=['version']).version('hilbertbench'))
"
hb-verify --help
deactivate && rm -rf /tmp/hb-test
```

## Upload to TestPyPI first

TestPyPI is a sandbox copy of PyPI. Upload there, install from there,
confirm it works — *then* do the real thing.

```bash
twine upload --repository testpypi dist/*
# install from TestPyPI (real deps still come from real PyPI):
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ hilbertbench
```

You need a TestPyPI account and an API token (pypi.org settings →
API tokens). Store it in `~/.pypirc` or paste at the prompt.

## Upload to PyPI (the irreversible step)

Only after TestPyPI looks right:

```bash
twine upload dist/*
```

The name `hilbertbench` is currently unclaimed on PyPI — the first
successful upload claims it for this project.

## Tag, GitHub release, Zenodo

```bash
git tag -a v1.0.0 -m "HilbertBench 1.0.0"
git push origin v1.0.0
```

Then on GitHub: Releases → draft a release from the `v1.0.0` tag. If the
Zenodo–GitHub integration is enabled (Zenodo account → GitHub → flip the
repository switch on **before** creating the release), publishing the
GitHub release automatically deposits the archive and mints the DOI.
Record that DOI — it is what the paper cites as the instrument version.

## Also freeze the schema

A v1.0.0 release means the trace format is now public. Tag the schema so
INV-005 (frozen schema versions) has something to point at:

```bash
git tag -a schema-v1.0 -m "Freeze trace schema v1.0"
git push origin schema-v1.0
```

After this tag, `schemas/v1.0/` is permanent; format changes go to a new
version directory.

## Version bumping for the next release

After releasing, the convention (semantic versioning):

- **patch** (`1.0.1`) — bug fixes, no API change
- **minor** (`1.1.0`) — backward-compatible additions (new analyzer, new
  integration, new optional schema field in `schemas/v1.1/`)
- **major** (`2.0.0`) — a breaking API or schema change

Update `version` in `pyproject.toml`, and never reuse a version that has
already been uploaded to PyPI.
