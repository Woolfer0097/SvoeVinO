# Repository Guidelines

## Project Structure & Data

This repository contains two independent Python 3.11+ packages under `modules/`. `dinov2_retrieval` implements visual wine search and its API; `wine_label_preprocessing` handles image crops, photometry, cylindrical unwrapping, and benchmarking. Each package has its own `pyproject.toml`, `src/`, `tests/`, and README. Keep module dependencies one-way: the packages do not import each other. Design notes are in `docs/superpowers/{specs,plans}/`.

`Датасет/` is a data bundle, not application source. It includes extracted images, archives, and evaluation inputs. `queries.tsv` has `query_id` and `image_path` columns (three entries); `strapi_output0709.csv` is a UTF-8 product export with 4,147 records, with an identical copy under `Датасет/Датасет/`. The participant instructions are in `Датасет/README.md`. Avoid broad edits to or generated outputs in this directory when changing package code.

## Build, Test & Development

Run commands inside the relevant module directory; each package uses its own virtual environment.

```bash
# preprocessing
cd modules/wine_label_preprocessing
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q

# retrieval
cd modules/dinov2_retrieval
python -m venv .venv && .venv/bin/pip install -e '.[dev,db,api]'
.venv/bin/pytest -q
```

Retrieval integration tests need PostgreSQL and DINOv2 dependencies (`.[ml]`) as well as weights; run them explicitly with `pytest -m integration -q`. For the local API/database workflow, use the module's `docker compose` configuration and follow its README setup for `.env`, data, and model cache.

## Coding Style & Testing

Use four-space indentation, Python `snake_case` for modules/functions, `PascalCase` for classes, and `test_*.py` for pytest files. Match nearby typing and docstring conventions. No formatter or linter is configured. Add focused tests beside the affected package's existing tests; keep tests independent of network, GPU, and external services unless marked integration.

## Commits & Pull Requests

History uses short descriptive subjects, sometimes in English and sometimes Russian; no strict prefix convention is established. Keep commits focused. Pull requests should describe the affected module and behavior, list relevant test commands/results, note configuration or data prerequisites, and update that module's README when usage changes. Include screenshots only for visible API or documentation changes.

## Configuration

Keep secrets in local environment files and out of commits. Retrieval's `.env`, reference data, and model cache are runtime inputs; consult the module README before changing their paths or Docker mounts.
