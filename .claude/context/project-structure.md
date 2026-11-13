# Project structure

## Directories

- `src/`: all project code, two packages. Installed by `poetry install`, imported as `<package>.<module>` from any notebook, no `sys.path` edits.
  - `core/`: code affecting training and decisions (what gets predicted, submitted). `cli` (one run per fold), datasets, folds, `model` (LightningModule), `metrics`, `predict`.
  - `helpers/`: code only easing project code: plumbing, tracking, display, figures. Paths and constants (`config`), `tracking`, `progress`, `cv`, `submission`, `plots`.
- `configs/`: training configs, `<slug>/<model>.yaml`, one per model.
- `notebooks/`: `experiments/` (one per card), `submissions/`.
- `data/kaggle/`: Kaggle zip as is, read-only. `data/processed/`: generated files. `data/submissions/`: CSVs for upload. All gitignored.
- `docs/`: `task/` (task, rules), `figures/` (all figures). `docs/course/`: course materials, gitignored.
  - `neuroslope/`: all AI-written docs. `methods/` (topic docs), `reports/` (`<topic>_report.md`), `experiments/` (cards), `ai_usage/` (AI chat transcripts), `code notes/`, `backlog.md`, `theory_answers.md`.

## Branches

- `playground`: scratch, cut from `main`, notebooks and configs straight in `notebooks/`, `configs/`. Never merged into `main`.

## Experiment files

Folders created only when first file lands:

- Card: `docs/neuroslope/experiments/<slug>.md`
- Notebook: `notebooks/experiments/<slug>.ipynb`
- Configs: `configs/<slug>/<model>.yaml`: top `name`, `experiment` plus `model`, `data`, `trainer` sections
- Batches: `configs/<slug>/batches/<name>.yaml`: `session`, `gpu`, `configs` list, for `make batch BATCH=...`
- Figures: `docs/figures/<slug>*.png`
