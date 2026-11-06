# NLP1 2026 - PAWS paraphrase detection

Task: binary paraphrase detection on PAWS (`google-research-datasets/paws`, config `labeled_final`).
Used for all three NLP1 projects (P1 classical classifiers -> P2 compositional encoders -> P3 LLMs) and the blog post.

## Setup
```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .   # makes src/paws importable from scripts and notebooks
git clone https://github.com/nlp-illc/nlp1-student external/nlp1-student   # provides the `part1` package
git -C external/nlp1-student checkout 6d93c951fa2bd0a92b6e90ca954865f1c1165eef   # pinned: external/ is gitignored and the course repo may change
```

## P1 pipeline
```bash
python scripts/01_explore_data.py      # dataset stats -> results/p1/data_stats.json
python scripts/02_train_tokenizer.py   # BPE tokenizer -> results/p1/tokenizer.json
python scripts/03_run_cv.py            # random folds: 5 systems + majority baseline -> results/p1/cv/
python scripts/04_analyse.py           # random folds: results table, box plot, Bayesian ROPE comparisons
python scripts/05_run_grouped_grid.py  # primary results: full 3x3 grid on grouped folds -> results/p1_full/cv/
python scripts/06_analyse_grouped_grid.py   # primary results: table, plots, ROPE comparisons, random vs grouped
python scripts/07_check_leakage.py     # official validation check and leakage breakdown (uses the saved models)
```
Scripts 03 and 05 take `--smoke` for a fast check in a separate folder.

## Layout
See `docs/DESIGN.md` for design decisions and what each file is for.

## AI assistance
Logged in `AI_USAGE.md` (required for the blog post).
