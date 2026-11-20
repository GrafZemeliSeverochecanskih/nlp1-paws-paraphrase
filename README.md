# NLP1 2026 - PAWS paraphrase detection

Task: binary paraphrase detection on PAWS (`google-research-datasets/paws`, config `labeled_final`).
Used for all three NLP1 projects (P1 classical classifiers -> P2 compositional encoders -> P3 LLMs) and the blog post.

## Setup
```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .   # makes src/paws importable from scripts and notebooks
git clone https://github.com/nlp-illc/nlp1-student external/nlp1-student   # provides the `part1` and `part2` packages
git -C external/nlp1-student checkout b2c2394   # pinned (T2: part1 + part2): external/ is gitignored and the course repo may change
# P2 only: GloVe 2024 50d (kept zipped; ~300 MB)
curl -L -o data/glove.2024.wikigiga.50d.zip https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.50d.zip
# P2 embedding comparison only: GloVe 2014 6B (862 MB; part2 reads single-file zips, so extract the .txt files)
curl -L -o data/glove6B/glove.6B.zip --create-dirs https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip
python -c "import zipfile; zipfile.ZipFile('data/glove6B/glove.6B.zip').extractall('data/glove6B')"
# optional, NVIDIA GPU: replace the CPU torch with a CUDA build
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
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

## P2 pipeline
```bash
python scripts/08_embedding_coverage.py   # GloVe coverage -> results/p2/coverage.json
python scripts/09_run_p2_grid.py --smoke  # fast check in results/smoke_p2
python scripts/09_run_p2_grid.py          # pooling / LSTM, siamese / cross-encoding on grouped folds -> results/p2/cv/
python scripts/10_analyse_p2.py           # table, box plot, ROPE comparisons (incl. best P1 system)
python scripts/11_error_analysis.py pool+cross lstm+siamese # Bayesian feature analysis -> results/p2/bda/
python scripts/12_official_validation.py lstm+siamese       # saved fold models on the official validation split
python scripts/09_run_p2_grid.py --seeds 0 1 2 --only lstm+siamese   # 3 CV runs -> results/p2_seeds0-1-2/
python scripts/09_run_p2_grid.py --subsample 5000 --only lstm+siamese # one learning-curve point -> results/p2_n5000/
python scripts/13_learning_curve.py                         # accuracy vs subsample size -> results/p2/learning_curve.png
python scripts/09_run_p2_grid.py --glove 6B-100d --only pool+siamese bilstm+siamese   # other embeddings -> results/p2_glove6B-100d/ (also 08 --glove)
```
Device is `cuda` when available, otherwise `cpu` (`P2Config.device`).

## Layout
See `docs/DESIGN.md` for design decisions and what each file is for.

## AI assistance
Logged in `AI_USAGE.md` (required for the blog post).
