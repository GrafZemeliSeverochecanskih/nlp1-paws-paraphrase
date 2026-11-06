# Design notes

This repo is my work for NLP1 (UvA MSc AI, 2026). I am working alone.
Task: binary paraphrase detection on PAWS (`google-research-datasets/paws`, config `labeled_final`;
columns `id`, `sentence1`, `sentence2`, `label` with 1 = paraphrase). The same task is used for P1, P2, P3 and a blog post.

The course provides tutorial code in `external/nlp1-student/part1/` (clone of `nlp-illc/nlp1-student`).
Code under `external/` is never edited; it is reused through imports.

## Notes on the tutorial code (part1)
- Everything works on lists of `(text: str, label: int)` tuples.
- `part1.clslib.Encoder`: `__len__` = dimensionality, `__call__(text) -> 1-D np.array`.
  Note: the tutorial's `TokenIndicator.__len__` refers to a global `tokenizer`, not `self.tokenizer` (bug, don't copy).
- `part1.experiment.train_tokenizer(corpus, vocab_size, min_frequency)` -> HuggingFace `tokenizers.Tokenizer` (BPE, Metaspace pretokenizer).
- `part1.experiment.cross_validation_split(pairs, k, seed)` -> `{fold: {"training", "validation", "training_ids", "validation_ids"}}`.
- `part1.experiment.cross_validation_experiment(data_splits, outcome_space, encoder, model_option, save_to, loglin_args, smlp_args)`
  with `model_option in {"logistic", "mlp"}` -> `{"models", "predictions", "accuracies"}` (OrderedDicts keyed by fold).
- `part1.loglin.fit / predict` and `part1.smlp.train_neural_network_model` call `encoder(text)` per example and stack with `csr_matrix(list_of_dense_vectors)`.
- `part1.smlp.SparseMLP(input_dim, hidden_dim, num_classes, dropout)` is a one-hidden-layer MLP over sparse input; helpers `SparseDataset`, `sparse_collate`, `train_one_epoch`, `evaluate` are reusable.
- `part1.clslib.NamedClasses([...])` is the outcome space.
- `part1.analysis.BenavoliCorrelatedT(scores_a, scores_b, runs=..., names=[...])` -> `.probs(rope)`, `.plot(rope)`, `.effect_size`, `.mean_diff`.
  Pass `runs=len(seeds)` when scores are concatenated over several CV runs.

## Design decisions
- Pairs as one string: a pair is stored as `sentence1 + SEP + sentence2` (SEP defined once in `data.py`) so every tutorial function works unchanged; pair encoders split on SEP.
- Pair features: encode each sentence to v1, v2 and combine as `[|v1 - v2|, v1 * v2]`.
- Three representations: token indicators, bag-of-words counts (both from the tutorial), BPE-token bigrams (my addition: captures local word order, which unigrams cannot see, and PAWS pairs often differ in word order).
- Three architectures: log-linear (`logistic`), one-hidden-layer MLP (`mlp`), two-hidden-layer sparse MLP (my addition, in `models.py`).
- Full 3x3 grid on grouped folds (primary results, `results/p1_full/`): 5-fold CV, one seed, 20k pairs. Pairs linked by a shared sentence stay in one fold, because 99.5% of train pairs share a sentence with another pair.
- Random folds (comparison, `results/p1/`): five systems with the tutorial's random split.
- Baseline: majority class, always reported.
- Metrics: accuracy (primary, used for ROPE) + F1 of the paraphrase class.
- Leakage finding: on random folds 71.2% of validation pairs share a sentence with training, and accuracy depends on whether the training labels of that sentence agree with the pair (0.873) or disagree (0.373). Disagreement is the more common case, so random folds understated the bigram MLPs. `scripts/07_check_leakage.py` reproduces this.
- Known limitation (MLP selection): the tutorial's MLP training keeps the epoch with the best loss on the validation fold it then reports, which favours the MLPs slightly over log-linear. On the official validation split, which shares no sentence with train, the grouped bigram scores stay within 0.006 of their CV scores.
- Tokenizer and bigram vocabulary: fitted on the distinct train sentences, without labels, before CV (as in the tutorial), so validation folds' text is seen by these unsupervised steps.
- Memory: dense vectors per example are stacked by the tutorial. Keep vocab small (1k BPE types, 5k bigrams by frequency), use float32, and subsample to 20k train pairs via config. The tutorial code stays unedited.
- Test set: PAWS `test` is not used in P1. All model selection uses CV on `train`; the official `validation` split is only a final check on the bigram models.

## File map
| Path | Purpose |
|---|---|
| `src/paws/config.py` | Paths, seeds, k, vocab sizes, model hyperparameters, ROPE, subsample size |
| `src/paws/tutorial.py` | Adds `external/nlp1-student` to `sys.path`; re-exports `part1` |
| `src/paws/data.py` | Load PAWS, SEP, pair<->text, label names, dataset statistics |
| `src/paws/splits.py` | Grouped CV folds and a check that no sentence crosses folds |
| `src/paws/tokenization.py` | Train / save / load BPE tokenizer |
| `src/paws/encoders.py` | `PairTokenIndicator`, `PairBagOfWords`, `PairBigram` |
| `src/paws/models.py` | Two-hidden-layer sparse MLP + training/prediction functions |
| `src/paws/evaluation.py` | Accuracy, F1, majority baseline |
| `src/paws/experiment.py` | Cross-validates a system and runs the random-fold systems, saving per-fold scores as JSON |
| `src/paws/analysis.py` | Tables, box plots, ROPE comparisons |
| `scripts/01..07_*.py` | Thin CLI entry points, one per pipeline step |
| `notebooks/` | Exploration and results presentation (call into `src/paws`, no heavy logic) |
| `notes/`, `slides/` | Error analysis, blog ideas, oral backup slides |

## Conventions
- Python 3.13, type hints, docstrings, no notebooks-only logic.
- Every random step takes a seed from `config.py`.
- Results are written to `results/p1_full/` and `results/p1/` as JSON/CSV/PNG; scripts must be re-runnable.
- Non-obvious choices get a comment explaining *why* (the oral asks about every choice).
