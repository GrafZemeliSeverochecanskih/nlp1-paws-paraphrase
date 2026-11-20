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

## Notes on the tutorial code (part2)
- `part2.clslib.PretrainedEmbeddings(path, num_dims, skip_first_line=False)` reads a GloVe `.txt` or single-file `.zip`; id 0 = `<pad>`, id 1 = `<unk>`; `.vocab` is frozen, `.vectors` is a numpy matrix.
- `part2.clslib.BasicTokenizer(lowercase, remove_punct, remove_stopwords, language, vocab)`: NLTK word tokens; `ids` does not lowercase by itself.
- `part2.clslib.EmbedAndPoolEncoder(embedding_matrix, average_pool, padding_idx, freeze_embeddings)` and `LSTMEncoder(hidden_size, embedding_matrix, padding_idx, freeze_embeddings, word_dropout)`, both with `.enc_dim` and `.pad_idx`; `SoftmaxRegressor(hidden_size1, hidden_size2, num_classes, encoder, dropout)` on top.
- `part2.experiment.cross_validation(outcome_space, data_splits, tokenizer, emb_model, enc_type, enc_kwargs, regressor_kwargs, opt_kwargs, device, save_to)` with `enc_type in {"embed-and-pool", "lstm"}` -> `{"models", "predictions", "accuracies"}` keyed by fold.
- `part2.experiment.train` builds the encoder class from `enc_type` and keeps the epoch with the best validation loss; `load(path, emb_model, device)` rebuilds it the same way; `predict_logprobs(model, tokenizer, texts, batch_size, device)`.
- `part2.analysis.fit_bda_model(num_systems, sys_id, features, correctness, seed)` -> MCMC; `posterior_to_arviz(mcmc, system_names, feature_names)`.

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
| `src/paws/tutorial.py` | Adds `external/nlp1-student` to `sys.path`; re-exports `part1` and `part2` |
| `src/paws/data.py` | Load PAWS, SEP, pair<->text, label names, dataset statistics |
| `src/paws/splits.py` | Grouped CV folds and a check that no sentence crosses folds |
| `src/paws/tokenization.py` | Train / save / load BPE tokenizer |
| `src/paws/encoders.py` | `PairTokenIndicator`, `PairBagOfWords`, `PairBigram` |
| `src/paws/models.py` | Two-hidden-layer sparse MLP + training/prediction functions |
| `src/paws/evaluation.py` | Accuracy, F1, majority baseline |
| `src/paws/experiment.py` | Cross-validates a system and runs the random-fold systems, saving per-fold scores as JSON |
| `src/paws/analysis.py` | Tables, box plots, ROPE comparisons |
| `src/paws/embeddings.py` | P2: load GloVe, `PairTokenizer` (pair -> GloVe ids with a separator), coverage |
| `src/paws/neural.py` | P2: `SiameseEncoder`, `BiLSTMEncoder`, `encoder_patch` (plugs them into `part2.experiment.train`) |
| `src/paws/error_features.py` | P2: per-pair features for the Bayesian error analysis |
| `scripts/01..07_*.py` | P1: thin CLI entry points, one per pipeline step |
| `scripts/08..13_*.py` | P2: coverage, grid, analysis (incl. same-word-set accuracy, error examples), BDA error analysis, official validation, learning curve |
| `notebooks/` | Exploration and results presentation (call into `src/paws`, no heavy logic) |
| `notes/`, `slides/` | Error analysis, blog ideas, oral backup slides |

## P2 design decisions
- Folds: same subsample, k and seeds as P1 (`P2Config` inherits them), so P2 folds equal P1's grouped folds and P1 systems are a paired reference.
- Training: tutorial `part2` training is reused unchanged; `neural.encoder_patch` swaps the encoder class it builds (as P1 patched `SparseMLP`).
- Embeddings: GloVe 2024 Wikipedia+Gigaword 50d, because PAWS is built from Wikipedia. On the 40,223 distinct train sentences 0.97% of tokens and 8.3% of types are OOV, and OOV is 5.5x higher for capitalised words (2.45% vs 0.44%): the gaps are rare names, the entities PAWS swaps.
- Tokenizer: `BasicTokenizer` with lowercase=True (the GloVe vocab is lowercase) and remove_punct=False (True lowers token OOV only to 0.75% but deletes 13% of tokens, incl. the commas that mark the clauses PAWS reorders).
- Separator: a new random row appended to the GloVe vectors (my addition: no word maps to it, so the siamese split is exact; random because zeros is `<pad>`).
- Pair combination: cross-encoding (one sequence `s1 [SEP] s2`, the tutorial's setting) vs siamese (my addition: one shared encoder per sentence, then `[|u - v|, u * v]` as in P1), so P1 vs P2 differs only in the sentence encoder.
- Encoders: mean/sum pooling and LSTM from the tutorial, BiLSTM (my addition, in `neural.py`: final forward and backward states concatenated, so a swap near the sentence start is not dominated by the sentence end).
- Grid: {pooling, LSTM, BiLSTM} x {cross, siamese} on 5 grouped folds, one seed, plus sum pooling, LSTM state 64/256 and trainable embeddings; one change per comparison in `scripts/10`. The top two systems are rerun with 3 seeds (`results/p2_seeds0-1-2/`).
- Result: only siamese + order-aware encoding learns PAWS (bilstm+siamese 0.736, lstm+siamese 0.716, majority 0.560); pooling and every cross-encoded LSTM stay at majority. Siamese beats cross by +0.149 (LSTM) and +0.173 (BiLSTM), P > 0.999.
- BiLSTM vs LSTM (siamese, 3 seeds): 0.739 vs 0.718, P(BiLSTM better) 0.96. LSTM state 64/128/256: 0.731/0.736/0.740, mostly ties.
- P1 vs P2: the P1 bigram MLP (0.617) beats lstm+cross (P 0.99) and loses to lstm+siamese (P 0.999), so the gain comes from the encoder plus the comparison, not from pretrained embeddings alone.
- Trainable embeddings: no gain (0.714 vs 0.716 frozen) and the best epoch is the first or second, so 64M embedding parameters overfit 16k pairs. Kept frozen; `bilstm+siamese+trainable` is defined but not run.
- Learning curve (`scripts/13`, 2k/5k/10k/20k pairs): lstm+siamese rises 0.657 -> 0.716 and is still rising; cross and pooling stay at majority at every size.
- Embedding alternative (GloVe 2014 6B, `results/p2_glove6B-*`): 2014 50d has twice the token OOV (1.83%) but scores the same (0.732 vs 0.736, P(tie) 0.52). Size helps the BiLSTM: 0.732/0.741/0.753 for 50/100/300d; pooling stays at majority.
- 300d with 3 seeds (`results/p2_seeds0-1-2_glove6B-300d/`): 0.750 vs 0.739 for 2024 50d, P(better) 0.60, P(tie) 0.40, so likely better but not decisive. 2024 50d stays the main embedding (better coverage, smaller, faster).
- Known limitation (as in P1): `part2.experiment.train` keeps the best-loss epoch on the validation fold it then reports. On the official validation split the scores hold: bilstm+siamese 0.741 (CV 0.736), lstm+siamese 0.720 (CV 0.716).
- Error features (my addition, `error_features.py`, on the model's own tokens): same word set, Jaccard overlap, order change (1 - Kendall tau of the shared words' positions), OOV rate, capitalised words not at sentence start, length.
- Bayesian error analysis (`scripts/11`): the gold label is a control column, because near-majority systems are "correct" mostly when the label is *different*. Findings in `notes/error_analysis.md`.
- Variants: `--k/--seeds/--subsample/--glove` write to their own `results/p2_*` folder, so runs on different folds never share a table; `run_p2_system` saves after every seed and resumes a stopped run.
- Storage: every saved model contains the GloVe matrix (part of the state dict even when frozen), so model folders are large and gitignored; the CV JSONs keep every score and out-of-fold prediction.

## Conventions
- Python 3.13, type hints, docstrings, no notebooks-only logic.
- Every random step takes a seed from `config.py`.
- Results are written to `results/p1_full/`, `results/p1/`, `results/p2/` and `results/p2_*/` as JSON/CSV/PNG; scripts must be re-runnable.
- Non-obvious choices get a comment explaining *why* (the oral asks about every choice).
