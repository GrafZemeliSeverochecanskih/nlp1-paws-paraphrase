"""P2 step 5: results table, box plot and ROPE comparisons for P2 systems, with P1's best grouped system as reference.

Also: accuracy on same-word-set pairs vs the rest (the main research question) and example pairs for the error
analysis, both from the saved out-of-fold predictions (no retraining). Same `--k/--seeds/--subsample` as 09.
"""

import argparse
import json
import random

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from paws.analysis import box_plot, load_scores, oof_predictions, results_table, rope_table
from paws.config import EMBEDDINGS, ROOT, p2_variant, variant_parser
from paws.data import OUTCOMES, load_pairs, split_pair
from paws.embeddings import PairTokenizer, load_embeddings
from paws.error_features import pair_features

P1_REFERENCE = "bigram+mlp"  # best P1 system on grouped folds (results/p1_full/results_table.csv)
FIXER, FIXED = "lstm+siamese", "pool+cross"  # examples where FIXER is right and FIXED is wrong
NUM_EXAMPLES = 10

args = argparse.ArgumentParser(parents=[variant_parser()]).parse_args()
cfg = p2_variant(args)
scores = load_scores(cfg)

# P2 inherits P1's subsample, k and seeds, so the folds are identical and the comparison can be paired.
p1 = json.loads((ROOT / "results" / "p1_full" / "cv" / f"{P1_REFERENCE}.json").read_text())
same_folds = all(p1["config"][key] == json.loads(json.dumps(cfg.to_dict()))[key] for key in ("subsample_size", "k", "seeds"))
if same_folds:
    scores[f"p1:{P1_REFERENCE}"] = p1
else:
    print(f"Skipping P1 reference: {P1_REFERENCE} was run on different folds")

# Systems trained with other embeddings (09 --glove NAME) on the same folds, as "glove<NAME>:<system>".
if cfg.glove_path == EMBEDDINGS["2024-50d"][0]:
    for glove in EMBEDDINGS:
        folder = cfg.results_dir.with_name(cfg.results_dir.name + f"_glove{glove}")
        for path in sorted((folder / "cv").glob("*.json")):
            saved = json.loads(path.read_text())
            if "folds" in saved and saved.get("seeds_done") == list(cfg.seeds):
                scores[f"glove{glove}:{path.stem}"] = saved

# (A, B): does B beat A? Each pair changes one thing.
pairs = [
    ("majority", "pool+cross"),  # can pooling learn anything on PAWS?
    ("pool+cross", "pool+siamese"),  # pair combination, pooling
    ("lstm+cross", "lstm+siamese"),  # pair combination, LSTM
    ("bilstm+cross", "bilstm+siamese"),  # pair combination, BiLSTM
    ("pool+cross", "lstm+cross"),  # word order, cross-encoding
    ("pool+siamese", "lstm+siamese"),  # word order, siamese
    ("lstm+cross", "bilstm+cross"),  # both directions, cross-encoding
    ("lstm+siamese", "bilstm+siamese"),  # both directions, siamese
    ("lstm+siamese", "lstm+siamese+trainable"),  # frozen vs trainable GloVe
    ("bilstm+siamese", "bilstm+siamese+trainable"),  # frozen vs trainable GloVe, best system
    ("pool+cross", "pool+cross+sum"),  # mean vs sum pooling
    ("pool+siamese", "pool+siamese+sum"),
    ("bilstm+siamese+h64", "bilstm+siamese"),  # LSTM state size 64 -> 128 -> 256
    ("bilstm+siamese", "bilstm+siamese+h256"),
    ("glove6B-50d:pool+siamese", "pool+siamese"),  # GloVe 2014 (400k words) vs 2024 (1.29M), both 50d
    ("glove6B-50d:bilstm+siamese", "bilstm+siamese"),
    ("glove6B-50d:pool+siamese", "glove6B-100d:pool+siamese"),  # embedding size 50 -> 100 -> 300 (GloVe 2014)
    ("glove6B-100d:pool+siamese", "glove6B-300d:pool+siamese"),
    ("glove6B-50d:bilstm+siamese", "glove6B-100d:bilstm+siamese"),
    ("glove6B-100d:bilstm+siamese", "glove6B-300d:bilstm+siamese"),
    ("bilstm+siamese", "glove6B-300d:bilstm+siamese"),  # main embedding vs the best one found
    (f"p1:{P1_REFERENCE}", "lstm+cross"),  # P1 vs P2, same pair input
    (f"p1:{P1_REFERENCE}", "lstm+siamese"),  # P1 vs best P2
]
pairs = [(a, b) for a, b in pairs if a in scores and b in scores]

table = results_table(scores)
rope = rope_table(scores, cfg, pairs)
box_plot(scores, cfg)
table.to_csv(cfg.results_dir / "results_table.csv", index=False)
rope.to_csv(cfg.results_dir / "rope_table.csv", index=False)
print(table.to_string(index=False))
print(rope.to_string(index=False))

# Same-word-set pairs: both sentences use the same words, so only word order can tell the label.
data = load_pairs(cfg)
tokenizer = PairTokenizer(cfg, load_embeddings(cfg))
features = [pair_features(text, tokenizer) for text, _ in data]
same = np.array([f["same_word_set"] == 1.0 for f in features])
gold = np.array([label for _, label in data])
rows = [
    {  # what always predicting the group's most frequent label scores, per group
        "system": "majority label within group",
        "accuracy_same_word_set": max(gold[same].mean(), 1 - gold[same].mean()),
        "accuracy_other": max(gold[~same].mean(), 1 - gold[~same].mean()),
        "n_same_word_set": int(same.sum()),
        "n_other": int((~same).sum()),
    }
]
neural = {name: s for name, s in scores.items() if "folds" in s}  # P2 systems: saved predictions per pair
for name, saved in neural.items():
    hits, groups = [], []
    for seed in cfg.seeds:
        predictions = oof_predictions(saved, seed)
        ids = np.array(list(predictions))
        hits.append(np.array(list(predictions.values())) == gold[ids])
        groups.append(same[ids])
    hits, groups = np.concatenate(hits), np.concatenate(groups)
    rows.append(
        {
            "system": name,
            "accuracy_same_word_set": hits[groups].mean(),
            "accuracy_other": hits[~groups].mean(),
            "n_same_word_set": int(groups.sum()),
            "n_other": int((~groups).sum()),
        }
    )
same_table = pd.DataFrame(rows)
same_table.to_csv(cfg.results_dir / "same_word_set_table.csv", index=False)
print(same_table.to_string(index=False))

# Example pairs for notes/error_analysis.md, from the first seed.
correct = {
    name: {i: p == data[i][1] for i, p in oof_predictions(saved, cfg.seeds[0]).items()}
    for name, saved in neural.items()
    if not name.startswith("glove")  # embedding variants would change the examples quoted in the notes
}
groups = {}
if FIXER in correct and FIXED in correct:
    groups[f"{FIXER} right, {FIXED} wrong"] = [i for i in correct[FIXER] if correct[FIXER][i] and not correct[FIXED][i]]
if correct:
    groups[f"all {len(correct)} P2 systems wrong"] = [i for i in range(len(data)) if not any(c.get(i, True) for c in correct.values())]
lines = ["# P2 error examples", "", f"Out-of-fold predictions, seed {cfg.seeds[0]}. Systems: {', '.join(correct)}.", ""]
for title, ids in groups.items():
    lines += [f"## {title} ({len(ids)} of {len(data)} pairs)", ""]
    for i in sorted(random.Random(cfg.seeds[0]).sample(ids, min(NUM_EXAMPLES, len(ids)))):
        sentence1, sentence2 = split_pair(data[i][0])
        f = features[i]
        lines += [
            f"- id {i}, gold **{OUTCOMES.name(data[i][1])}**, same_word_set {f['same_word_set']:.0f}, "
            f"order_change {f['order_change']:.2f}, oov_rate {f['oov_rate']:.2f}",
            f"  - S1: {sentence1}",
            f"  - S2: {sentence2}",
        ]
    lines.append("")
(cfg.results_dir / "error_examples.md").write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {cfg.results_dir / 'error_examples.md'}: " + ", ".join(f"{t}: {len(ids)}" for t, ids in groups.items()))
