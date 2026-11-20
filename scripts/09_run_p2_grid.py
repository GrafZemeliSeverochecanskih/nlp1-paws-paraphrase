"""P2 steps 3-5: pooling and LSTM encoders, siamese vs cross-encoding, on P1's grouped folds -> results/p2/cv/.

Each system is saved when it finishes, and a rerun reuses finished systems. `--smoke` does a fast check in
results/smoke_p2 first. `--only NAME ...` runs a subset (e.g. for the staged search).
`--seeds 0 1 2` (top systems with 3 CV runs), `--subsample N` (learning curve) and `--glove NAME` (other embeddings)
write to their own folder, e.g. results/p2_seeds0-1-2, results/p2_n5000, results/p2_glove6B-100d.
"""

import argparse
import json
import time

import numpy as np

from paws.config import ROOT, P2Config, p2_variant, variant_parser
from paws.data import load_pairs
from paws.embeddings import PairTokenizer, load_embeddings
from paws.evaluation import majority_scores
from paws.experiment import run_p2_system
from paws.splits import assert_no_shared_sentences, grouped_cross_validation_split

# Pooling vs LSTM vs BiLSTM, each cross-encoded (one sequence) and siamese (each sentence alone, then [|u-v|, u*v]).
# "enc_kwargs" go to the tutorial encoder (e.g. average_pool, word_dropout, hidden_size, freeze_embeddings).
SYSTEMS = {
    "pool+cross": {"enc_type": "embed-and-pool", "siamese": False, "enc_kwargs": {"average_pool": True}},
    "pool+siamese": {"enc_type": "embed-and-pool", "siamese": True, "enc_kwargs": {"average_pool": True}},
    # Sum instead of mean pooling: keeps sentence length in the vector norm.
    "pool+cross+sum": {"enc_type": "embed-and-pool", "siamese": False, "enc_kwargs": {"average_pool": False}},
    "pool+siamese+sum": {"enc_type": "embed-and-pool", "siamese": True, "enc_kwargs": {"average_pool": False}},
    "lstm+cross": {"enc_type": "lstm", "siamese": False},
    "lstm+siamese": {"enc_type": "lstm", "siamese": True},
    "bilstm+cross": {"enc_type": "lstm", "siamese": False, "bidirectional": True},
    "bilstm+siamese": {"enc_type": "lstm", "siamese": True, "bidirectional": True},
    # State size of the best system (default 128, P2Config.lstm_hidden).
    "bilstm+siamese+h64": {"enc_type": "lstm", "siamese": True, "bidirectional": True, "enc_kwargs": {"hidden_size": 64}},
    "bilstm+siamese+h256": {"enc_type": "lstm", "siamese": True, "bidirectional": True, "enc_kwargs": {"hidden_size": 256}},
    # Trainable vs frozen GloVe, on both siamese LSTMs (bilstm+siamese is the best frozen system).
    "lstm+siamese+trainable": {"enc_type": "lstm", "siamese": True, "enc_kwargs": {"freeze_embeddings": False}},
    "bilstm+siamese+trainable": {
        "enc_type": "lstm",
        "siamese": True,
        "bidirectional": True,
        "enc_kwargs": {"freeze_embeddings": False},
    },
}

parser = argparse.ArgumentParser(parents=[variant_parser()])
parser.add_argument("--smoke", action="store_true", help="2 folds, 500 pairs, 2 epochs, separate results folder")
parser.add_argument("--only", nargs="+", choices=list(SYSTEMS), help="run only these systems")
args = parser.parse_args()

if args.smoke:
    cfg = P2Config(k=2, subsample_size=500, num_epochs=2, results_dir=ROOT / "results" / "smoke_p2")
else:
    cfg = p2_variant(args)
cfg.make_dirs()
(cfg.results_dir / "cv").mkdir(exist_ok=True)
print(f"Device: {cfg.device}, results: {cfg.results_dir}", flush=True)

emb = load_embeddings(cfg)
tokenizer = PairTokenizer(cfg, emb)
pairs = load_pairs(cfg)
splits = {seed: grouped_cross_validation_split(pairs, cfg.k, seed) for seed in cfg.seeds}
for data_splits in splits.values():
    assert_no_shared_sentences(data_splits, len(pairs))

baseline = [majority_scores(fold) for data_splits in splits.values() for fold in data_splits.values()]
(cfg.results_dir / "cv" / "majority.json").write_text(
    json.dumps({"accuracies": [a for a, _ in baseline], "f1": [f for _, f in baseline]}, indent=2)
)

for name in args.only or SYSTEMS:
    result = run_p2_system(name, SYSTEMS[name], tokenizer, emb, splits, cfg)
    print(f"DONE {name} {time.strftime('%H:%M:%S')} {np.mean(result['accuracies']):.4f}", flush=True)
