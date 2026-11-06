"""Step 5: the full 3x3 grid on grouped folds, saved in results/p1_full/. These are the primary results.

Run after step 2 (it reads the tokenizer from results/p1/). Each system is saved when it finishes, and a rerun
reuses finished systems. `--smoke` does a fast check in results/smoke_full first.
"""

import argparse
import json
import time

import numpy as np

from paws import tokenization
from paws.config import ROOT, Config
from paws.data import load_pairs, train_sentences
from paws.encoders import PairBagOfWords, PairBigram, PairTokenIndicator
from paws.evaluation import majority_scores
from paws.experiment import run_system
from paws.splits import assert_no_shared_sentences, grouped_cross_validation_split

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true", help="2 folds, 500 pairs, 2 epochs, separate results folder")
args = parser.parse_args()

if args.smoke:
    cfg = Config(k=2, subsample_size=500, num_epochs=2, loglin_max_iter=5, results_dir=ROOT / "results" / "smoke_full")
else:
    cfg = Config(results_dir=ROOT / "results" / "p1_full")
cfg.make_dirs()
(cfg.results_dir / "cv").mkdir(exist_ok=True)

tokenizer = tokenization.load(cfg)
encoders = {
    "indicator": PairTokenIndicator(tokenizer),
    "bow": PairBagOfWords(tokenizer),
    "bigram": PairBigram(tokenizer, train_sentences(cfg), cfg.max_bigrams),
}
pairs = load_pairs(cfg)
splits = {seed: grouped_cross_validation_split(pairs, cfg.k, seed) for seed in cfg.seeds}
for data_splits in splits.values():
    assert_no_shared_sentences(data_splits, len(pairs))
print("Fold sizes:", [len(fold["validation"]) for fold in splits[cfg.seeds[0]].values()], flush=True)

baseline = [majority_scores(fold) for data_splits in splits.values() for fold in data_splits.values()]
(cfg.results_dir / "cv" / "majority.json").write_text(
    json.dumps({"accuracies": [a for a, _ in baseline], "f1": [f for _, f in baseline]}, indent=2)
)


def run(rep: str, architecture: str) -> dict:
    """Run one system and report when it finished."""
    model_option = "logistic" if architecture == "loglin" else "mlp"
    result = run_system(f"{rep}+{architecture}", encoders[rep], model_option, splits, cfg, architecture == "mlp2")
    print(f"DONE {rep}+{architecture} {time.strftime('%H:%M:%S')} {np.mean(result['accuracies']):.4f}", flush=True)
    return result


# Log-linear first, then the best representation's MLPs, so the core comparison survives a stop before the end.
logistic = {rep: run(rep, "loglin") for rep in encoders}
best = max(logistic, key=lambda rep: np.mean(logistic[rep]["accuracies"]))
print(f"Best representation with log-linear: {best}", flush=True)
for rep in [best] + [rep for rep in encoders if rep != best]:
    run(rep, "mlp")
    run(rep, "mlp2")
