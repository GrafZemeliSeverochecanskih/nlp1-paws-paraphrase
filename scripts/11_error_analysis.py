"""P2 step 6: which pair features predict correctness, per system (part2.analysis.fit_bda_model) -> results/p2/bda/."""

import argparse
import json
import random

import arviz as az
import numpy as np

from paws.analysis import oof_predictions
from paws.config import p2_variant, variant_parser
from paws.data import load_pairs
from paws.embeddings import PairTokenizer, load_embeddings
from paws.error_features import FEATURE_NAMES, pair_features
from paws.tutorial import part2

parser = argparse.ArgumentParser(parents=[variant_parser()])
parser.add_argument("systems", nargs="+", help="names of saved systems in results/p2/cv/")
# MCMC over all 20k pairs x systems is slow on CPU, so pairs are subsampled (final run: --max-pairs 4000).
parser.add_argument("--max-pairs", type=int, default=3000)
args = parser.parse_args()

cfg = p2_variant(args)
out = cfg.results_dir / "bda"
out.mkdir(parents=True, exist_ok=True)
pairs = load_pairs(cfg)
tokenizer = PairTokenizer(cfg, load_embeddings(cfg))

# Correctness per pair from the first seed's out-of-fold predictions: every pair is validated exactly once.
correct = {}
for name in args.systems:
    saved = json.loads((cfg.results_dir / "cv" / f"{name}.json").read_text())
    correct[name] = {i: int(p == pairs[i][1]) for i, p in oof_predictions(saved, cfg.seeds[0]).items()}

ids = sorted(set.intersection(*(set(c) for c in correct.values())))
ids = sorted(random.Random(cfg.seeds[0]).sample(ids, min(args.max_pairs, len(ids))))
# The gold label is a control column: systems near the majority baseline are "correct" mostly when the label is
# "different", so without it a feature that merely correlates with the label would look predictive of correctness.
names = FEATURE_NAMES + ["is_paraphrase"]
features = np.array([[pair_features(pairs[i][0], tokenizer)[f] for f in FEATURE_NAMES] + [pairs[i][1]] for i in ids], dtype=float)

sys_id = np.repeat(np.arange(len(args.systems)), len(ids))
correctness = np.concatenate([[correct[name][i] for i in ids] for name in args.systems])
mcmc = part2.analysis.fit_bda_model(
    len(args.systems), sys_id, np.tile(features, (len(args.systems), 1)), correctness, seed=cfg.seeds[0]
)
posterior = part2.analysis.posterior_to_arviz(mcmc, args.systems, names)
summary = az.summary(posterior, var_names=["sys_shift", "w"])
summary.to_csv(out / "bda_summary.csv")
print(summary.to_string())
