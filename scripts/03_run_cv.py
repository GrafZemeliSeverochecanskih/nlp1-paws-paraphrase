"""Step 3: run all cross-validation experiments. `--smoke` does a fast check in results/smoke first."""

import argparse

from paws.config import ROOT, Config
from paws.experiment import run_all

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true", help="2 folds, 500 pairs, 2 epochs, separate results folder")
args = parser.parse_args()

if args.smoke:
    run_all(Config(k=2, subsample_size=500, num_epochs=2, loglin_max_iter=5, results_dir=ROOT / "results" / "smoke"))
else:
    run_all(Config())
