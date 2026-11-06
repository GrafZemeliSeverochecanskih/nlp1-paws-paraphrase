"""Step 6: results table, box plot and ROPE analysis for the grouped grid, plus a comparison with the random folds."""

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from paws.analysis import box_plot, load_scores, results_table, rope_table
from paws.config import ROOT, Config

cfg = Config(results_dir=ROOT / "results" / "p1_full")
scores = load_scores(cfg)
pairs = [
    ("majority", "bigram+loglin"),
    ("majority", "bigram+mlp"),
    ("indicator+loglin", "bow+loglin"),
    ("bow+loglin", "bigram+loglin"),
    ("indicator+loglin", "bigram+loglin"),
    ("bow+mlp", "bigram+mlp"),
    ("indicator+mlp", "bigram+mlp"),
    ("bigram+loglin", "bigram+mlp"),
    ("bigram+loglin", "bigram+mlp2"),
    ("bigram+mlp", "bigram+mlp2"),
    ("bow+loglin", "bow+mlp"),
    ("bow+mlp", "bow+mlp2"),
    ("indicator+loglin", "indicator+mlp"),
    ("indicator+mlp", "indicator+mlp2"),
]
table = results_table(scores)
rope = rope_table(scores, cfg, pairs)
box_plot(scores, cfg)
table.to_csv(cfg.results_dir / "results_table.csv", index=False)
rope.to_csv(cfg.results_dir / "rope_table.csv", index=False)

# The folds differ, so only the means can be compared; a paired Bayesian comparison needs the same folds.
random_folds = results_table(load_scores(Config())).rename(columns=lambda c: c if c == "system" else f"random_{c}")
versus = table.merge(random_folds, on="system", how="left")
versus.to_csv(cfg.results_dir / "random_vs_grouped.csv", index=False)

pd.set_option("display.width", 200)
print(table.to_markdown(floatfmt=".4f"))
print(rope.to_markdown(floatfmt=".4f"))
print(versus[["system", "accuracy_mean", "random_accuracy_mean"]].to_markdown(floatfmt=".4f"))
