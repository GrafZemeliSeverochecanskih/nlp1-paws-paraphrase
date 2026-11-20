"""P2 learning curve: mean CV accuracy per system against the subsample size.

Reads results/p2/ (the default subsample) and every results/p2_n<N>/ made with `09_run_p2_grid.py --subsample N`.
Only runs with the default k, seeds and embeddings are used, so every point is the same kind of estimate.
"""

import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paws.config import ROOT, P2Config

default = P2Config()
curve = defaultdict(dict)
for path in sorted(ROOT.glob("results/p2*/cv/*.json")):
    saved = json.loads(path.read_text())
    config = saved.get("config")  # majority.json has none
    same_setup = config and config["glove_path"] == str(default.glove_path)
    if same_setup and config["k"] == default.k and config["seeds"] == list(default.seeds):
        curve[saved["name"]][config["subsample_size"]] = float(np.mean(saved["accuracies"]))

curve = {name: points for name, points in curve.items() if len(points) > 1}
if not curve:
    raise SystemExit("No system has runs at two or more subsample sizes yet (run 09 with --subsample N).")
table = pd.DataFrame(curve).sort_index()
table.index.name = "subsample_size"
table.to_csv(default.results_dir / "learning_curve.csv")
print(table.to_string())

fig, ax = plt.subplots()
for name, points in curve.items():
    sizes = sorted(points)
    ax.plot(sizes, [points[n] for n in sizes], marker="o", label=name)
ax.set_xscale("log")
ax.set_xlabel(f"Train pairs in the subsample ({default.k}-fold CV: each model trains on {default.k - 1}/{default.k})")
ax.set_ylabel("Mean CV accuracy")
ax.legend()
fig.savefig(default.results_dir / "learning_curve.png", dpi=200, bbox_inches="tight")
