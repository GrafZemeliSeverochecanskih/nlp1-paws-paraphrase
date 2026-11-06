"""Step 4: results table, box plot and ROPE analysis."""

import matplotlib

matplotlib.use("Agg")

from paws.analysis import box_plot, load_scores, results_table, rope_table
from paws.config import Config

cfg = Config()
scores = load_scores(cfg)
table = results_table(scores)
rope = rope_table({name: s for name, s in scores.items() if name != "majority"}, cfg)
box_plot(scores, cfg)
table.to_csv(cfg.results_dir / "results_table.csv", index=False)
rope.to_csv(cfg.results_dir / "rope_table.csv", index=False)
print(table.to_markdown(floatfmt=".4f"))
print(rope.to_markdown(floatfmt=".4f"))
