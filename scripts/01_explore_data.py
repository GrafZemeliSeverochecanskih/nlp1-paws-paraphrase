"""Step 1: dataset statistics."""

import json

from paws.config import Config
from paws.data import load_split, split_stats

cfg = Config()
cfg.make_dirs()
stats = {split: split_stats(load_split(cfg, split)) for split in ("train", "validation", "test")}
(cfg.results_dir / "data_stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
