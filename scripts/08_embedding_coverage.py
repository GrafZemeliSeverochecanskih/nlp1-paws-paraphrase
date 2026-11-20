"""P2 step 1: GloVe coverage of the PAWS train sentences -> results/p2/coverage.json (`--glove NAME`: that embedding's folder)."""

import argparse
import json

from paws.config import p2_variant, variant_parser
from paws.data import train_sentences
from paws.embeddings import PairTokenizer, coverage, load_embeddings

cfg = p2_variant(argparse.ArgumentParser(parents=[variant_parser()]).parse_args())
cfg.make_dirs()
emb = load_embeddings(cfg)
print(f"GloVe: {len(emb.vocab)} words x {cfg.emb_dim} dims", flush=True)
stats = coverage(PairTokenizer(cfg, emb), emb, train_sentences(cfg))
(cfg.results_dir / "coverage.json").write_text(json.dumps({"config": cfg.to_dict(), **stats}, indent=2))
print(json.dumps(stats, indent=2))
