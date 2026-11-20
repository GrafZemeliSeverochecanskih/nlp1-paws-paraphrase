"""P2 final check: the saved fold models of the given systems on the official PAWS validation split.

That split shares no sentence with train, so it tests whether the CV scores (best epoch picked on the validation
fold) hold on unseen sentences, as P1's 07 did for the bigram models. Run after 09; same `--k/--seeds/--subsample`.
"""

import argparse
import json

import numpy as np

from paws.config import p2_variant, variant_parser
from paws.data import SEP, load_split
from paws.embeddings import PairTokenizer, load_embeddings
from paws.neural import encoder_patch
from paws.tutorial import part2

parser = argparse.ArgumentParser(parents=[variant_parser()])
parser.add_argument("systems", nargs="+", help="names of saved systems in results/p2/cv/")
args = parser.parse_args()

cfg = p2_variant(args)
emb = load_embeddings(cfg)
tokenizer = PairTokenizer(cfg, emb)  # appends the SEP row, so the embedding matrix matches the trained models
official = load_split(cfg, "validation")
texts = [row["sentence1"] + SEP + row["sentence2"] for row in official]
gold = np.array([row["label"] for row in official])
print(f"Official validation: {len(gold)} pairs, majority accuracy {max(gold.mean(), 1 - gold.mean()):.4f}")

results = {}
for name in args.systems:
    saved = json.loads((cfg.results_dir / "cv" / f"{name}.json").read_text())
    spec, seed = saved["spec"], cfg.seeds[0]
    accuracies = []
    for fold in [f["fold"] for f in saved["folds"] if f["seed"] == seed]:
        path = cfg.results_dir / "models" / name / f"seed-{seed}" / f"fold-{fold}" / "model"
        # part2.experiment.load rebuilds the tutorial class; the patch makes it rebuild the siamese/BiLSTM encoder.
        with encoder_patch(spec["enc_type"], spec["siamese"], tokenizer.sep_id, spec.get("bidirectional", False)):
            model = part2.experiment.load(str(path), emb, device=cfg.device)
        logprobs = part2.experiment.predict_logprobs(model, tokenizer, texts, cfg.batch_size, device=cfg.device)
        accuracies.append(float((logprobs.argmax(1) == gold).mean()))
    results[name] = {"official_accuracies": accuracies, "cv_accuracies": saved["accuracies"][: len(accuracies)]}
    print(f"{name}: official {np.mean(accuracies):.4f} vs CV {np.mean(results[name]['cv_accuracies']):.4f}")

(cfg.results_dir / "official_validation.json").write_text(json.dumps(results, indent=2))
