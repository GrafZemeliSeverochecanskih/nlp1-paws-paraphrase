"""Run the five P1 systems (encoder x architecture) with k-fold CV and save results to disk."""

import json
import random
from unittest.mock import patch

import numpy as np
import torch

from paws import tokenization
from paws.config import Config
from paws.data import OUTCOMES, load_pairs, train_sentences
from paws.encoders import PairBagOfWords, PairBigram, PairEncoder, PairTokenIndicator
from paws.evaluation import fold_f1, majority_scores
from paws.models import TwoLayerSparseMLP
from paws.tutorial import part1


def set_seed(seed: int) -> None:
    """Seed every generator training touches; part1 training functions take no seed themselves."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_system(
    name: str, encoder: PairEncoder, model_option: str, splits: dict[int, dict], cfg: Config, two_layer: bool = False
) -> dict:
    """Cross-validate one system over all seeds. Reuses a saved result when it was made with the same settings."""
    path = cfg.results_dir / "cv" / f"{name}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["config"] == cfg.to_dict():
            return saved

    scores = {"name": name, "config": cfg.to_dict(), "accuracies": [], "f1": []}
    for seed, data_splits in splits.items():
        set_seed(seed)
        # part1.smlp has no hook for another model; patching it keeps the training loop identical to the 1-layer MLP.
        model_class = TwoLayerSparseMLP if two_layer else part1.smlp.SparseMLP
        with patch.object(part1.smlp, "SparseMLP", model_class):
            result = part1.experiment.cross_validation_experiment(
                data_splits,
                OUTCOMES,
                encoder,
                model_option,
                save_to=str(cfg.results_dir / "models" / name / f"seed-{seed}"),
                loglin_args=cfg.loglin_args(),
                smlp_args=cfg.smlp_args(),
            )
        scores["accuracies"] += [float(accuracy) for accuracy in result["accuracies"].values()]
        scores["f1"] += [fold_f1(data_splits[fold], preds) for fold, preds in result["predictions"].items()]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scores, indent=2))
    return scores


def run_all(cfg: Config) -> None:
    """Majority baseline, three encoders with log-linear, then the best encoder with both MLPs."""
    cfg.make_dirs()
    tokenizer = tokenization.load(cfg)
    encoders = {
        "indicator": PairTokenIndicator(tokenizer),
        "bow": PairBagOfWords(tokenizer),
        "bigram": PairBigram(tokenizer, train_sentences(cfg), cfg.max_bigrams),
    }
    pairs = load_pairs(cfg)
    splits = {seed: part1.experiment.cross_validation_split(pairs, cfg.k, seed) for seed in cfg.seeds}

    baseline = [majority_scores(fold) for data_splits in splits.values() for fold in data_splits.values()]
    (cfg.results_dir / "cv").mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "cv" / "majority.json").write_text(
        json.dumps({"name": "majority", "accuracies": [a for a, _ in baseline], "f1": [f for _, f in baseline]}, indent=2)
    )

    logistic = {rep: run_system(f"{rep}+loglin", encoder, "logistic", splits, cfg) for rep, encoder in encoders.items()}
    best = max(logistic, key=lambda rep: np.mean(logistic[rep]["accuracies"]))
    print(f"Best representation with log-linear: {best}")
    run_system(f"{best}+mlp", encoders[best], "mlp", splits, cfg)
    run_system(f"{best}+mlp2", encoders[best], "mlp", splits, cfg, two_layer=True)
