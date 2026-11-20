"""Run the P1 systems (encoder x architecture) and the P2 neural systems with k-fold CV; save results to disk."""

import json
import random
from unittest.mock import patch

import numpy as np
import torch

from paws import tokenization
from paws.config import Config, P2Config
from paws.data import OUTCOMES, load_pairs, train_sentences
from paws.encoders import PairBagOfWords, PairBigram, PairEncoder, PairTokenIndicator
from paws.evaluation import fold_f1, majority_scores
from paws.models import TwoLayerSparseMLP
from paws.neural import encoder_patch
from paws.tutorial import part1, part2


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
        if saved["config"] == json.loads(json.dumps(cfg.to_dict())):  # JSON round trip: saved seeds are a list
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


def run_p2_system(name: str, spec: dict, tokenizer, emb, splits: dict[int, dict], cfg: P2Config) -> dict:
    """Cross-validate one P2 system over all seeds with part2.experiment.cross_validation.

    spec: {"enc_type": "embed-and-pool" | "lstm", "siamese": bool, "bidirectional": bool (optional),
           "enc_kwargs": dict passed to the tutorial encoder (optional)}.
    Saves accuracies, F1 and per-fold predictions with validation ids (for error analysis) to cv/<name>.json.
    Saved after every seed, so a stopped run resumes at the first unfinished seed; reuses a finished result
    made with the same settings and spec.
    """
    path = cfg.results_dir / "cv" / f"{name}.json"
    settings = json.loads(json.dumps({"config": cfg.to_dict(), "spec": spec}))  # JSON round trip: tuples -> lists
    scores = {"name": name, **settings, "accuracies": [], "f1": [], "folds": [], "seeds_done": []}
    if path.exists():
        saved = json.loads(path.read_text())
        if {"config": saved["config"], "spec": saved["spec"]} == settings:
            saved.setdefault("seeds_done", list(cfg.seeds))  # files from before per-seed saving are complete
            if saved["seeds_done"] == list(cfg.seeds):
                return saved
            scores = saved

    enc_kwargs = {"freeze_embeddings": cfg.freeze_embeddings} | spec.get("enc_kwargs", {})
    if spec["enc_type"] == "lstm":
        enc_kwargs.setdefault("hidden_size", cfg.lstm_hidden)
    for seed, data_splits in splits.items():
        if seed in scores["seeds_done"]:
            continue
        set_seed(seed)
        with encoder_patch(spec["enc_type"], spec["siamese"], tokenizer.sep_id, spec.get("bidirectional", False)):
            result = part2.experiment.cross_validation(
                OUTCOMES,
                data_splits,
                tokenizer,
                emb,
                spec["enc_type"],
                dict(enc_kwargs),  # train() adds padding_idx in place
                cfg.regressor_kwargs(),
                cfg.opt_kwargs(),
                device=cfg.device,
                save_to=str(cfg.results_dir / "models" / name / f"seed-{seed}"),
            )
        for fold, predictions in result["predictions"].items():
            scores["accuracies"].append(float(result["accuracies"][fold]))
            scores["f1"].append(fold_f1(data_splits[fold], predictions))
            scores["folds"].append(
                {
                    "seed": seed,
                    "fold": fold,
                    "validation_ids": data_splits[fold]["validation_ids"],
                    "predictions": [int(p) for p in predictions],
                }
            )
        scores["seeds_done"].append(seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(scores))
    return scores
