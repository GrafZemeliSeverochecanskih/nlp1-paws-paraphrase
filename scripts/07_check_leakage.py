"""Step 7: explain the difference between random and grouped folds, using the saved models.

Scores the bigram models on the official validation split, which shares no sentence with train, and breaks down
random-fold accuracy by what training showed for the shared sentence. Run after steps 3 and 5.
"""

from collections import defaultdict
from unittest.mock import patch

import numpy as np

from paws import tokenization
from paws.config import ROOT, Config
from paws.data import SEP, load_pairs, load_split, split_pair, train_sentences
from paws.encoders import PairBigram
from paws.models import TwoLayerSparseMLP
from paws.splits import grouped_cross_validation_split
from paws.tutorial import part1

cfg = Config()
pairs = load_pairs(cfg)
corpus = train_sentences(cfg)
encoder = PairBigram(tokenization.load(cfg), corpus, cfg.max_bigrams)
official = [(row["sentence1"] + SEP + row["sentence2"], row["label"]) for row in load_split(cfg, "validation")]
official_texts = [text for text, _ in official]
official_gold = np.array([label for _, label in official])

known = set(corpus)
shared = np.mean([any(sentence in known for sentence in split_pair(text)) for text in official_texts])
print(f"Official validation pairs sharing a sentence with train: {shared:.3f}")


def predict(system: str, folder, fold: int, texts: list[str]) -> np.ndarray:
    """Predictions of one saved model."""
    path = f"{folder}/models/{system}/seed-0/fold-{fold}/model"
    if system.endswith("loglin"):
        return part1.loglin.predict(texts, part1.loglin.load(path), encoder)
    model_class = TwoLayerSparseMLP if system.endswith("mlp2") else part1.smlp.SparseMLP
    with patch.object(part1.smlp, "SparseMLP", model_class):
        model = part1.smlp.load(path)
    return part1.smlp.predict_probs(texts, model, encoder, batch_size=256).argmax(1)


folders = {"random": ROOT / "results" / "p1", "grouped": ROOT / "results" / "p1_full"}
for system in ("bigram+loglin", "bigram+mlp", "bigram+mlp2"):
    for name, folder in folders.items():
        accuracies = [float((predict(system, folder, fold, official_texts) == official_gold).mean()) for fold in range(5)]
        print(f"Official validation, {system} trained on {name} folds: {np.mean(accuracies):.4f}")

random_splits = part1.experiment.cross_validation_split(pairs, cfg.k, cfg.seeds[0])
grouped_splits = grouped_cross_validation_split(pairs, cfg.k, cfg.seeds[0])
for name, splits in (("Random", random_splits), ("Grouped", grouped_splits)):
    shares = []
    for fold in splits.values():
        seen = {sentence for text, _ in fold["training"] for sentence in split_pair(text)}
        shares.append(np.mean([any(s in seen for s in split_pair(text)) for text, _ in fold["validation"]]))
    print(f"{name} folds, validation pairs sharing a sentence with training: {np.mean(shares):.3f}")

for system in ("bigram+loglin", "bigram+mlp"):
    buckets = defaultdict(list)
    for fold_id, fold in random_splits.items():
        training_labels = defaultdict(set)
        for text, label in fold["training"]:
            for sentence in split_pair(text):
                training_labels[sentence].add(label)
        texts = [text for text, _ in fold["validation"]]
        predictions = predict(system, folders["random"], fold_id, texts)
        for (text, label), prediction in zip(fold["validation"], predictions):
            labels = set().union(*(training_labels.get(sentence, set()) for sentence in split_pair(text)))
            if not labels:
                bucket = "no shared sentence"
            elif labels == {label}:
                bucket = "training labels all agree"
            elif label not in labels:
                bucket = "training labels all disagree"
            else:
                bucket = "training labels mixed"
            buckets[bucket].append(prediction == label)
    for bucket, hits in sorted(buckets.items()):
        print(f"Random folds, {system}, {bucket}: n={len(hits)} accuracy={np.mean(hits):.4f}")
