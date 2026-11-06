"""Grouped cross-validation: pairs linked by a shared sentence stay in one fold, so no sentence crosses folds."""

import random
from collections import defaultdict

from paws.data import split_pair


def grouped_cross_validation_split(pairs: list[tuple[str, int]], k: int, seed: int = 0) -> dict[int, dict]:
    """Same output format as part1.experiment.cross_validation_split, with groups assigned whole to folds."""
    parent = {}

    def find(sentence: str) -> str:
        parent.setdefault(sentence, sentence)
        while parent[sentence] != sentence:
            parent[sentence] = parent[parent[sentence]]
            sentence = parent[sentence]
        return sentence

    sentences = [split_pair(text) for text, _ in pairs]
    for sentence1, sentence2 in sentences:
        parent[find(sentence1)] = find(sentence2)

    groups = defaultdict(list)
    for i, (sentence1, _) in enumerate(sentences):
        groups[find(sentence1)].append(i)

    shuffled = list(groups.values())
    random.Random(seed).shuffle(shuffled)
    folds = [[] for _ in range(k)]
    for group in shuffled:
        min(folds, key=len).extend(group)

    splits = {}
    for i in range(k):
        training_ids = [j for f, fold in enumerate(folds) if f != i for j in fold]
        splits[i] = {
            "training": [pairs[j] for j in training_ids],
            "validation": [pairs[j] for j in folds[i]],
            "training_ids": training_ids,
            "validation_ids": folds[i],
        }
    return splits


def assert_no_shared_sentences(splits: dict[int, dict], num_pairs: int) -> None:
    """Fail before training if a sentence crosses a fold boundary or a pair is lost or validated twice."""
    assert sorted(j for fold in splits.values() for j in fold["validation_ids"]) == list(range(num_pairs))
    for fold in splits.values():
        training = {sentence for text, _ in fold["training"] for sentence in split_pair(text)}
        validation = {sentence for text, _ in fold["validation"] for sentence in split_pair(text)}
        assert not training & validation
