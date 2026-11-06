"""Metrics (accuracy, paraphrase-class F1) and the majority-class baseline."""

from collections import Counter

from sklearn.metrics import accuracy_score, f1_score


def fold_f1(fold: dict, predictions) -> float:
    """F1 of the paraphrase class on a fold's validation set."""
    return f1_score([label for _, label in fold["validation"]], predictions, pos_label=1)


def majority_scores(fold: dict) -> tuple[float, float]:
    """(accuracy, F1) of always predicting the most common label of the fold's training set."""
    majority = Counter(label for _, label in fold["training"]).most_common(1)[0][0]
    gold = [label for _, label in fold["validation"]]
    predictions = [majority] * len(gold)
    return accuracy_score(gold, predictions), f1_score(gold, predictions, pos_label=1, zero_division=0)
