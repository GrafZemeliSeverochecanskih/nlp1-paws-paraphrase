"""Load PAWS, convert sentence pairs to the tutorial's (text, label) format and back, compute dataset statistics."""

import random
from collections import Counter

from datasets import load_dataset

from paws.config import Config
from paws.tutorial import part1

SEP = " [SEP] "
OUTCOMES = part1.clslib.NamedClasses(["different", "paraphrase"])


def split_pair(text: str) -> tuple[str, str]:
    """Inverse of the SEP join; fails loudly if a text does not contain exactly one SEP."""
    sentence1, sentence2 = text.split(SEP)
    return sentence1, sentence2


def load_split(cfg: Config, split: str) -> list[dict]:
    """Rows of a PAWS split as dicts with `sentence1`, `sentence2`, `label`."""
    return list(load_dataset(cfg.dataset_name, cfg.dataset_config, cache_dir=str(cfg.data_dir))[split])


def load_pairs(cfg: Config) -> list[tuple[str, int]]:
    """Train pairs as (text, label) tuples, subsampled if configured, so tutorial functions work unchanged."""
    pairs = [(row["sentence1"] + SEP + row["sentence2"], row["label"]) for row in load_split(cfg, "train")]
    if cfg.subsample_size is not None and cfg.subsample_size < len(pairs):
        pairs = random.Random(cfg.seeds[0]).sample(pairs, cfg.subsample_size)
    return pairs


def train_sentences(cfg: Config) -> list[str]:
    """Distinct train sentences, in first-seen order: the corpus for BPE and the bigram vocabulary.

    Distinct because PAWS reuses sentences across pairs, which would over-count their tokens.
    """
    rows = load_split(cfg, "train")
    return list(dict.fromkeys(sentence for row in rows for sentence in (row["sentence1"], row["sentence2"])))


def split_stats(rows: list[dict]) -> dict:
    """Size, label balance, lengths and word overlap per label for one split."""
    sentences = Counter(sentence for row in rows for sentence in (row["sentence1"], row["sentence2"]))
    stats = {
        "pairs": len(rows),
        "paraphrase_share": sum(row["label"] for row in rows) / len(rows),
        "distinct_sentences": len(sentences),
        # A sentence seen more than once across pairs links those pairs; random CV can split them across folds.
        "pairs_sharing_a_sentence": sum(
            sentences[row["sentence1"]] > 1 or sentences[row["sentence2"]] > 1 for row in rows
        )
        / len(rows),
        "mean_words_per_sentence": sum(len(sentence.split()) * n for sentence, n in sentences.items())
        / sum(sentences.values()),
    }
    for label, name in OUTCOMES.items():
        word_sets = [
            (set(row["sentence1"].split()), set(row["sentence2"].split())) for row in rows if row["label"] == label
        ]
        stats[f"{name}_mean_word_overlap"] = sum(len(a & b) / len(a | b) for a, b in word_sets) / len(word_sets)
        stats[f"{name}_same_word_set_share"] = sum(a == b for a, b in word_sets) / len(word_sets)
    return stats
