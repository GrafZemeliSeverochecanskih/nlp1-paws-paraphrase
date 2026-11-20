"""P2 step 6: per-pair features that may explain which pairs a system gets right (input to part2.analysis.fit_bda_model)."""

import math

import scipy.stats

from paws.data import split_pair
from paws.embeddings import PairTokenizer

# Order fixes the columns of the feature matrix.
FEATURE_NAMES = ["same_word_set", "word_overlap", "order_change", "oov_rate", "num_capitalised", "length"]


def pair_features(text: str, tokenizer: PairTokenizer) -> dict[str, float]:
    """Features of one SEP-joined pair, keyed by FEATURE_NAMES; all values are finite floats.

    Word features use the model's own tokens (tokenizer.words: lowercased NLTK tokens, punctuation kept).
        same_word_set: 1.0 if both sentences use the same set of words (pooling cannot separate these)
        word_overlap: Jaccard overlap of the word sets (as in data.split_stats)
        order_change: 1 - Kendall tau of the first positions of the shared words (0 = same order, 2 = reversed)
        oov_rate: share of the pair's tokens that are <unk> in GloVe (SEP excluded)
        num_capitalised: capitalised words not at sentence start, from the cased text (a cheap entity proxy)
        length: tokens in both sentences
    """
    sentence1, sentence2 = split_pair(text)
    w1 = tokenizer.words.tokens(sentence1)
    w2 = tokenizer.words.tokens(sentence2)

    s1 = set(w1)
    s2 = set(w2)
    same_word_set = 1 if s1 == s2 else 0
    word_overlap = len(s1 & s2) / len(s1 | s2)

    intersect = s1.intersection(s2)
    first_occurrence = dict()
    for w in intersect:
        first_occurrence[w] = [w1.index(w), w2.index(w)]

    ordered = sorted(first_occurrence.values())
    pos1 = [p1 for p1, p2 in ordered]
    pos2 = [p2 for p1, p2 in ordered]

    if len(pos1) < 2:
        order_change = 0
    else:
        tau = scipy.stats.kendalltau(pos1, pos2).statistic
        if math.isnan(tau):
            order_change = 0
        else:
            order_change = 1 - tau

    ids = [i for i in tokenizer.ids(text) if i != tokenizer.sep_id]
    oov_rate = sum(i == tokenizer.emb.vocab.default_index for i in ids) / len(ids)

    # From the original text: w1/w2 are lowercased. Skips each sentence's first word.
    num_capitalised = sum(word[:1].isupper() for s in (sentence1, sentence2) for word in s.split()[1:])

    return {
        "same_word_set": float(same_word_set),
        "word_overlap": float(word_overlap),
        "order_change": float(order_change),
        "oov_rate": float(oov_rate),
        "num_capitalised": float(num_capitalised),
        "length": float(len(w1) + len(w2)),
    }
