"""Pair encoders compatible with part1.clslib.Encoder: token indicators, bag-of-words, bigram (n-gram) features."""

from collections import Counter

import numpy as np
from tokenizers import Tokenizer

from paws.data import split_pair
from paws.tutorial import part1


class PairEncoder(part1.clslib.Encoder):
    """Encodes each sentence to v1, v2 and returns [|v1 - v2|, v1 * v2]: what differs and what is shared.

    Both parts are symmetric, so swapping the sentences does not change the features. A linear model on a plain
    concatenation could not express similarity, since it scores each sentence independently.
    Subclasses set `self.dim` and implement `sentence_vector`. Vectors are float32 because the tutorial stacks
    them densely before converting to sparse, and float64 would double that memory.
    """

    def __len__(self) -> int:
        return 2 * self.dim

    def __call__(self, text: str) -> np.ndarray:
        v1, v2 = (self.sentence_vector(sentence) for sentence in split_pair(text))
        return np.concatenate([np.abs(v1 - v2), v1 * v2])


class PairBagOfWords(PairEncoder):
    """Counts of BPE tokens per sentence."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.dim = tokenizer.get_vocab_size()

    def sentence_vector(self, sentence: str) -> np.ndarray:
        return np.bincount(self.tokenizer.encode(sentence).ids, minlength=self.dim).astype(np.float32)


class PairTokenIndicator(PairBagOfWords):
    """Presence of BPE tokens per sentence. Written here because the tutorial's __len__ reads a global tokenizer."""

    def sentence_vector(self, sentence: str) -> np.ndarray:
        return (super().sentence_vector(sentence) > 0).astype(np.float32)


class PairBigram(PairEncoder):
    """Counts of the most frequent adjacent BPE token pairs: unlike unigrams, these see local word order."""

    def __init__(self, tokenizer: Tokenizer, corpus: list[str], max_bigrams: int) -> None:
        self.tokenizer = tokenizer
        counts = Counter(bigram for sentence in corpus for bigram in self._bigrams(sentence))
        self.index = {bigram: i for i, (bigram, _) in enumerate(counts.most_common(max_bigrams))}
        self.dim = len(self.index)

    def _bigrams(self, sentence: str) -> zip:
        ids = self.tokenizer.encode(sentence).ids
        return zip(ids, ids[1:])

    def sentence_vector(self, sentence: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for bigram in self._bigrams(sentence):
            if bigram in self.index:
                vector[self.index[bigram]] += 1
        return vector
