"""P2 step 1-2: load GloVe, tokenize pairs into GloVe ids, measure vocabulary coverage."""

from collections import Counter

import nltk
import numpy as np

from paws.config import P2Config
from paws.data import split_pair
from paws.tutorial import part2

SEP_TOKEN = "[SEP]"  # string form of the separator, for tokens(); ids() uses PairTokenizer.sep_id


def load_embeddings(cfg: P2Config) -> part2.clslib.PretrainedEmbeddings:
    """GloVe table with id 0 = <pad> and id 1 = <unk> (the mean vector), as built by the tutorial class."""
    # str(): the tutorial loader picks the file format with path.endswith(...)
    return part2.clslib.PretrainedEmbeddings(str(cfg.glove_path), cfg.emb_dim, skip_first_line=cfg.glove_skip_first_line)


class PairTokenizer(part2.clslib.Tokenizer):
    """Turns a SEP-joined pair (see data.SEP) into one id sequence: ids(sentence1) + [sep_id] + ids(sentence2).

    Cross-encoding models read the whole sequence; siamese models split it again at `sep_id` (neural.SiameseEncoder).
    """

    def __init__(self, cfg: P2Config, emb: part2.clslib.PretrainedEmbeddings) -> None:
        nltk.download("punkt_tab", quiet=True)  # needed by BasicTokenizer's nltk.word_tokenize
        self.emb = emb
        self.words = part2.clslib.BasicTokenizer(cfg.lowercase, cfg.remove_punct, False, "english", emb.vocab)
        # Separator = a new row after the last GloVe word: no word can map to it, so the siamese split is exact.
        # The vocab is frozen, so the row is added to the vectors only (vocab.word(sep_id) returns <unk>).
        self.sep_id: int = len(emb.vocab)
        if len(emb.vectors) == len(emb.vocab):  # add the row once, even if several tokenizers share `emb`
            # Random, not zeros: zeros is <pad>, and with frozen embeddings the vector never changes.
            # Same scale as GloVe so it is neither ignored nor dominant; seeded for reproducibility.
            rng = np.random.default_rng(cfg.seeds[0])
            sep_vector = rng.normal(0.0, emb.vectors[2:].std(), size=(1, emb.vectors.shape[1])).astype(np.float32)
            emb.vectors = np.vstack([emb.vectors, sep_vector])

    def tokens(self, text: str) -> list[str]:
        """Word tokens of both sentences with a separator token between them."""
        sentence1, sentence2 = split_pair(text)
        return self.words.tokens(sentence1) + [SEP_TOKEN] + self.words.tokens(sentence2)

    def ids(self, text: str) -> list[int]:
        """GloVe ids of both sentences with `sep_id` between them; unknown words map to <unk>."""
        sentence1, sentence2 = split_pair(text)
        return self.words.ids(sentence1) + [self.sep_id] + self.words.ids(sentence2)


def coverage(tokenizer: PairTokenizer, emb: part2.clslib.PretrainedEmbeddings, sentences: list[str]) -> dict:
    """Coverage of `sentences` by the GloVe vocabulary, as a JSON-safe dict.

    A token is OOV when the tokenizer maps it to <unk>, so the numbers reflect the real pipeline (incl. lowercasing).
    Capitalised = starts with an upper-case letter in the original text: a cheap proxy for named entities.
    """
    cased = part2.clslib.BasicTokenizer(False, tokenizer.words.remove_punct, False, "english")
    unk = emb.vocab.default_index
    counts, oov_counts = Counter(), Counter()
    capitalised, capitalised_oov, sentences_with_oov = 0, 0, 0
    for sentence in sentences:
        has_oov = False
        for token in cased.tokens(sentence):
            looked_up = token.lower() if tokenizer.words.lowercase else token
            is_oov = emb.vocab.index(looked_up) == unk
            counts[looked_up] += 1
            oov_counts[looked_up] += is_oov
            if token[:1].isupper():
                capitalised += 1
                capitalised_oov += is_oov
            has_oov |= is_oov
        sentences_with_oov += has_oov

    oov_counts = +oov_counts  # drop zero counts
    num_tokens = sum(counts.values())
    return {
        "sentences": len(sentences),
        "tokens": num_tokens,
        "types": len(counts),
        "token_oov_rate": sum(oov_counts.values()) / num_tokens,
        "type_oov_rate": len(oov_counts) / len(counts),
        "capitalised_token_oov_rate": capitalised_oov / max(capitalised, 1),
        "other_token_oov_rate": (sum(oov_counts.values()) - capitalised_oov) / max(num_tokens - capitalised, 1),
        "sentences_with_oov": sentences_with_oov / len(sentences),
        "top_oov": oov_counts.most_common(30),
    }
