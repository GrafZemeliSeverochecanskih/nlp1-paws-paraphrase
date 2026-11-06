"""Train, save and load the BPE tokenizer (wraps part1.experiment.train_tokenizer)."""

from tokenizers import Tokenizer

from paws.config import Config
from paws.tutorial import part1


def train_and_save(corpus: list[str], cfg: Config) -> Tokenizer:
    """Train BPE on `corpus` and save it to `cfg.tokenizer_path`."""
    tokenizer = part1.experiment.train_tokenizer(corpus, cfg.vocab_size, cfg.min_frequency)
    cfg.tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(cfg.tokenizer_path))
    return tokenizer


def load(cfg: Config) -> Tokenizer:
    """Load the tokenizer saved by `train_and_save`."""
    return Tokenizer.from_file(str(cfg.tokenizer_path))
