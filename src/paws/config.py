"""Central settings: paths, seeds, number of folds, tokenizer and model hyperparameters, ROPE size."""

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


class Config:
    """All P1 settings. Every argument has a default, so scripts use `Config()` and override only what changes."""

    def __init__(
        self,
        root: Path = ROOT,
        data_dir: Path = ROOT / "data",
        results_dir: Path = ROOT / "results" / "p1",
        tokenizer_path: Path = ROOT / "results" / "p1" / "tokenizer.json",
        dataset_name: str = "google-research-datasets/paws",
        dataset_config: str = "labeled_final",
        subsample_size: int | None = 20000,
        seeds: tuple[int, ...] = (0,),
        k: int = 5,
        vocab_size: int = 1000,
        min_frequency: int = 2,
        max_bigrams: int = 5000,
        loglin_max_iter: int = 10000,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 20,
        device: str = "cpu",
        rope: float = 0.01,
    ) -> None:
        """Store all P1 settings.

        Args:
            root: Repo root, derived from this file's location so scripts work from any working directory.
            subsample_size: Train pairs used for CV; 20,000 limits the dense vectors the tutorial stacks in memory.
                None uses all 49,401.
            seeds: One full CV run per seed; BenavoliCorrelatedT needs runs=len(seeds).
            k: Number of CV folds; same as the tutorial.
            vocab_size: BPE vocabulary size; the tutorial's value, small because vectors are stacked densely.
            min_frequency: Minimum count for a BPE merge; 2 drops merges seen only once (noise).
            max_bigrams: Number of most frequent BPE bigrams (counted on train only); caps feature dim and memory.
            loglin_max_iter: Passed to part1.loglin.fit; the tutorial's 100 and also 1000 did not converge on PAWS.
            hidden_dim: First hidden size of both MLPs; the two-layer MLP adds a second layer of half this size.
            device: "cpu" because the sparse MLPs are small; use "cuda" if training gets slow.
            rope: Half-width of the ROPE on accuracy. 0.01 is about one fold's standard error (n=4,000, p=0.5: 0.008),
                so smaller differences are within what one validation fold can resolve.

        The MLP values are the defaults of part1.smlp.train_neural_network_model, so the one-layer MLP
        matches the course baseline and the two-layer MLP differs only in depth.
        """
        self.root = root
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.tokenizer_path = tokenizer_path
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.subsample_size = subsample_size
        self.seeds = seeds
        self.k = k
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.max_bigrams = max_bigrams
        self.loglin_max_iter = loglin_max_iter
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.device = device
        self.rope = rope

    def loglin_args(self) -> dict:
        """Keyword arguments for `cross_validation_experiment(..., loglin_args=...)`."""
        return {"max_iter": self.loglin_max_iter}

    def smlp_args(self) -> dict:
        """Keyword arguments for `cross_validation_experiment(..., smlp_args=...)`.

        Keys must match `part1.smlp.train_neural_network_model` exactly; they are passed as **kwargs.
        """
        return {
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_epochs": self.num_epochs,
            "device": self.device,
        }

    def to_dict(self) -> dict:
        """JSON-safe copy of all settings, saved next to every result for reproducibility."""
        return {key: str(value) if isinstance(value, Path) else value for key, value in vars(self).items()}

    def make_dirs(self) -> None:
        """Create the output folders. Called by scripts, not on import."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


class P2Config(Config):
    """P2 settings. Inherits P1's data, fold and seed settings unchanged, so P2 folds equal P1's grouped folds
    and P1 systems can be compared with P2 systems fold by fold.
    """

    def __init__(
        self,
        results_dir: Path = ROOT / "results" / "p2",
        glove_path: Path = ROOT / "data" / "glove.2024.wikigiga.50d.zip",
        emb_dim: int = 50,
        glove_skip_first_line: bool = False,
        lowercase: bool = True,
        remove_punct: bool = False,
        lstm_hidden: int = 128,
        regressor_hidden: int = 50,
        freeze_embeddings: bool = True,
        **kwargs,
    ) -> None:
        """Store P2 settings; remaining keyword arguments go to Config (k, seeds, subsample_size, num_epochs, ...).

        Args:
            glove_path: Zipped GloVe file, read directly by part2.clslib.PretrainedEmbeddings (no unzipping).
            emb_dim, glove_skip_first_line: Must match the file; values for GloVe 2024 50d from the tutorial.
            lowercase: Required: the GloVe vocab is all lowercase and BasicTokenizer.ids does not lowercase,
                so lowercase=False maps every capitalised word to <unk>.
            remove_punct: False. True lowers token OOV only from 0.97% to 0.75% (it fixes hyphenated compounds;
                rare names stay OOV either way) but deletes 13% of tokens, incl. the commas that mark the clause
                boundaries PAWS reorders, which the LSTM can use (scripts/08_embedding_coverage.py).
            lstm_hidden: LSTM state size.
            regressor_hidden: Both hidden layers of part2.clslib.SoftmaxRegressor (tutorial default 50).
            freeze_embeddings: Keep GloVe fixed; set False for the trainable-embeddings experiment.
        """
        kwargs.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(results_dir=results_dir, **kwargs)
        self.glove_path = glove_path
        self.emb_dim = emb_dim
        self.glove_skip_first_line = glove_skip_first_line
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.lstm_hidden = lstm_hidden
        self.regressor_hidden = regressor_hidden
        self.freeze_embeddings = freeze_embeddings

    def regressor_kwargs(self) -> dict:
        """Keyword arguments for part2.clslib.SoftmaxRegressor (minus encoder and num_classes, added by part2)."""
        return {"hidden_size1": self.regressor_hidden, "hidden_size2": self.regressor_hidden, "dropout": self.dropout}

    def opt_kwargs(self) -> dict:
        """Keyword arguments for part2.experiment.train(opt_kwargs=...); keys must match it exactly."""
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
        }


# Embedding alternatives for `--glove` (name -> file, dims). The default is GloVe 2024 50d; the 6B files are GloVe 2014
# Wikipedia+Gigaword (400k words), extracted from glove.6B.zip because part2 reads only single-file zips.
EMBEDDINGS = {
    "2024-50d": (ROOT / "data" / "glove.2024.wikigiga.50d.zip", 50),
    "6B-50d": (ROOT / "data" / "glove6B" / "glove.6B.50d.txt", 50),
    "6B-100d": (ROOT / "data" / "glove6B" / "glove.6B.100d.txt", 100),
    "6B-300d": (ROOT / "data" / "glove6B" / "glove.6B.300d.txt", 300),
}


def variant_parser() -> argparse.ArgumentParser:
    """CLI options shared by the P2 scripts to pick a run other than the default (use as a parent parser)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--k", type=int, help="number of folds (default: P1's)")
    parser.add_argument("--seeds", type=int, nargs="+", help="one CV run per seed (default: P1's)")
    parser.add_argument("--subsample", type=int, help="train pairs (default: P1's); for the learning curve")
    parser.add_argument("--glove", choices=list(EMBEDDINGS), help="embedding file (default: 2024-50d)")
    return parser


def p2_variant(args: argparse.Namespace) -> P2Config:
    """P2Config with the options of `variant_parser`. Any non-default value gets its own results folder
    (e.g. results/p2_seeds0-1-2, results/p2_n5000, results/p2_glove6B-100d), so runs never mix in one table.
    """
    default = P2Config()
    overrides = {"k": args.k, "seeds": tuple(args.seeds) if args.seeds else None, "subsample_size": args.subsample}
    changed = {key: value for key, value in overrides.items() if value is not None and value != getattr(default, key)}
    tags = [{"k": f"k{args.k}", "seeds": "seeds" + "-".join(map(str, args.seeds or [])), "subsample_size": f"n{args.subsample}"}[key] for key in changed]
    if getattr(args, "glove", None) and EMBEDDINGS[args.glove][0] != default.glove_path:
        changed["glove_path"], changed["emb_dim"] = EMBEDDINGS[args.glove]
        tags.append(f"glove{args.glove}")
    if not changed:
        return default
    return P2Config(results_dir=ROOT / "results" / "_".join(["p2"] + tags), **changed)
