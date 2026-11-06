"""Central settings: paths, seeds, number of folds, tokenizer and model hyperparameters, ROPE size."""

from pathlib import Path

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
