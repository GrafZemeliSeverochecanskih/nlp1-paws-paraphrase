"""Third architecture: a two-hidden-layer sparse MLP."""

from torch import nn

from paws.tutorial import part1


class TwoLayerSparseMLP(part1.smlp.SparseMLP):
    """SparseMLP plus a second hidden layer of half the first's size. Sparse input layer and forward are inherited."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__(input_dim, hidden_dim, num_classes, dropout)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
            nn.LogSoftmax(dim=-1),
        )
