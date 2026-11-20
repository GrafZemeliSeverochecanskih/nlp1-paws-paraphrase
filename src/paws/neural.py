"""P2 step 3-4: pair encoders on top of the tutorial's part2 encoders.

part2.experiment.train builds `clslib.EmbedAndPoolEncoder` or `clslib.LSTMEncoder` from `enc_type`.
`encoder_patch` swaps that class for a builder that can add a bidirectional LSTM and/or a siamese wrapper,
so the tutorial's training loop, best-epoch selection and saving stay unchanged (same trick as P1's SparseMLP patch).
"""

from contextlib import AbstractContextManager
from unittest.mock import patch

import torch
from torch import nn

from paws.tutorial import part2

ENCODER_CLASS_NAME = {"embed-and-pool": "EmbedAndPoolEncoder", "lstm": "LSTMEncoder"}
# Originals, captured before any patch, so builders always wrap the real tutorial classes.
ORIGINAL = {enc_type: getattr(part2.clslib, name) for enc_type, name in ENCODER_CLASS_NAME.items()}


class BiLSTMEncoder(ORIGINAL["lstm"]):
    """Tutorial LSTMEncoder with bidirectional=True (the pinned tutorial class has no such argument).

    The output concatenates the final forward state (after the last word) and the final backward state
    (after the first word), so both ends of the sentence reach the classifier; a one-directional final state
    is dominated by the sentence end, where a PAWS swap near the start can fade. enc_dim = 2 * hidden_size.
    Embedding, word dropout and packing are the tutorial's, unchanged.
    """

    def __init__(self, hidden_size, embedding_matrix, padding_idx=0, freeze_embeddings=True, word_dropout=0.0):
        super().__init__(hidden_size, embedding_matrix, padding_idx, freeze_embeddings, word_dropout)
        # Replaces (not adds to) the tutorial's one-directional LSTM, so no unused parameters are trained or saved.
        self.lstm = nn.LSTM(
            input_size=self.embedding.embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    @property
    def enc_dim(self) -> int:
        """Forward and backward states concatenated; read by SoftmaxRegressor and SiameseEncoder."""
        return 2 * self.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, len) ids -> (batch, 2 * hidden_size): tutorial forward, except the two final states are concatenated."""
        embedded = self.embedding(x)
        valid = x.ne(self.embedding.padding_idx)
        if self.training and self.word_dropout > 0:
            drop_mask = (torch.rand(x.shape, device=embedded.device) < self.word_dropout) & valid
            embedded = embedded.masked_fill(drop_mask.unsqueeze(-1), 0.0)

        lengths = torch.sum(valid, dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, (hidden, cell) = self.lstm(packed)
        # hidden: (2, batch, hidden_size); [-2] = forward direction, [-1] = backward direction
        return torch.cat([hidden[-2], hidden[-1]], dim=-1)


class SiameseEncoder(nn.Module):
    """Encodes sentence1 and sentence2 separately with one shared encoder, then combines u and v.

    Input: the PairTokenizer sequence ids(sentence1) + [sep_id] + ids(sentence2), padded with pad_idx.
    """

    def __init__(self, inner: nn.Module, sep_id: int) -> None:
        super().__init__()
        self.inner = inner
        self.sep_id = sep_id

    @property
    def pad_idx(self) -> int:
        """Needed by part2.experiment.predict_logprobs."""
        return self.inner.pad_idx

    @property
    def enc_dim(self) -> int:
        """Size of the vector `forward` returns ([|u - v|, u * v]: twice the inner size); read by SoftmaxRegressor."""
        return 2 * self.inner.enc_dim

    def split_at_sep(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(batch, len) ids -> (x1, x2): sentence1 and sentence2 ids, each re-padded with pad_idx.

        The SEP position differs per row (sentence1 lengths differ), so each row is sliced on its own.
        The SEP itself is dropped; trailing pads in x2 are harmless because both inner encoders ignore pad_idx.
        """
        sep_idx = (x == self.sep_id).nonzero()[:, 1]
        # Without exactly one SEP per row, zip() would pair rows with the wrong positions silently.
        assert len(sep_idx) == len(x), "every row needs exactly one SEP"

        x1 = nn.utils.rnn.pad_sequence([row[:sep] for row, sep in zip(x, sep_idx)], batch_first=True, padding_value=self.pad_idx)
        x2 = nn.utils.rnn.pad_sequence([row[sep + 1:] for row, sep in zip(x, sep_idx)], batch_first=True, padding_value=self.pad_idx)
        return x1, x2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode both sentences with the shared inner encoder and return [|u - v|, u * v], shape (batch, enc_dim).

        Symmetric in the two sentences (swapping them gives the same features), like the paraphrase relation,
        and the same pair features as P1, so P1 and P2 differ only in the sentence encoder.
        """
        x1, x2 = self.split_at_sep(x)
        u = self.inner(x1)
        v = self.inner(x2)
        return torch.cat([torch.abs(u - v), u * v], dim=-1)


def encoder_patch(enc_type: str, siamese: bool, sep_id: int, bidirectional: bool = False) -> AbstractContextManager:
    """Context manager under which part2.experiment.train builds the requested encoder for `enc_type`."""
    if bidirectional and enc_type != "lstm":
        raise ValueError("bidirectional only applies to enc_type='lstm'")
    base = BiLSTMEncoder if bidirectional else ORIGINAL[enc_type]

    def build(embedding_matrix: torch.Tensor, **enc_kwargs) -> nn.Module:
        inner = base(embedding_matrix=embedding_matrix, **enc_kwargs)
        return SiameseEncoder(inner, sep_id) if siamese else inner

    return patch.object(part2.clslib, ENCODER_CLASS_NAME[enc_type], build)
