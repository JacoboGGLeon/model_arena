import torch
from torch import nn


class GatedResidualNetwork(nn.Module):
    """Compact TFT gated residual block with identity-capable skip path."""

    def __init__(self, size: int, dropout: float) -> None:
        super().__init__()
        self.activation = nn.ELU()
        self.dense_in = nn.Linear(size, size)
        self.dense_out = nn.Linear(size, size)
        self.gate = nn.Linear(size, 2 * size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(size)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.dense_out(self.activation(self.dense_in(value)))
        hidden = self.dropout(hidden)
        gated, control = self.gate(hidden).chunk(2, dim=-1)
        return self.norm(value + gated * torch.sigmoid(control))


class TFTTemporalBackbone(nn.Module):
    """Parallel TFT backbone: LSTM encoding, GRNs and temporal attention.

    It deliberately emits all H queries together. It is therefore a Family 6
    backbone even though its historical encoder is recurrent internally.
    """

    def __init__(
        self, hidden_size: int, dropout: float, *, attention_heads: int = 4
    ) -> None:
        super().__init__()
        if hidden_size % attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        self.input = nn.Linear(1, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.history_grn = GatedResidualNetwork(hidden_size, dropout)
        self.state_grn = GatedResidualNetwork(hidden_size, dropout)
        self.query_grn = GatedResidualNetwork(hidden_size, dropout)
        self.attention = nn.MultiheadAttention(
            hidden_size, attention_heads, dropout=dropout, batch_first=True
        )
        self.attention_gate = nn.Linear(hidden_size, 2 * hidden_size)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.output_grn = GatedResidualNetwork(hidden_size, dropout)

    def forward(
        self, history_y: torch.Tensor, future_queries: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, (hidden, _) = self.encoder(self.input(history_y))
        encoded = self.history_grn(encoded)
        state = self.state_grn(hidden[-1])
        queries = self.query_grn(future_queries + state[:, None, :])
        attended, _ = self.attention(queries, encoded, encoded, need_weights=False)
        candidate, control = self.attention_gate(attended).chunk(2, dim=-1)
        fused = self.attention_norm(queries + candidate * torch.sigmoid(control))
        return state, self.output_grn(fused)
