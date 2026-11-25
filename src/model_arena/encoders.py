"""Ordered-history encoders sharing a horizon-context contract with the TFT.

Inputs are already projected to width. Outputs are [batch, horizon, width].
These alternatives contain no TFT attention or LSTM layers.
"""
from torch import nn


class ContextEncoder(nn.Module):
    def __init__(self, *, kind, window, width, depth, dropout, activation, targets):
        super().__init__()
        self.input = nn.Linear(targets, width)
        if kind == 'mlp':
            layers = [nn.Flatten(1), nn.Linear(window*width, width), activation(), nn.Dropout(dropout)]
            for _ in range(depth-1):
                layers.extend([nn.Linear(width, width), activation(), nn.Dropout(dropout)])
            self.history = nn.Sequential(*layers)
        elif kind == 'gru':
            self.history = nn.GRU(width, width, depth, batch_first=True,
                                  dropout=dropout if depth > 1 else 0.)
        else:
            raise ValueError('ContextEncoder supports mlp or gru')
        self.kind = kind
        self.context = nn.Sequential(nn.Linear(width*2, width), activation(), nn.Dropout(dropout))

    def forward(self, history, queries):
        if self.kind == 'gru':
            _, hidden = self.history(history)
            state = hidden[-1]
        else:
            state = self.history(history)
        from torch import cat
        return self.context(cat((state[:, None].expand(-1, queries.shape[1], -1), queries), dim=-1))
