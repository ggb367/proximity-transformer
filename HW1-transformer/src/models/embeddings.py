import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding (precomputed).
    Adds PE to input embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 10_000, dropout: float = 0.0):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [L, D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe[None, ...])  # [1, L, D]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        return self.dropout(x)