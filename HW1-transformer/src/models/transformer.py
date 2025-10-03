import torch
import torch.nn as nn
from typing import Optional
from .encoder import EncoderLayer
from .decoder import Decoder
from .embeddings import PositionalEncoding

def make_length_mask(lengths: torch.Tensor, L: int) -> torch.Tensor:
    """
    Build a boolean padding mask from lengths.
    Args:
        lengths: int32/int64 [B] original (unpadded) lengths
        L: int, max length in the batch (padded length)
    Returns:
        mask: Bool [B, 1, 1, L] where True marks PAD positions
    """
    device = lengths.device
    idx = torch.arange(L, device=device)[None, :]           # [1, L]
    pad = idx >= lengths[:, None]                           # [B, L]
    return pad[:, None, None, :]                            # [B,1,1,L]

class Transformer(nn.Module):
    """
    Maps padded sensor sequences [B, L] (int32) to EE [B, 3] (float).
    - Projects scalar sensor value -> d_model
    - Adds sinusoidal PE
    - N pre-norm Transformer encoder layers
    - Masked mean pooling over time
    - MLP regression head to 3D
    """
    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        in_scale: float = 1.0,          # simple global scaling for raw ints
        use_layernorm_input: bool = True,
    ):
        super().__init__()
        self.in_scale = in_scale

        self.input_proj = nn.Linear(1, d_model)             # scalar -> d_model
        self.posenc = PositionalEncoding(d_model, dropout=dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout=dropout, activation=activation)
            for _ in range(num_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        self.head_seq = nn.Sequential(
            nn.LayerNorm(d_model) if use_layernorm_input else nn.Identity(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        self.head_global = nn.Sequential(
            nn.LayerNorm(d_model), 
            nn.Linear(d_model, d_model), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(d_model, 3))

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, sensor_pad: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sensor_pad: int32/float32 [B, L] (padded with zeros by pad_collate)
            lengths: int32 [B] original lengths
        Returns:
            pred: float32 [B, 3]
        """
        x = sensor_pad.float() * self.in_scale                # [B, L]
        x = x.unsqueeze(-1)                                   # [B, L, 1]
        x = self.input_proj(x)                                # [B, L, D]
        x = self.posenc(x)
        
        mask = make_length_mask(lengths.to(x.device), x.size(1))  # [B,1,1,L]
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        x = self.enc_norm(x)                                  # [B, L, D]

        # Masked mean pool over time
        B, L, D = x.shape
        valid = (~mask.squeeze(1).squeeze(1)).float()         # [B, L] 1=keep, 0=pad
        denom = torch.clamp(valid.sum(dim=1, keepdim=True), min=1.0)  # [B,1]
        pooled = (x * valid.unsqueeze(-1)).sum(dim=1) / denom # [B, D]
        
        global_pred = self.head_global(pooled)  # [B, 3]

        return global_pred                             
