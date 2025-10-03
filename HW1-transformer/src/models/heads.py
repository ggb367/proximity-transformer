import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Scaled Dot-Product Attention.
    - Supports self-attn (Q=K=V) and cross-attn (Q from x, K/V from memory).
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor, B: int) -> torch.Tensor:
        # [B, L, D] -> [B, H, L, d_k]
        return x.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        q: torch.Tensor,           # [B, Lq, D]
        k: torch.Tensor,           # [B, Lk, D]
        v: torch.Tensor,           # [B, Lv, D]
        attn_mask: Optional[torch.Tensor] = None,  # broadcastable, True = mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Lq, D = q.size()
        Lk = k.size(1)

        q_proj = self._shape(self.W_q(q), B)  # [B,H,Lq,d_k]
        k_proj = self._shape(self.W_k(k), B)  # [B,H,Lk,d_k]
        v_proj = self._shape(self.W_v(v), B)  # [B,H,Lk,d_k]

        # Scaled dot-product attention
        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B,H,Lq,Lk]

        if attn_mask is not None:
            # attn_mask True => mask out => set to -inf
            scores = scores.masked_fill(attn_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)  # [B,H,Lq,Lk]
        attn = self.dropout(attn)

        ctx = torch.matmul(attn, v_proj)  # [B,H,Lq,d_k]
        ctx = ctx.transpose(1, 2).contiguous().view(B, Lq, D)  # [B,Lq,D]
        out = self.W_o(ctx)  # [B,Lq,D]
        return out, attn


class PositionwiseFFN(nn.Module):
    """
    Feed-forward network: Linear -> ReLU -> Dropout -> Linear
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, activation: str = "relu"):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        if activation == "relu":
            self.act = F.relu
        elif activation.lower() == "gelu":
            self.act = F.gelu
        else:
            raise ValueError("activation must be 'relu' or 'gelu'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.dropout(self.act(self.lin1(x))))


class PreNormResidual(nn.Module):
    """
    Pre-norm residual block: y = x + sublayer(LayerNorm(x))
    """
    def __init__(self, d_model: int, sublayer: nn.Module, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.sublayer = sublayer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        y = self.sublayer(self.norm(x), *args, **kwargs)
        return x + self.dropout(y)