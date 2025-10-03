import torch
import torch.nn as nn
import math
from typing import Optional
from .heads import MultiHeadAttention, PositionwiseFFN, PreNormResidual
from .embeddings import PositionalEncoding
from ..utils.utils import make_padding_mask, make_causal_mask, combine_masks

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout=dropout, activation=activation)

        self.block1 = PreNormResidual(d_model, sublayer=self._self_attn, dropout=dropout)
        self.block2 = PreNormResidual(d_model, sublayer=self._cross_attn, dropout=dropout)
        self.block3 = PreNormResidual(d_model, sublayer=self.ffn, dropout=dropout)

    def _self_attn(self, x: torch.Tensor, self_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out, _ = self.self_attn(x, x, x, attn_mask=self_mask)
        return out

    def _cross_attn(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out, _ = self.cross_attn(x, memory, memory, attn_mask=memory_mask)
        return out

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.block1(x, self_mask)
        x = self.block2(x, memory, memory_mask)
        x = self.block3(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        pad_id: int,
        max_len: int = 10_000,
        dropout: float = 0.1,
        activation: str = "relu",
        tie_embedding: bool = False,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.posenc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout=dropout, activation=activation)
        for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

        self.tie_embedding = tie_embedding
        # If tying embeddings with a final projection externally, set tie later.

    def forward(
        self,
        tgt_tokens: torch.Tensor,           # [B, Lt]
        memory: torch.Tensor,               # [B, Ls, D]
        tgt_mask: Optional[torch.Tensor] = None,     # [B,1,Lt,Lt] or broadcastable
        memory_mask: Optional[torch.Tensor] = None,  # [B,1,Lt,Ls] or broadcastable
    ) -> torch.Tensor:
        if tgt_mask is None:
            pad_mask = make_padding_mask(tgt_tokens, self.pad_id)               # [B,1,1,Lt]
            causal = make_causal_mask(tgt_tokens.size(1), device=tgt_tokens.device)  # [1,1,Lt,Lt]
            tgt_mask = combine_masks(pad_mask, causal)

        x = self.embed(tgt_tokens) * math.sqrt(self.embed.embedding_dim)
        x = self.posenc(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.norm(x)
