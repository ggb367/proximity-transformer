import torch
import torch.nn as nn
import math
from typing import Optional
from .heads import MultiHeadAttention, PositionwiseFFN, PreNormResidual
from .embeddings import PositionalEncoding
from ..utils.utils import make_padding_mask


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout=dropout, activation=activation)

        self.block1 = PreNormResidual(d_model, sublayer=self._self_attn, dropout=dropout)
        self.block2 = PreNormResidual(d_model, sublayer=self.ffn, dropout=dropout)

    def _self_attn(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out, _ = self.self_attn(x, x, x, attn_mask=mask)
        return out

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.block1(x, mask=src_mask)
        x = self.block2(x)
        return x


class Encoder(nn.Module):
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
            EncoderLayer(d_model, num_heads, d_ff, dropout=dropout, activation=activation)
        for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

        if tie_embedding:
            # allows sharing with a decoder projection if attached later
            self.embed.weight.requires_grad = True

    def forward(self, src_tokens: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src_tokens: LongTensor [B, Ls]
            src_mask: BoolTensor [B,1,1,Ls] (True=mask), optional.
                      If None, constructed from pad_id.
        Returns:
            memory: FloatTensor [B, Ls, D]
        """
        if src_mask is None:
            src_mask = make_padding_mask(src_tokens, self.pad_id)

        x = self.embed(src_tokens) * math.sqrt(self.embed.embedding_dim)
        x = self.posenc(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
