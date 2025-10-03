from __future__ import annotations
from typing import Optional

import torch

def make_padding_mask(seq: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Create a boolean padding mask where True marks positions to be masked out.

    Args:
        seq: LongTensor [B, L] token ids
        pad_id: int, padding token id
    Returns:
        mask: BoolTensor [B, 1, 1, L]  suitable for attention broadcasting
              True = masked (ignore), False = keep
    """
    mask = (seq == pad_id)  # [B, L]
    return mask[:, None, None, :]  # [B, 1, 1, L]


def make_causal_mask(L: int, device=None, dtype=torch.bool) -> torch.Tensor:
    """
    Standard causal mask for auto-regressive decoding.
    Returns BoolTensor [1, 1, L, L] where True = masked (future positions).
    """
    i = torch.arange(L, device=device)
    j = torch.arange(L, device=device)
    causal = (j[None, :] > i[:, None])  # [L, L]
    return causal[None, None, :, :].to(dtype)


def combine_masks(*masks: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    OR-combine a list of masks. Any None is ignored.
    Each mask must be broadcastable to the others (e.g., [B,1,1,L], [1,1,L,L]).
    True means "mask out".
    """
    masks = [m for m in masks if m is not None]
    if not masks:
        return None
    out = masks[0]
    for m in masks[1:]:
        out = out | m
    return out