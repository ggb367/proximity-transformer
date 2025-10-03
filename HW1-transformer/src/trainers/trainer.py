from __future__ import annotations
import math
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader,
        val_loader=None,
        device: str = "cuda",
        amp: bool = True,
        grad_clip: float = 1.0,
        ckpt_dir: Optional[str] = None,
        scheduler: Optional[Any] = None,
        log_every: int = 100,
        huber_beta: float = 0.01,   # SmoothL1 beta
    ):
        self.model = model.to(device)
        self.optim = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.amp = amp and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.amp)
        self.grad_clip = grad_clip
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.scheduler = scheduler
        self.log_every = log_every

        # configure criterion but we’ll do masking/weighting manually
        self.criterion = nn.SmoothL1Loss(beta=huber_beta, reduction="none")

        if self.ckpt_dir:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _build_mask_from_lengths(lengths: torch.Tensor, L: int) -> torch.Tensor:
        """
        Returns Bool mask [B, L] where True = valid (keep), False = pad.
        """
        idx = torch.arange(L, device=lengths.device)[None, :]
        return idx < lengths[:, None]

    def _compute_loss(
        self,
        pred: torch.Tensor,      # [B,3] or [B,L,3]
        target: torch.Tensor,    # [B,3] or [B,L,3]
        lengths: torch.Tensor,   # [B]
    ) -> Tuple[torch.Tensor, int]:
        """
        Returns (loss_scalar, valid_elem_count_for_weighting)
        Uses SmoothL1 over valid elements only.
        """
        if pred.dim() == 2:  # [B,3]  (seq→vector)
            # Ensure target shape is [B,3]
            if target.dim() != 2:
                raise ValueError(f"Expected target [B,3] for vector prediction, got {target.shape}")
            per_elem = self.criterion(pred, target)  # [B,3]
            loss = per_elem.mean()                   # mean over B*3
            valid = pred.size(0) * pred.size(1)      # B*3
            return loss, valid

        elif pred.dim() == 3:  # [B,L,3] (seq→seq)
            B, L, C = pred.shape
            if target.dim() == 2 and target.shape == (B, C):
                # If you accidentally pass [B,3], broadcast it (not recommended).
                # Better is to provide per-step targets. We’ll broadcast with a warning.
                # Remove the warning print if you prefer silence.
                # print("[warn] target is [B,3] but pred is [B,L,3]; broadcasting target across time.")
                target = target[:, None, :].expand(B, L, C)

            if target.shape != pred.shape:
                raise ValueError(f"pred {pred.shape} and target {target.shape} must match for seq→seq")

            valid_mask = self._build_mask_from_lengths(lengths, L)  # [B,L], True=keep
            valid_mask = valid_mask.unsqueeze(-1)                   # [B,L,1]
            per_elem = self.criterion(pred, target)                 # [B,L,3]
            per_elem = per_elem * valid_mask                        # zero out pads
            valid_count = int(valid_mask.sum().item()) * C          # #valid positions * 3 axes
            # Avoid div-by-zero; if no valid tokens (shouldn't happen), fallback to mean over all
            loss = per_elem.sum() / max(valid_count, 1)
            return loss, max(valid_count, 1)

        else:
            raise ValueError(f"Unsupported pred dim {pred.dim()}")

    def save(self, step: int, best: bool = False):
        if not self.ckpt_dir:
            return
        obj = {
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": step,
        }
        path = self.ckpt_dir / (f"ckpt_step_{step}.pt" if not best else "ckpt_best.pt")
        torch.save(obj, path)

    def evaluate(self):
        if self.val_loader is None:
            return {"val_loss": float("nan"), "val_mae": float("nan")}
        self.model.eval()
        tot_loss = 0.0
        tot_mae = 0.0
        denom_loss = 0
        denom_mae = 0
        with torch.no_grad():
            for sensor_pad, ee, _, lengths in self.val_loader:
                sensor_pad = sensor_pad.to(self.device, non_blocking=True)
                ee = ee.to(self.device, non_blocking=True)
                lengths = lengths.to(self.device, non_blocking=True)

                with autocast(enabled=self.amp):
                    pred = self.model(sensor_pad, lengths)   # [B,3] or [B,L,3]
                    loss, valid_count = self._compute_loss(pred, ee, lengths)

                # Aggregate weighted loss
                tot_loss += loss.item() * valid_count
                denom_loss += valid_count

                # MAE aggregation
                if pred.dim() == 2:
                    # [B,3]
                    mae_sum = (pred - ee).abs().sum().item()
                    tot_mae += mae_sum
                    denom_mae += pred.numel()  # B*3
                else:
                    # [B,L,3]
                    B, L, C = pred.shape
                    valid_mask = self._build_mask_from_lengths(lengths, L).unsqueeze(-1)  # [B,L,1]
                    mae_sum = ((pred - ee).abs() * valid_mask).sum().item()
                    tot_mae += mae_sum
                    denom_mae += int(valid_mask.sum().item()) * C

        return {
            "val_loss": tot_loss / max(denom_loss, 1),
            "val_mae":  tot_mae  / max(denom_mae, 1),
        }

    def fit(self, epochs: int, start_step: int = 0):
        step = start_step
        best_val = float("inf")

        for epoch in range(1, epochs + 1):
            self.model.train()
            for i, (sensor_pad, ee, _, lengths) in enumerate(self.train_loader, 1):
                sensor_pad = sensor_pad.to(self.device, non_blocking=True)
                ee = ee.to(self.device, non_blocking=True)
                lengths = lengths.to(self.device, non_blocking=True)

                self.optim.zero_grad(set_to_none=True)
                with autocast(enabled=self.amp):
                    pred = self.model(sensor_pad, lengths)   # [B,3] or [B,L,3]
                    loss, _ = self._compute_loss(pred, ee, lengths)

                self.scaler.scale(loss).backward()
                if self.grad_clip and self.grad_clip > 0:
                    self.scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optim)
                self.scaler.update()
                if self.scheduler:
                    self.scheduler.step()

                step += 1
                if step % self.log_every == 0:
                    print(f"[epoch {epoch}] step {step} loss {loss.item():.6f}")

            # end epoch → evaluate & checkpoint
            metrics = self.evaluate()
            print(f"[epoch {epoch}] val_loss {metrics['val_loss']:.6f}  val_mae {metrics['val_mae']:.6f}")
            if metrics["val_loss"] < best_val:
                best_val = metrics["val_loss"]
                self.save(step, best=True)
            self.save(step, best=False)

        print("Training done.")
