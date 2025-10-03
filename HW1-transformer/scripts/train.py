from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path
from typing import Iterator, Tuple
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

# Add the HW1-transformer directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.dataset import EEDataset, pad_collate
from src.models.transformer import Transformer
from src.trainers.trainer import Trainer


class EEDatasetSplit(EEDataset):
    """
    Restricts iteration to a row-range [start, end) within the (single) Parquet file.
    If multiple files exist, it applies the split only to the first file and then yields
    the rest entirely (simple, deterministic).
    """
    def __init__(self, *args, start_row: int = 0, end_row: int = 2**63 - 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_row = int(start_row)
        self.end_row = int(end_row)

    def _iter_file(self, file_path: str) -> Iterator[Tuple[np.ndarray, np.ndarray, np.int64]]:
        # Override to honor [start_row, end_row)
        pf = pq.ParquetFile(file_path)
        cols = [self.sensor_col, self.ee_x_col, self.ee_y_col, self.ee_z_col, self.timestamp_col]

        # compute cumulative rows to skip / take
        total_rows = pf.metadata.num_rows
        start = max(0, self.start_row)
        end = min(total_rows, self.end_row)
        if start >= end:
            return iter(())  # empty

        # Walk batches and slice by global row index
        seen = 0
        for rb in pf.iter_batches(batch_size=self.rows_per_batch, columns=cols):
            n = rb.num_rows
            batch_start = seen
            batch_end = seen + n
            seen = batch_end

            take_start = max(start - batch_start, 0)
            take_end = min(end - batch_start, n)
            if take_start >= take_end:
                if batch_end >= end:
                    break
                continue

            # Slice the batch rows we need
            rb = rb.slice(take_start, take_end - take_start)

            # Reuse parent parsing
            sensor_arr = rb.column(rb.schema.get_field_index(self.sensor_col))
            flat = sensor_arr.values.to_numpy(zero_copy_only=False)
            list_size = sensor_arr.type.list_size
            n = len(sensor_arr)
            # After slicing the record batch, we need to slice the flattened data too
            flat_sliced = flat[:n * list_size]
            sensor_np = flat_sliced.reshape(n, list_size).astype(np.int32, copy=False)

            ee_x = rb.column(rb.schema.get_field_index(self.ee_x_col)).to_numpy(zero_copy_only=False)
            ee_y = rb.column(rb.schema.get_field_index(self.ee_y_col)).to_numpy(zero_copy_only=False)
            ee_z = rb.column(rb.schema.get_field_index(self.ee_z_col)).to_numpy(zero_copy_only=False)
            ts   = rb.column(rb.schema.get_field_index(self.timestamp_col)).to_numpy(zero_copy_only=False)

            ee_np = np.stack([ee_x, ee_y, ee_z], axis=1).astype(np.float32, copy=False)

            for i in range(n):
                yield sensor_np[i], ee_np[i], np.int64(ts[i])

        return iter(())  # just in case


def make_split_loaders(
    data_dir: str,
    pattern: str,
    batch_size: int,
    num_workers: int,
    rows_per_batch: int,
    val_frac: float,
    pin_memory: bool = True,
):
    # Determine row count from the (first) shard
    files = sorted(Path(data_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No Parquet files found at {data_dir}/{pattern}")
    pf0 = pq.ParquetFile(str(files[0]))
    N = pf0.metadata.num_rows
    n_train = int((1.0 - val_frac) * N)

    train_ds = EEDatasetSplit(
        data_dir=data_dir,
        pattern=pattern,
        shuffle_files=False,
        rows_per_batch=rows_per_batch,
        start_row=0,
        end_row=n_train,
    )
    val_ds = EEDatasetSplit(
        data_dir=data_dir,
        pattern=pattern,
        shuffle_files=False,
        rows_per_batch=rows_per_batch,
        start_row=n_train,
        end_row=N,
    )

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers,
        collate_fn=pad_collate, pin_memory=pin_memory, persistent_workers=False
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, num_workers=0,
        collate_fn=pad_collate, pin_memory=pin_memory, persistent_workers=False
    )
    return train_dl, val_dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="HW1-transformer/data/data_cleaned")
    ap.add_argument("--pattern", default="ee_dataset_*.parquet")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--rows_per_batch", type=int, default=4096)

    # model / train hparams
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--in_scale", type=float, default=1.0)

    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=0)  # keep 0 for IterableDataset split safety
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--ckpt_dir", default="HW1-transformer/outputs/checkpoints")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    train_dl, val_dl = make_split_loaders(
        data_dir=args.data_dir,
        pattern=args.pattern,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rows_per_batch=args.rows_per_batch,
        val_frac=args.val_frac,
        pin_memory=True,
    )

    model = Transformer(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        in_scale=args.in_scale,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Cosine schedule with warmup (optional)
    total_steps = args.epochs * math.ceil((len(val_dl.dataset.files) > 0 and 1) or 1)  # dummy; scheduler optional
    scheduler = None

    trainer = Trainer(
        model=model,
        optimizer=optim,
        train_loader=train_dl,
        val_loader=val_dl,
        device=args.device,
        amp=args.amp,
        grad_clip=args.grad_clip,
        ckpt_dir=args.ckpt_dir,
        scheduler=scheduler,
        log_every=100,
    )

    trainer.fit(epochs=args.epochs)


if __name__ == "__main__":
    main()
