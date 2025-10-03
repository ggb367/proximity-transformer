# ee_dataloader.py
import glob
import os
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, DataLoader


class EEDataset(IterableDataset):
    """
    Streams (sensor_raw[int32[L]], ee[float32,3], timestamp[int64]) from Parquet shards.
    - No indexing: this is built for very large datasets.
    - Efficient: uses ParquetFile.iter_batches to avoid loading whole files.
    """
    def __init__(
        self,
        data_dir: str,
        pattern: str = "ee_dataset_*.parquet",
        columns: Tuple[str, str, str, str] = ("sensor_raw", "ee_x", "ee_y", "ee_z"),
        timestamp_col: str = "timestamp_ns",
        shuffle_files: bool = False,
        rows_per_batch: int = 4096,
    ):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if not self.files:
            raise FileNotFoundError(f"No Parquet files found at {data_dir}/{pattern}")
        self.sensor_col, self.ee_x_col, self.ee_y_col, self.ee_z_col = columns
        self.timestamp_col = timestamp_col
        self.shuffle_files = shuffle_files
        self.rows_per_batch = rows_per_batch

    def _iter_file(self, file_path: str) -> Iterator[Tuple[np.ndarray, np.ndarray, np.int64]]:
        """Yield samples from a single parquet file in streaming batches."""
        pf = pq.ParquetFile(file_path)
        cols = [self.sensor_col, self.ee_x_col, self.ee_y_col, self.ee_z_col, self.timestamp_col]

        for rb in pf.iter_batches(batch_size=self.rows_per_batch, columns=cols):
            # rb: pyarrow.RecordBatch
            # sensor_raw is FixedSizeListArray => flatten then reshape
            sensor_arr: pa.FixedSizeListArray = rb.column(rb.schema.get_field_index(self.sensor_col))
            flat = sensor_arr.values.to_numpy(zero_copy_only=False)  # 1D array
            list_size = sensor_arr.type.list_size
            n = len(sensor_arr)
            sensor_np = flat.reshape(n, list_size).astype(np.int32, copy=False)

            ee_x = rb.column(rb.schema.get_field_index(self.ee_x_col)).to_numpy(zero_copy_only=False)
            ee_y = rb.column(rb.schema.get_field_index(self.ee_y_col)).to_numpy(zero_copy_only=False)
            ee_z = rb.column(rb.schema.get_field_index(self.ee_z_col)).to_numpy(zero_copy_only=False)
            ts   = rb.column(rb.schema.get_field_index(self.timestamp_col)).to_numpy(zero_copy_only=False)

            ee_np = np.stack([ee_x, ee_y, ee_z], axis=1).astype(np.float32, copy=False)

            # Yield row-wise to keep collate flexible for padding
            for i in range(n):
                yield sensor_np[i], ee_np[i], np.int64(ts[i])

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.int64]]:
        files = self.files[:]
        if self.shuffle_files:
            rng = np.random.default_rng()
            rng.shuffle(files)

        for fp in files:
            yield from self._iter_file(fp)


def pad_collate(
    batch: List[Tuple[np.ndarray, np.ndarray, np.int64]],
    pad_value: int = 0,
):
    """
    Pads sensor_raw to the max length in the batch.
    Returns:
      sensor_raw: int32 [B, Lmax]
      ee:         float32 [B, 3]
      ts:         int64 [B]
      lengths:    int32 [B] (original sensor lengths, if useful)
    """
    sensors, ees, tss = zip(*batch)
    lengths = torch.as_tensor([s.shape[0] for s in sensors], dtype=torch.int32)

    Lmax = int(lengths.max().item())
    B = len(sensors)
    sensor_pad = torch.full((B, Lmax), pad_value, dtype=torch.int32)
    for i, s in enumerate(sensors):
        l = s.shape[0]
        sensor_pad[i, :l] = torch.from_numpy(s)

    ee = torch.from_numpy(np.stack(ees, axis=0))            # [B, 3], float32
    ts = torch.as_tensor(tss, dtype=torch.int64)            # [B]

    return sensor_pad, ee, ts, lengths


def make_dataloader(
    data_dir: str,
    pattern: str = "ee_dataset_*.parquet",
    batch_size: int = 256,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    shuffle_files: bool = False,
    rows_per_batch: int = 4096,
    pin_memory: bool = True,
):
    ds = EEDataset(
        data_dir=data_dir,
        pattern=pattern,
        shuffle_files=shuffle_files,
        rows_per_batch=rows_per_batch,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=pad_collate,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
