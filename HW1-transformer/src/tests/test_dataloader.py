import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import pytest
from pathlib import Path
import os

from datasets.dataset import EEDataset, pad_collate

def _write_single_parquet(path: Path, n_rows: int, list_size: int):
    """
    Create ONE parquet with:
      - timestamp_ns: int64 [n_rows]
      - ee_x/ee_y/ee_z: float32 [n_rows]
      - sensor_raw: FixedSizeList<int32>[n_rows] with width=list_size (same for all rows)
    """
    rng = np.random.default_rng(0)
    # deterministic ascending sensor values per row
    sensors = [np.arange(i * 10, i * 10 + list_size, dtype=np.int32) for i in range(n_rows)]
    flat = np.concatenate(sensors, axis=0)
    sensor_flat_arr = pa.array(flat, type=pa.int32())
    sensor_col = pa.FixedSizeListArray.from_arrays(sensor_flat_arr, list_size=list_size)

    ts = pa.array(np.arange(1_000, 1_000 + n_rows, dtype=np.int64), type=pa.int64())
    ee_x = pa.array(rng.normal(size=n_rows).astype(np.float32), type=pa.float32())
    ee_y = pa.array(rng.normal(size=n_rows).astype(np.float32), type=pa.float32())
    ee_z = pa.array(rng.normal(size=n_rows).astype(np.float32), type=pa.float32())

    table = pa.table(
        {
            "timestamp_ns": ts,
            "ee_x": ee_x,
            "ee_y": ee_y,
            "ee_z": ee_z,
            "sensor_raw": sensor_col,
        }
    )
    pq.write_table(table, str(path), compression="zstd")


def test_single_parquet_streams_and_shapes(tmp_path: Path):
    """EEDataset should stream from a single parquet and pad_collate should return expected tensors."""
    shard = tmp_path / "ee_dataset_00000.parquet"
    _write_single_parquet(shard, n_rows=7, list_size=6)

    ds = EEDataset(
        data_dir=str(tmp_path),
        pattern="ee_dataset_*.parquet",   # matches exactly one file
        shuffle_files=False,
        rows_per_batch=4,                 # exercise iter_batches
    )

    # Batch size 5 to test across record-batch boundaries within the same file
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=5,
        num_workers=0,
        collate_fn=pad_collate,
        pin_memory=False,
    )

    it = iter(dl)
    sensor_1, ee_1, ts_1, lengths_1 = next(it)

    # First batch: 5 rows; sensor width = 6 (no extra padding expected)
    assert sensor_1.shape == (5, 6)
    assert ee_1.shape == (5, 3)
    assert ts_1.shape == (5,)
    assert lengths_1.tolist() == [6] * 5
    assert sensor_1.dtype == torch.int32
    assert ee_1.dtype == torch.float32
    assert ts_1.dtype == torch.int64

    # Check exact sensor values from our deterministic writer
    # row0: [0..5], row1: [10..15]
    assert sensor_1[0].tolist() == [0, 1, 2, 3, 4, 5]
    assert sensor_1[1].tolist() == [10, 11, 12, 13, 14, 15]

    # Second (final) batch: remaining 2 rows, still width 6
    sensor_2, ee_2, ts_2, lengths_2 = next(it)
    assert sensor_2.shape == (2, 6)
    assert lengths_2.tolist() == [6, 6]
    assert sensor_2[0].tolist() == [50, 51, 52, 53, 54, 55]
    with pytest.raises(StopIteration):
        _ = next(it)
        

DATA_DIR = "HW1-transformer/data/data_cleaned"

@pytest.mark.skipif(not any(f.endswith(".parquet") for f in os.listdir(DATA_DIR)) if os.path.isdir(DATA_DIR) else True,
                    reason="real parquet not found")
def test_smoke_real_single_parquet():
    ds = EEDataset(data_dir=DATA_DIR, pattern="*.parquet", rows_per_batch=4096)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=0, collate_fn=pad_collate)
    batch = next(iter(dl))
    sensor, ee, ts, lengths = batch
    assert sensor.ndim == 2 and ee.shape[-1] == 3
