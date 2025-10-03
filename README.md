## Transformers HW – Training and Evaluation Guide

This repository contains a minimal Transformer-based model and tooling to train on robot sensor → end-effector (EE) datasets stored in Parquet, and to evaluate trained checkpoints with useful metrics and visualizations.

### Prerequisites

- Python 3.10+
- uv (recommended) for environment management and reproducible installs

Install dependencies (creates a local virtual environment under the project):

```bash
uv sync
```

Activate the environment (optional, `uv run` works without activation):

```bash
source .venv/bin/activate
```

### Data

Training expects Parquet files under `HW1-transformer/data/data_cleaned/` matching `ee_dataset_*.parquet`. You can adjust paths and patterns via CLI flags.

## Training

Run the training script (default hyperparameters work out of the box):

```bash
uv run HW1-transformer/scripts/train.py
```

Common flags:

- `--data_dir` (default: `HW1-transformer/data/data_cleaned`)
- `--pattern` (default: `ee_dataset_*.parquet`)
- `--val_frac` (default: `0.2`) – fraction of the first shard used for validation
- `--rows_per_batch` (default: `4096`) – streaming batch size when reading Parquet
- Model: `--d_model 256 --num_layers 4 --num_heads 8 --d_ff 1024 --dropout 0.1 --in_scale 1.0`
- Training: `--batch_size 512 --epochs 10 --lr 3e-4 --weight_decay 1e-4 --grad_clip 1.0 --amp`
- `--ckpt_dir` (default: `HW1-transformer/outputs/checkpoints`)
- `--device` (auto-detects CUDA if available)

During training, checkpoints are saved to `HW1-transformer/outputs/checkpoints/` (e.g., `ckpt_best.pt`, `ckpt_step_*.pt`).

## Evaluation

Evaluate a trained checkpoint against ground-truth EE data and generate metrics and plots:

```bash
uv run HW1-transformer/scripts/evaluate.py \
  --checkpoint HW1-transformer/outputs/checkpoints/ckpt_best.pt \
  --data_dir HW1-transformer/data/data_cleaned \
  --pattern "ee_dataset_*.parquet" \
  --batch_size 512 \
  --device cuda \
  --output_dir HW1-transformer/outputs/evaluation
```

Useful options:

- `--max_samples N` – quickly evaluate only the first N samples
- `--create_plots` – generate PNG plots (enabled by default)

Outputs are written to `--output_dir` (default `HW1-transformer/outputs/evaluation`). See "Outputs" below.

## Compare Multiple Checkpoints

Quickly score all checkpoints in a folder (by MAE) to find the best one:

```bash
uv run HW1-transformer/scripts/compare_checkpoints.py \
  --checkpoint_dir HW1-transformer/outputs/checkpoints \
  --data_dir HW1-transformer/data/data_cleaned \
  --pattern "ee_dataset_*.parquet" \
  --batch_size 512 \
  --device cuda \
  --max_batches 10
```

This prints a ranked table and saves `checkpoint_comparison.json` into the checkpoint directory.

## What the Scripts Do

- `HW1-transformer/scripts/train.py`
  - Streams Parquet data, pads variable-length sequences, trains a Transformer encoder to predict a single `[X, Y, Z]` EE vector per sequence
  - Logs periodic training/validation losses
  - Saves rolling checkpoints and `ckpt_best.pt` (best validation)

- `HW1-transformer/scripts/evaluate.py`
  - Loads a checkpoint and computes metrics on the dataset
  - Produces visualizations and a machine-readable JSON with metrics

- `HW1-transformer/scripts/compare_checkpoints.py`
  - Iterates all checkpoints and computes a quick MAE-based score over a few batches to rank models

## Outputs

### Checkpoints (`HW1-transformer/outputs/checkpoints/`)

- `ckpt_best.pt` – best-performing checkpoint by validation metric
- `ckpt_step_*.pt` – periodic snapshots with embedded training step
- `checkpoint_comparison.json` – (from compare script) summary with MAE per checkpoint

### Evaluation Reports (default `HW1-transformer/outputs/evaluation/`)

- `evaluation_metrics.json` – numeric results including:
  - `mae`, `mse`, `rmse`, `r2` (R² is clamped to avoid pathological values)
  - Per-axis `mae_x/mae_y/mae_z`, `mse_x/mse_y/mse_z`
  - `mean_euclidean_error`, `std_euclidean_error`, `num_samples`
- Plots (PNG):
  - `model_eval_scatter_plots.png` – predicted vs ground-truth per axis with y=x reference
  - `model_eval_error_distributions.png` – error histograms per axis
  - `model_eval_3d_scatter.png` – 3D scatter of predictions vs ground truth
  - `model_eval_euclidean_errors.png` – histogram of Euclidean distance errors

If you run a full-dataset evaluation with a different output directory, expect the same file set under that path (e.g., `HW1-transformer/outputs/evaluation_full/`).

## Tips

- If you add new dependencies, update `pyproject.toml` and run `uv sync`.
- Use `--device cpu` if a GPU is unavailable.
- For quick iteration, use `--max_samples` in `evaluate.py` or reduce `--max_batches` in `compare_checkpoints.py`.

# Transformers for Robotics Homework