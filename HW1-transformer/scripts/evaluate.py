#!/usr/bin/env python3
"""
Evaluation script for the transformer model.
Evaluates trained model against ground truth data and provides comprehensive metrics.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add the HW1-transformer directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.dataset import EEDataset, pad_collate
from src.models.transformer import Transformer
from src.trainers.trainer import Trainer


def load_model(checkpoint_path: str, device: str = "cpu") -> Transformer:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model parameters from checkpoint
    if 'model_state_dict' in checkpoint:
        model_params = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        model_params = checkpoint['model']
    else:
        model_params = checkpoint
    
    # Create model with default parameters (should match training)
    model = Transformer(
        d_model=256,
        num_layers=4,
        num_heads=8,
        d_ff=1024,
        dropout=0.1,
        in_scale=1.0,
    )
    
    # Load the state dict
    model.load_state_dict(model_params)
    model.to(device)
    model.eval()
    
    return model


def evaluate_model(
    model: Transformer,
    dataloader: DataLoader,
    device: str = "cpu"
) -> Dict[str, float]:
    """Evaluate model and return comprehensive metrics."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_lengths = []
    
    with torch.no_grad():
        for batch in dataloader:
            sensor_pad, ee, _, lengths = batch
            sensor_pad = sensor_pad.to(device)
            ee = ee.to(device)
            lengths = lengths.to(device)
            
            # Get predictions
            pred = model(sensor_pad, lengths)
            
            all_predictions.append(pred.cpu().numpy())
            all_targets.append(ee.cpu().numpy())
            all_lengths.append(lengths.cpu().numpy())
    
    # Concatenate all predictions and targets
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    lengths = np.concatenate(all_lengths, axis=0)
    
    # Calculate metrics
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    
    # Calculate R² with bounds checking
    try:
        r2 = r2_score(targets, predictions)
        # Clamp R² to reasonable bounds to avoid numerical issues
        r2 = max(-1000.0, min(1.0, r2))
    except:
        r2 = -999.0  # Indicate calculation failed
    
    # Per-axis metrics
    mae_x = mean_absolute_error(targets[:, 0], predictions[:, 0])
    mae_y = mean_absolute_error(targets[:, 1], predictions[:, 1])
    mae_z = mean_absolute_error(targets[:, 2], predictions[:, 2])
    
    mse_x = mean_squared_error(targets[:, 0], predictions[:, 0])
    mse_y = mean_squared_error(targets[:, 1], predictions[:, 1])
    mse_z = mean_squared_error(targets[:, 2], predictions[:, 2])
    
    # Euclidean distance error
    euclidean_errors = np.linalg.norm(predictions - targets, axis=1)
    mean_euclidean_error = np.mean(euclidean_errors)
    std_euclidean_error = np.std(euclidean_errors)
    
    metrics = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'mae_x': float(mae_x),
        'mae_y': float(mae_y),
        'mae_z': float(mae_z),
        'mse_x': float(mse_x),
        'mse_y': float(mse_y),
        'mse_z': float(mse_z),
        'mean_euclidean_error': float(mean_euclidean_error),
        'std_euclidean_error': float(std_euclidean_error),
        'num_samples': int(len(predictions))
    }
    
    return metrics, predictions, targets, lengths


def create_visualizations(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_dir: str,
    prefix: str = "eval"
) -> None:
    """Create comprehensive visualizations of model performance."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Scatter plots for each axis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.flatten()
    
    for i, axis_name in enumerate(['X', 'Y', 'Z']):
        ax = axes[i]
        ax.scatter(targets[:, i], predictions[:, i], alpha=0.6, s=10)
        
        # Perfect prediction line
        min_val = min(targets[:, i].min(), predictions[:, i].min())
        max_val = max(targets[:, i].max(), predictions[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel(f'True {axis_name}')
        ax.set_ylabel(f'Predicted {axis_name}')
        ax.set_title(f'{axis_name}-axis: Predictions vs Ground Truth')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / f'{prefix}_scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Error distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.flatten()
    
    for i, axis_name in enumerate(['X', 'Y', 'Z']):
        ax = axes[i]
        errors = predictions[:, i] - targets[:, i]
        ax.hist(errors, bins=50, alpha=0.7, density=True)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.set_xlabel(f'{axis_name}-axis Error')
        ax.set_ylabel('Density')
        ax.set_title(f'{axis_name}-axis: Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / f'{prefix}_error_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot ground truth in blue
    ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2], 
               c='blue', alpha=0.6, s=20, label='Ground Truth')
    
    # Plot predictions in red
    ax.scatter(predictions[:, 0], predictions[:, 1], predictions[:, 2], 
               c='red', alpha=0.6, s=20, label='Predictions')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scatter: Ground Truth vs Predictions')
    ax.legend()
    
    plt.savefig(output_path / f'{prefix}_3d_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Euclidean distance error distribution
    euclidean_errors = np.linalg.norm(predictions - targets, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.hist(euclidean_errors, bins=50, alpha=0.7, density=True)
    plt.axvline(euclidean_errors.mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {euclidean_errors.mean():.4f}')
    plt.xlabel('Euclidean Distance Error')
    plt.ylabel('Density')
    plt.title('Euclidean Distance Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / f'{prefix}_euclidean_errors.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_path}")


def print_metrics(metrics: Dict[str, float], dataset_name: str = "Dataset") -> None:
    """Print formatted metrics."""
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS - {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"Number of samples: {metrics['num_samples']:,}")
    print(f"\nOverall Metrics:")
    print(f"  MAE (Mean Absolute Error): {metrics['mae']:.6f}")
    print(f"  MSE (Mean Squared Error):  {metrics['mse']:.6f}")
    print(f"  RMSE (Root Mean Squared Error): {metrics['rmse']:.6f}")
    print(f"  R² (Coefficient of Determination): {metrics['r2']:.6f}")
    print(f"  Mean Euclidean Distance Error: {metrics['mean_euclidean_error']:.6f}")
    print(f"  Std Euclidean Distance Error:  {metrics['std_euclidean_error']:.6f}")
    
    print(f"\nPer-Axis MAE:")
    print(f"  X-axis: {metrics['mae_x']:.6f}")
    print(f"  Y-axis: {metrics['mae_y']:.6f}")
    print(f"  Z-axis: {metrics['mae_z']:.6f}")
    
    print(f"\nPer-Axis MSE:")
    print(f"  X-axis: {metrics['mse_x']:.6f}")
    print(f"  Y-axis: {metrics['mse_y']:.6f}")
    print(f"  Z-axis: {metrics['mse_z']:.6f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained transformer model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", default="HW1-transformer/data/data_cleaned", 
                       help="Directory containing parquet files")
    parser.add_argument("--pattern", default="ee_dataset_*.parquet", 
                       help="Pattern to match parquet files")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for evaluation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run evaluation on")
    parser.add_argument("--output_dir", default="HW1-transformer/outputs/evaluation",
                       help="Directory to save evaluation results")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (for quick testing)")
    parser.add_argument("--create_plots", action="store_true", default=True,
                       help="Create visualization plots")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.device)
    print("Model loaded successfully!")
    
    # Create dataset
    print(f"Loading data from {args.data_dir}...")
    dataset = EEDataset(
        data_dir=args.data_dir,
        pattern=args.pattern,
        shuffle_files=False,
        rows_per_batch=4096
    )
    
    # Limit dataset size if specified
    if args.max_samples:
        # Create a limited dataset by taking only the first max_samples
        limited_data = []
        count = 0
        for sample in dataset:
            limited_data.append(sample)
            count += 1
            if count >= args.max_samples:
                break
        
        # Create a simple dataset wrapper
        class LimitedDataset:
            def __init__(self, data):
                self.data = data
            
            def __iter__(self):
                return iter(self.data)
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        dataset = LimitedDataset(limited_data)
        print(f"Limited dataset to {len(limited_data)} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=0,
        collate_fn=pad_collate,
        pin_memory=True
    )
    
    # Evaluate model
    print("Evaluating model...")
    metrics, predictions, targets, lengths = evaluate_model(model, dataloader, args.device)
    
    # Print results
    print_metrics(metrics, "Test Dataset")
    
    # Create visualizations
    if args.create_plots:
        print("Creating visualizations...")
        create_visualizations(predictions, targets, args.output_dir, "model_eval")
    
    # Save metrics to file
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metrics_file = output_path / "evaluation_metrics.json"
    import json
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")
    
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
