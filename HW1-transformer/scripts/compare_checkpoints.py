#!/usr/bin/env python3
"""
Compare different model checkpoints to find the best performing one.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add the HW1-transformer directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.dataset import EEDataset, pad_collate
from src.models.transformer import Transformer


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


def evaluate_model_quick(
    model: Transformer,
    dataloader: DataLoader,
    device: str = "cpu",
    max_batches: int = 10
) -> float:
    """Quick evaluation returning only MAE."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
                
            sensor_pad, ee, _, lengths = batch
            sensor_pad = sensor_pad.to(device)
            ee = ee.to(device)
            lengths = lengths.to(device)
            
            # Get predictions
            pred = model(sensor_pad, lengths)
            
            all_predictions.append(pred.cpu().numpy())
            all_targets.append(ee.cpu().numpy())
    
    # Concatenate all predictions and targets
    predictions = torch.tensor(np.concatenate(all_predictions, axis=0))
    targets = torch.tensor(np.concatenate(all_targets, axis=0))
    
    # Calculate MAE
    mae = torch.mean(torch.abs(predictions - targets)).item()
    
    return mae


def main():
    parser = argparse.ArgumentParser(description="Compare model checkpoints")
    parser.add_argument("--checkpoint_dir", default="HW1-transformer/outputs/checkpoints",
                       help="Directory containing checkpoints")
    parser.add_argument("--data_dir", default="HW1-transformer/data/data_cleaned", 
                       help="Directory containing parquet files")
    parser.add_argument("--pattern", default="ee_dataset_*.parquet", 
                       help="Pattern to match parquet files")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for evaluation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run evaluation on")
    parser.add_argument("--max_batches", type=int, default=10,
                       help="Maximum number of batches to evaluate (for quick comparison)")
    
    args = parser.parse_args()
    
    # Find all checkpoint files
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Create dataset
    print(f"Loading data from {args.data_dir}...")
    dataset = EEDataset(
        data_dir=args.data_dir,
        pattern=args.pattern,
        shuffle_files=False,
        rows_per_batch=4096
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=0,
        collate_fn=pad_collate,
        pin_memory=True
    )
    
    # Evaluate each checkpoint
    results = []
    
    for checkpoint_file in sorted(checkpoint_files):
        print(f"\nEvaluating {checkpoint_file.name}...")
        
        try:
            # Load model
            model = load_model(str(checkpoint_file), args.device)
            
            # Quick evaluation
            mae = evaluate_model_quick(model, dataloader, args.device, args.max_batches)
            
            # Get training step if available
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            step = checkpoint.get('step', 'unknown')
            
            results.append({
                'checkpoint': checkpoint_file.name,
                'step': step,
                'mae': mae
            })
            
            print(f"  Step: {step}, MAE: {mae:.6f}")
            
        except Exception as e:
            print(f"  Error loading {checkpoint_file.name}: {e}")
            results.append({
                'checkpoint': checkpoint_file.name,
                'step': 'error',
                'mae': float('inf')
            })
    
    # Sort by MAE (lower is better)
    results.sort(key=lambda x: x['mae'])
    
    # Print results
    print(f"\n{'='*80}")
    print("CHECKPOINT COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"{'Rank':<4} {'Checkpoint':<25} {'Step':<10} {'MAE':<12}")
    print(f"{'-'*80}")
    
    for i, result in enumerate(results):
        rank = i + 1
        checkpoint = result['checkpoint']
        step = result['step']
        mae = result['mae']
        
        if mae == float('inf'):
            print(f"{rank:<4} {checkpoint:<25} {step:<10} {'ERROR':<12}")
        else:
            print(f"{rank:<4} {checkpoint:<25} {step:<10} {mae:<12.6f}")
    
    print(f"{'='*80}")
    
    # Find best checkpoint
    best_result = results[0]
    if best_result['mae'] != float('inf'):
        print(f"\nBest checkpoint: {best_result['checkpoint']}")
        print(f"Training step: {best_result['step']}")
        print(f"MAE: {best_result['mae']:.6f}")
    else:
        print("\nNo valid checkpoints found!")
    
    # Save results
    output_file = Path(args.checkpoint_dir) / "checkpoint_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
