#!/usr/bin/env python3
"""
Ablation study comparing GELU vs SwiGLU activation functions.
"""

import torch
import numpy as np
import time
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))

from src.model import GPT, GPTConfig
from src.dataset import ShakespeareDataset, get_batch
from src.tokenizer import get_tokenizer


def run_activation_ablation(activation_type, config_override=None):
    """Run training with specified activation function."""

    # Setup
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n{'='*50}")
    print(f"Running ablation with {activation_type.upper()} activation")
    print(f"Device: {device}")
    print(f"{'='*50}")

    # Load data
    dataset = ShakespeareDataset(data_dir='data', block_size=128)
    tokenizer = get_tokenizer('character')
    train_data, val_data, tokenizer = dataset.load_and_split(tokenizer, train_ratio=0.9)

    # Model configuration
    config = GPTConfig(
        block_size=128,
        vocab_size=tokenizer.vocab_size,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1,
        device=device,
        activation=activation_type
    )

    if config_override:
        for k, v in config_override.items():
            setattr(config, k, v)

    # Create model
    model = GPT(config)
    model.to(device)

    # Training settings
    batch_size = 8
    max_iters = 100  # Reduced for faster ablation
    learning_rate = 3e-4
    eval_interval = 25

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Metrics storage
    metrics = {
        'activation': activation_type,
        'train_losses': [],
        'val_losses': [],
        'step_times': [],
        'iters': []
    }

    # Training loop
    print(f"Starting training for {max_iters} iterations...")
    model.train()

    for iter_num in range(max_iters):
        start_time = time.time()

        # Get batch
        X, Y = get_batch(train_data, batch_size, config.block_size, device)

        # Forward pass
        logits, loss = model(X, Y)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Record step time
        step_time = time.time() - start_time
        metrics['step_times'].append(step_time)

        # Logging
        if iter_num % 10 == 0:
            print(f"Iter {iter_num:4d} | Loss: {loss.item():.4f} | Step time: {step_time*1000:.2f}ms")
            metrics['train_losses'].append(loss.item())
            metrics['iters'].append(iter_num)

        # Evaluation
        if iter_num % eval_interval == 0 and iter_num > 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(10):  # Average over 10 batches
                    X_val, Y_val = get_batch(val_data, batch_size, config.block_size, device)
                    _, val_loss = model(X_val, Y_val)
                    val_losses.append(val_loss.item())
                avg_val_loss = np.mean(val_losses)
                metrics['val_losses'].append(avg_val_loss)
                print(f"  Val loss: {avg_val_loss:.4f}")
            model.train()

    # Final metrics
    avg_step_time = np.mean(metrics['step_times'])
    final_train_loss = metrics['train_losses'][-1] if metrics['train_losses'] else float('inf')
    final_val_loss = metrics['val_losses'][-1] if metrics['val_losses'] else float('inf')

    print(f"\n{activation_type.upper()} Results:")
    print(f"  Average step time: {avg_step_time*1000:.2f}ms")
    print(f"  Final train loss: {final_train_loss:.4f}")
    print(f"  Final val loss: {final_val_loss:.4f}")
    print(f"  Total parameters: {model.get_num_params()/1e6:.2f}M")

    return metrics


def plot_comparison(results):
    """Plot comparison of different activation functions."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot training loss
    ax1 = axes[0]
    for name, metrics in results.items():
        ax1.plot(metrics['iters'], metrics['train_losses'], label=name.upper(), marker='o', markersize=3)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot step times
    ax2 = axes[1]
    names = list(results.keys())
    avg_times = [np.mean(results[name]['step_times']) * 1000 for name in names]
    colors = ['blue', 'orange']
    bars = ax2.bar(names, avg_times, color=colors[:len(names)])
    ax2.set_ylabel('Average Step Time (ms)')
    ax2.set_title('Training Speed Comparison')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, time in zip(bars, avg_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f}ms', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('ablation_activation_results.png', dpi=100)
    plt.show()
    print(f"\nPlot saved as ablation_activation_results.png")


def main():
    """Run ablation study comparing activation functions."""

    results = {}

    # Run experiments
    for activation in ['gelu', 'swiglu']:
        metrics = run_activation_ablation(activation)
        results[activation] = metrics

    # Save results
    output_file = Path('ablation_activation_results.json')
    with open(output_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for name, metrics in results.items():
            json_results[name] = {
                k: v if not isinstance(v, np.ndarray) else v.tolist()
                for k, v in metrics.items()
            }
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Plot comparison
    plot_comparison(results)

    # Print summary
    print("\n" + "="*50)
    print("ABLATION SUMMARY")
    print("="*50)

    for name, metrics in results.items():
        avg_time = np.mean(metrics['step_times']) * 1000
        final_loss = metrics['train_losses'][-1]
        print(f"\n{name.upper()}:")
        print(f"  Avg step time: {avg_time:.2f}ms")
        print(f"  Final loss: {final_loss:.4f}")
        if metrics['val_losses']:
            print(f"  Final val loss: {metrics['val_losses'][-1]:.4f}")


if __name__ == "__main__":
    main()