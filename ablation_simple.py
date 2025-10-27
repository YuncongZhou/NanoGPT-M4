#!/usr/bin/env python3
"""
Simple ablation study comparing GELU vs SwiGLU activation functions.
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
from src.dataset import get_batch
from src.tokenizer import get_tokenizer
from src.dataset import ShakespeareDataset


def run_quick_ablation(activation_type, max_iters=50):
    """Run a quick training test with specified activation."""

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nTesting {activation_type.upper()} activation...")

    # Load data
    dataset = ShakespeareDataset(data_dir='data', block_size=64)
    tokenizer = get_tokenizer('character')
    train_data, val_data, tokenizer = dataset.load_and_split(tokenizer, train_ratio=0.9)

    # Small model for quick testing
    config = GPTConfig(
        block_size=64,
        vocab_size=tokenizer.vocab_size,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.1,
        device=device,
        activation=activation_type
    )

    model = GPT(config)
    model.to(device)

    # Training settings
    batch_size = 4
    learning_rate = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Metrics
    losses = []
    times = []

    model.train()
    for iter_num in range(max_iters):
        start_time = time.time()

        X, Y = get_batch(train_data, batch_size, config.block_size, device)
        logits, loss = model(X, Y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step_time = time.time() - start_time
        times.append(step_time)
        losses.append(loss.item())

        if iter_num % 10 == 0:
            print(f"  Iter {iter_num:3d} | Loss: {loss.item():.4f} | Time: {step_time*1000:.1f}ms")

    avg_time = np.mean(times) * 1000
    final_loss = losses[-1]
    print(f"  Average step time: {avg_time:.2f}ms")
    print(f"  Final loss: {final_loss:.4f}")

    return {
        'activation': activation_type,
        'avg_step_time_ms': avg_time,
        'final_loss': final_loss,
        'all_losses': losses,
        'param_count': model.get_num_params(),
        'times': times
    }


def plot_activation_comparison(results):
    """Create comparison plots for activation functions."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Training Loss Over Time
    ax1 = axes[0, 0]
    for name, data in results.items():
        iterations = range(len(data['all_losses']))
        ax1.plot(iterations, data['all_losses'], label=name.upper(), alpha=0.7, linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Step Time Comparison
    ax2 = axes[0, 1]
    names = list(results.keys())
    avg_times = [results[name]['avg_step_time_ms'] for name in names]
    colors = ['#1f77b4', '#ff7f0e']
    bars = ax2.bar(names, avg_times, color=colors[:len(names)])
    ax2.set_ylabel('Average Step Time (ms)')
    ax2.set_title('Training Speed Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, time_val in zip(bars, avg_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.2f}ms', ha='center', va='bottom')

    # Plot 3: Final Loss Comparison
    ax3 = axes[1, 0]
    final_losses = [results[name]['final_loss'] for name in names]
    bars = ax3.bar(names, final_losses, color=colors[:len(names)])
    ax3.set_ylabel('Final Loss')
    ax3.set_title('Final Loss Comparison')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, loss in zip(bars, final_losses):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}', ha='center', va='bottom')

    # Plot 4: Parameter Count
    ax4 = axes[1, 1]
    param_counts = [results[name]['param_count']/1000 for name in names]  # Convert to K
    bars = ax4.bar(names, param_counts, color=colors[:len(names)])
    ax4.set_ylabel('Parameters (K)')
    ax4.set_title('Model Size Comparison')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, params in zip(bars, param_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{params:.1f}K', ha='center', va='bottom')

    plt.suptitle('GELU vs SwiGLU Activation Function Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ablation_activation_comparison.png', dpi=100, bbox_inches='tight')
    print(f"\nPlot saved as ablation_activation_comparison.png")
    plt.close()


def main():
    print("="*50)
    print("ACTIVATION FUNCTION ABLATION")
    print("="*50)

    results = {}

    # Test both activations
    for activation in ['gelu', 'swiglu']:
        results[activation] = run_quick_ablation(activation, max_iters=50)

    # Print comparison
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)

    print(f"\n{'Metric':<20} {'GELU':>12} {'SwiGLU':>12} {'Diff':>12}")
    print("-" * 56)

    gelu_time = results['gelu']['avg_step_time_ms']
    swiglu_time = results['swiglu']['avg_step_time_ms']
    time_diff = ((swiglu_time - gelu_time) / gelu_time) * 100

    gelu_loss = results['gelu']['final_loss']
    swiglu_loss = results['swiglu']['final_loss']
    loss_diff = ((swiglu_loss - gelu_loss) / gelu_loss) * 100

    gelu_params = results['gelu']['param_count']
    swiglu_params = results['swiglu']['param_count']

    print(f"{'Step Time (ms)':<20} {gelu_time:>12.2f} {swiglu_time:>12.2f} {time_diff:>11.1f}%")
    print(f"{'Final Loss':<20} {gelu_loss:>12.4f} {swiglu_loss:>12.4f} {loss_diff:>11.1f}%")
    print(f"{'Parameters':<20} {gelu_params:>12,} {swiglu_params:>12,} {swiglu_params-gelu_params:>12,}")

    # Save results
    with open('ablation_activation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to ablation_activation_results.json")

    # Create plots
    plot_activation_comparison(results)


if __name__ == "__main__":
    main()