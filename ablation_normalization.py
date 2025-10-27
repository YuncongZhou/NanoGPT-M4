#!/usr/bin/env python3
"""
Ablation study comparing LayerNorm vs TanhNorm normalization methods.
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
from src.dataset import get_batch, ShakespeareDataset
from src.tokenizer import get_tokenizer


def run_norm_ablation(norm_type, alpha=0.5, max_iters=50):
    """Run a quick training test with specified normalization."""

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nTesting {norm_type.upper()} normalization...")
    if norm_type == "tanhnorm":
        print(f"  Alpha: {alpha}")

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
        norm_type=norm_type,
        norm_alpha=alpha
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

        # Gradient clipping for stability with tanh norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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
        'norm_type': norm_type,
        'avg_step_time_ms': avg_time,
        'final_loss': final_loss,
        'all_losses': losses,
        'param_count': model.get_num_params(),
        'alpha': alpha if norm_type == "tanhnorm" else None
    }


def test_alpha_values():
    """Test different alpha values for tanh normalization."""
    print("\n" + "="*50)
    print("TESTING DIFFERENT ALPHA VALUES FOR TANHNORM")
    print("="*50)

    # Use more reasonable alpha values based on the paper
    alphas = [0.2, 0.5, 0.8, 1.0]
    results = {}

    for alpha in alphas:
        print(f"\nAlpha = {alpha}")
        result = run_norm_ablation("tanhnorm", alpha=alpha, max_iters=30)
        results[f"tanhnorm_alpha_{alpha}"] = result

    return results


def plot_normalization_comparison(results):
    """Create comparison plots for normalization methods."""

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Training Loss Over Time (LayerNorm vs TanhNorm)
    ax1 = axes[0, 0]
    for name in ['layernorm', 'tanhnorm']:
        if name in results:
            iterations = range(len(results[name]['all_losses']))
            ax1.plot(iterations, results[name]['all_losses'],
                    label=name.replace('norm', 'Norm').replace('layer', 'Layer').replace('tanh', 'Tanh'),
                    alpha=0.7, linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss: LayerNorm vs TanhNorm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Alpha Value Comparison for TanhNorm
    ax2 = axes[0, 1]
    alpha_results = {k: v for k, v in results.items() if k.startswith('tanhnorm_alpha')}
    for name, data in alpha_results.items():
        alpha_val = data['alpha']
        iterations = range(len(data['all_losses']))
        ax2.plot(iterations, data['all_losses'], label=f'α={alpha_val}', alpha=0.7, linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('TanhNorm: Effect of Alpha Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Step Time Comparison
    ax3 = axes[0, 2]
    main_methods = ['layernorm', 'tanhnorm']
    avg_times = [results[name]['avg_step_time_ms'] for name in main_methods]
    colors = ['#2ca02c', '#d62728']
    bars = ax3.bar(['LayerNorm', 'TanhNorm'], avg_times, color=colors)
    ax3.set_ylabel('Average Step Time (ms)')
    ax3.set_title('Training Speed Comparison')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, time_val in zip(bars, avg_times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.2f}ms', ha='center', va='bottom')

    # Plot 4: Final Loss Comparison (Main Methods)
    ax4 = axes[1, 0]
    final_losses = [results[name]['final_loss'] for name in main_methods]
    bars = ax4.bar(['LayerNorm', 'TanhNorm'], final_losses, color=colors)
    ax4.set_ylabel('Final Loss')
    ax4.set_title('Final Loss: LayerNorm vs TanhNorm')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, loss in zip(bars, final_losses):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}', ha='center', va='bottom')

    # Plot 5: Alpha vs Final Loss
    ax5 = axes[1, 1]
    alphas = []
    alpha_losses = []
    for name, data in alpha_results.items():
        alphas.append(data['alpha'])
        alpha_losses.append(data['final_loss'])
    if alphas:
        ax5.plot(alphas, alpha_losses, 'o-', color='#d62728', markersize=8, linewidth=2)
        ax5.set_xlabel('Alpha Value')
        ax5.set_ylabel('Final Loss')
        ax5.set_title('TanhNorm: Alpha vs Final Loss')
        ax5.grid(True, alpha=0.3)
        for alpha, loss in zip(alphas, alpha_losses):
            ax5.annotate(f'{loss:.4f}', (alpha, loss), textcoords="offset points",
                        xytext=(0,10), ha='center', fontsize=9)

    # Plot 6: Convergence Rate Comparison
    ax6 = axes[1, 2]
    for name in ['layernorm', 'tanhnorm']:
        if name in results:
            losses = results[name]['all_losses']
            # Calculate relative improvement
            if len(losses) > 1:
                relative_improvement = [(losses[0] - losses[i])/losses[0] * 100
                                       for i in range(len(losses))]
                iterations = range(len(relative_improvement))
                ax6.plot(iterations, relative_improvement,
                        label=name.replace('norm', 'Norm').replace('layer', 'Layer').replace('tanh', 'Tanh'),
                        alpha=0.7, linewidth=2)
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Relative Improvement (%)')
    ax6.set_title('Convergence Rate Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle('LayerNorm vs TanhNorm Normalization Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ablation_normalization_comparison.png', dpi=100, bbox_inches='tight')
    print(f"\nPlot saved as ablation_normalization_comparison.png")
    plt.close()


def main():
    print("="*50)
    print("NORMALIZATION METHOD ABLATION")
    print("="*50)

    results = {}

    # Test LayerNorm
    results['layernorm'] = run_norm_ablation('layernorm', max_iters=50)

    # Test TanhNorm with default alpha (0.5 based on paper)
    results['tanhnorm'] = run_norm_ablation('tanhnorm', alpha=0.5, max_iters=50)

    # Test different alpha values
    alpha_results = test_alpha_values()
    results.update(alpha_results)

    # Print comparison
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)

    print(f"\n{'Method':<25} {'Step Time (ms)':>15} {'Final Loss':>12} {'Parameters':>12}")
    print("-" * 65)

    # Compare main methods
    ln_time = results['layernorm']['avg_step_time_ms']
    ln_loss = results['layernorm']['final_loss']
    ln_params = results['layernorm']['param_count']

    tn_time = results['tanhnorm']['avg_step_time_ms']
    tn_loss = results['tanhnorm']['final_loss']
    tn_params = results['tanhnorm']['param_count']

    print(f"{'LayerNorm':<25} {ln_time:>15.2f} {ln_loss:>12.4f} {ln_params:>12,}")
    print(f"{'TanhNorm (α=0.5)':<25} {tn_time:>15.2f} {tn_loss:>12.4f} {tn_params:>12,}")

    # Print alpha comparison
    print(f"\n{'TanhNorm Alpha Comparison':^65}")
    print("-" * 65)
    for key, result in results.items():
        if key.startswith('tanhnorm_alpha'):
            alpha = result['alpha']
            print(f"{'TanhNorm (α=' + str(alpha) + ')':<25} {result['avg_step_time_ms']:>15.2f} {result['final_loss']:>12.4f} {result['param_count']:>12,}")

    # Performance comparison
    time_diff = ((tn_time - ln_time) / ln_time) * 100
    loss_diff = ((tn_loss - ln_loss) / ln_loss) * 100

    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON (LayerNorm vs TanhNorm)")
    print("="*50)
    print(f"Speed difference: {time_diff:+.1f}% (negative = TanhNorm faster)")
    print(f"Loss difference: {loss_diff:+.1f}% (negative = TanhNorm better)")
    print(f"Parameter difference: {tn_params - ln_params:,} parameters")

    # Save results
    with open('ablation_normalization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to ablation_normalization_results.json")

    # Create plots
    plot_normalization_comparison(results)


if __name__ == "__main__":
    main()