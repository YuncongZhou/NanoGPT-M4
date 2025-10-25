#!/usr/bin/env python3
"""Quick training script for validation"""

import torch
import numpy as np
import time
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent))

from src.model import GPT, GPTConfig
from src.dataset import ShakespeareDataset, get_batch
from src.tokenizer import get_tokenizer

def quick_train():
    # Setup
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    print("Loading dataset...")
    dataset = ShakespeareDataset(data_dir='data', block_size=128)
    tokenizer = get_tokenizer('character')
    train_data, val_data, tokenizer = dataset.load_and_split(tokenizer, train_ratio=0.9)

    # Create tiny model
    print("Creating model...")
    config = GPTConfig(
        block_size=128,
        vocab_size=tokenizer.vocab_size,
        n_layer=3,
        n_head=3,
        n_embd=48,
        dropout=0.1,
        device=device
    )
    model = GPT(config)
    model.to(device)

    # Training settings
    batch_size = 8
    max_iters = 100
    learning_rate = 1e-3
    eval_interval = 20

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"Starting training for {max_iters} iterations...")
    losses = []
    model.train()

    for iter_num in range(max_iters):
        # Get batch
        X, Y = get_batch(train_data, batch_size, config.block_size, device)

        # Forward pass
        logits, loss = model(X, Y)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Logging
        if iter_num % 10 == 0:
            print(f"Iter {iter_num:4d} | Loss: {loss.item():.4f}")
            losses.append(loss.item())

        # Evaluation
        if iter_num % eval_interval == 0 and iter_num > 0:
            model.eval()
            with torch.no_grad():
                X_val, Y_val = get_batch(val_data, batch_size, config.block_size, device)
                _, val_loss = model(X_val, Y_val)
                print(f"  Val loss: {val_loss.item():.4f}")

                # Generate sample
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                generated = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=40)
                text = tokenizer.decode(generated[0].tolist())
                print(f"  Sample: {text[:80]}...")

            model.train()

    print("\nTraining complete!")

    # Save checkpoint
    checkpoint_dir = Path('checkpoints/quick_test')
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    checkpoint = {
        'model': model.state_dict(),
        'model_args': {
            'block_size': config.block_size,
            'vocab_size': config.vocab_size,
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_embd': config.n_embd,
            'dropout': config.dropout,
        },
        'iter_num': max_iters,
        'losses': losses
    }
    torch.save(checkpoint, checkpoint_dir / 'ckpt.pt')
    if hasattr(tokenizer, 'save'):
        tokenizer.save(str(checkpoint_dir / 'tokenizer.json'))

    print(f"Model saved to {checkpoint_dir}")

    # Final generation
    print("\nFinal text generation:")
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=40)
    text = tokenizer.decode(generated[0].tolist())
    print(text)

    return model, tokenizer

if __name__ == "__main__":
    quick_train()