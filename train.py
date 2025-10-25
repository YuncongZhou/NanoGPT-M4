#!/usr/bin/env python3
"""Main training script for nanoGPT"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.train import Trainer
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train nanoGPT model')
    parser.add_argument('--config', type=str, default='configs/tiny_shakespeare.yaml', help='Path to config file')
    parser.add_argument('--device', type=str, default=None, help='Device to use')
    parser.add_argument('--max_iters', type=int, default=None, help='Maximum iterations')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--compile', action='store_true', help='Compile model')
    args = parser.parse_args()

    kwargs = {}
    if args.device:
        kwargs['device'] = args.device
    if args.max_iters:
        kwargs['max_iters'] = args.max_iters
    if args.batch_size:
        kwargs['batch_size'] = args.batch_size
    if args.compile:
        kwargs['compile'] = True

    print("=" * 60)
    print(f"Training nanoGPT with config: {args.config}")
    print("=" * 60)

    trainer = Trainer(config_path=args.config, **kwargs)
    trainer.train()

if __name__ == "__main__":
    main()