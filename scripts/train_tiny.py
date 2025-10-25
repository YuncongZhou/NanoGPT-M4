#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import Trainer

def main():
    print("=" * 60)
    print("Training Tiny Shakespeare Model")
    print("=" * 60)

    # Override config for faster training during validation
    trainer = Trainer(
        config_path='configs/tiny_shakespeare.yaml',
        max_iters=1000,  # Reduced for quick validation
        eval_interval=100,
        log_interval=10
    )

    trainer.train()

if __name__ == "__main__":
    main()