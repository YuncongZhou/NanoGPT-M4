#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import ShakespeareDataset
from src.tokenizer import get_tokenizer

def main():
    print("Downloading Shakespeare dataset...")
    dataset = ShakespeareDataset(data_dir='data')
    dataset.download()

    print("\nPreparing tokenizer and splitting data...")
    tokenizer = get_tokenizer('character')
    train_data, val_data, tokenizer = dataset.load_and_split(tokenizer)

    print("\nDataset preparation complete!")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Sample characters: {list(tokenizer.stoi.keys())[:20]}")

if __name__ == "__main__":
    main()