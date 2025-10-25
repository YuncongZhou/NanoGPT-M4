import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Tuple, List
import requests
from pathlib import Path


class TextDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        block_size: int,
        device: str = "cpu",
    ):
        self.data = data
        self.block_size = block_size
        self.device = device

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class ShakespeareDataset:
    def __init__(self, data_dir: str = "data", block_size: int = 256):
        self.data_dir = Path(data_dir)
        self.block_size = block_size
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        self.file_path = self.data_dir / "shakespeare.txt"

    def download(self):
        if not self.file_path.exists():
            print(f"Downloading Shakespeare dataset...")
            response = requests.get(self.url)
            with open(self.file_path, 'w') as f:
                f.write(response.text)
            print(f"Dataset saved to {self.file_path}")
        else:
            print(f"Dataset already exists at {self.file_path}")

    def load_and_split(self, tokenizer, train_ratio: float = 0.9):
        self.download()

        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        if hasattr(tokenizer, 'train'):
            tokenizer.train(text)

        data = np.array(tokenizer.encode(text), dtype=np.uint16)

        n = len(data)
        train_data = data[:int(n * train_ratio)]
        val_data = data[int(n * train_ratio):]

        print(f"Train size: {len(train_data):,} tokens")
        print(f"Val size: {len(val_data):,} tokens")
        print(f"Vocab size: {tokenizer.vocab_size}")

        return train_data, val_data, tokenizer


class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, pin_memory=False, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.reset()

    def reset(self):
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            drop_last=True
        )
        self.iter = iter(self.loader)

    def next(self):
        try:
            batch = next(self.iter)
        except StopIteration:
            self.reset()
            batch = next(self.iter)
        return batch


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device: str = "cpu"):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device != 'cpu':
        x, y = x.to(device), y.to(device)
    return x, y