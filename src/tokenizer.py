import torch
import tiktoken
import numpy as np
from typing import List, Optional, Union


class CharacterTokenizer:
    def __init__(self, chars: Optional[str] = None):
        self.chars = chars if chars else ""
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0
        if chars:
            self._build_vocab(chars)

    def _build_vocab(self, chars: str):
        chars = sorted(list(set(chars)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def train(self, text: str):
        chars = sorted(list(set(text)))
        self._build_vocab(''.join(chars))

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids: List[int]) -> str:
        return ''.join([self.itos.get(i, '') for i in ids])

    def save(self, path: str):
        import json
        with open(path, 'w') as f:
            json.dump({
                'chars': self.chars,
                'vocab_size': self.vocab_size,
                'stoi': self.stoi,
                'itos': {str(k): v for k, v in self.itos.items()}
            }, f)

    @classmethod
    def load(cls, path: str):
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        tokenizer = cls()
        tokenizer.chars = data['chars']
        tokenizer.vocab_size = data['vocab_size']
        tokenizer.stoi = data['stoi']
        tokenizer.itos = {int(k): v for k, v in data['itos'].items()}
        return tokenizer


class BPETokenizer:
    def __init__(self, encoding_name: str = "gpt2"):
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, ids: List[int]) -> str:
        return self.enc.decode(ids)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(text) for text in texts]

    def decode_batch(self, ids_batch: List[List[int]]) -> List[str]:
        return [self.decode(ids) for ids in ids_batch]


def get_tokenizer(tokenizer_type: str = "character", **kwargs):
    if tokenizer_type == "character":
        return CharacterTokenizer(**kwargs)
    elif tokenizer_type == "bpe":
        return BPETokenizer(**kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


class DataCollator:
    def __init__(self, block_size: int, device: str = "cpu"):
        self.block_size = block_size
        self.device = device

    def __call__(self, batch):
        max_len = max(len(item['input_ids']) for item in batch)
        max_len = min(max_len, self.block_size + 1)

        input_ids_list = []
        labels_list = []

        for item in batch:
            ids = item['input_ids'][:max_len]
            if len(ids) < max_len:
                padding = [0] * (max_len - len(ids))
                ids = ids + padding

            input_ids_list.append(ids[:-1])
            labels_list.append(ids[1:])

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=self.device)
        labels = torch.tensor(labels_list, dtype=torch.long, device=self.device)

        return {
            'input_ids': input_ids,
            'labels': labels
        }