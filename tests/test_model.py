#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from src.model import GPT, GPTConfig


class TestGPTModel:
    def test_model_initialization(self):
        config = GPTConfig(
            block_size=128,
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=128,
            dropout=0.1
        )
        model = GPT(config)
        assert model is not None
        assert model.config.block_size == 128
        assert model.config.vocab_size == 100

    def test_forward_pass(self):
        config = GPTConfig(
            block_size=128,
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=128,
            dropout=0.0
        )
        model = GPT(config)
        model.eval()

        batch_size = 2
        seq_len = 64
        idx = torch.randint(0, 100, (batch_size, seq_len))

        with torch.no_grad():
            logits, loss = model(idx)

        assert logits.shape == (batch_size, 1, 100)
        assert loss is None

    def test_forward_with_targets(self):
        config = GPTConfig(
            block_size=128,
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=128,
            dropout=0.0
        )
        model = GPT(config)
        model.eval()

        batch_size = 2
        seq_len = 64
        idx = torch.randint(0, 100, (batch_size, seq_len))
        targets = torch.randint(0, 100, (batch_size, seq_len))

        with torch.no_grad():
            logits, loss = model(idx, targets)

        assert logits.shape == (batch_size, seq_len, 100)
        assert loss is not None
        assert loss.item() > 0

    def test_generate(self):
        config = GPTConfig(
            block_size=128,
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=128,
            dropout=0.0
        )
        model = GPT(config)
        model.eval()

        idx = torch.zeros((1, 1), dtype=torch.long)
        generated = model.generate(idx, max_new_tokens=10, temperature=1.0)

        assert generated.shape == (1, 11)
        assert generated.dtype == torch.long

    def test_mps_device(self):
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        config = GPTConfig(
            block_size=64,
            vocab_size=50,
            n_layer=1,
            n_head=2,
            n_embd=64,
            device="mps"
        )
        model = GPT(config)
        model.to("mps")

        idx = torch.randint(0, 50, (1, 32)).to("mps")
        with torch.no_grad():
            logits, loss = model(idx)

        assert logits.device.type == "mps"

    def test_parameter_count(self):
        config = GPTConfig(
            block_size=128,
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=128
        )
        model = GPT(config)

        num_params = model.get_num_params()
        assert num_params > 0
        assert num_params < 10_000_000  # Less than 10M params for tiny model


if __name__ == "__main__":
    test = TestGPTModel()
    test.test_model_initialization()
    print("✓ Model initialization test passed")
    test.test_forward_pass()
    print("✓ Forward pass test passed")
    test.test_forward_with_targets()
    print("✓ Forward with targets test passed")
    test.test_generate()
    print("✓ Generation test passed")
    test.test_parameter_count()
    print("✓ Parameter count test passed")
    if torch.backends.mps.is_available():
        test.test_mps_device()
        print("✓ MPS device test passed")
    print("\nAll tests passed!")