#!/usr/bin/env python3
"""
Test ablation features in the model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from src.model import GPT, GPTConfig


class TestActivationAblations:
    """Test different activation functions."""

    def test_gelu_activation(self):
        config = GPTConfig(
            block_size=64,
            vocab_size=50,
            n_layer=1,
            n_head=2,
            n_embd=32,
            activation="gelu"
        )
        model = GPT(config)

        # Check that model has standard MLP components
        block = model.transformer.h[0]
        assert hasattr(block.mlp, 'c_fc')
        assert hasattr(block.mlp, 'c_proj')
        assert hasattr(block.mlp, 'gelu')

        # Test forward pass
        x = torch.randint(0, 50, (2, 32))
        logits, loss = model(x)
        assert logits.shape == (2, 1, 50)

    def test_swiglu_activation(self):
        config = GPTConfig(
            block_size=64,
            vocab_size=50,
            n_layer=1,
            n_head=2,
            n_embd=32,
            activation="swiglu"
        )
        model = GPT(config)

        # Check that model has SwiGLU components
        block = model.transformer.h[0]
        assert hasattr(block.mlp, 'w1')
        assert hasattr(block.mlp, 'w2')
        assert hasattr(block.mlp, 'w3')

        # Test forward pass
        x = torch.randint(0, 50, (2, 32))
        logits, loss = model(x)
        assert logits.shape == (2, 1, 50)

    def test_parameter_count_difference(self):
        # GELU model
        config_gelu = GPTConfig(
            block_size=64,
            vocab_size=50,
            n_layer=2,
            n_head=2,
            n_embd=32,
            activation="gelu"
        )
        model_gelu = GPT(config_gelu)
        params_gelu = model_gelu.get_num_params()

        # SwiGLU model
        config_swiglu = GPTConfig(
            block_size=64,
            vocab_size=50,
            n_layer=2,
            n_head=2,
            n_embd=32,
            activation="swiglu"
        )
        model_swiglu = GPT(config_swiglu)
        params_swiglu = model_swiglu.get_num_params()

        # SwiGLU should have more parameters
        assert params_swiglu > params_gelu
        print(f"GELU params: {params_gelu}, SwiGLU params: {params_swiglu}")


class TestNormalizationAblations:
    """Test different normalization methods."""

    def test_layernorm(self):
        config = GPTConfig(
            block_size=64,
            vocab_size=50,
            n_layer=1,
            n_head=2,
            n_embd=32,
            norm_type="layernorm"
        )
        model = GPT(config)

        # Check that model uses LayerNorm
        block = model.transformer.h[0]
        assert block.ln_1.__class__.__name__ == 'LayerNorm'
        assert block.ln_2.__class__.__name__ == 'LayerNorm'

        # Test forward pass
        x = torch.randint(0, 50, (2, 32))
        logits, loss = model(x)
        assert logits.shape == (2, 1, 50)

    def test_tanhnorm(self):
        config = GPTConfig(
            block_size=64,
            vocab_size=50,
            n_layer=1,
            n_head=2,
            n_embd=32,
            norm_type="tanhnorm",
            norm_alpha=0.5
        )
        model = GPT(config)

        # Check that model uses TanhNorm
        block = model.transformer.h[0]
        assert block.ln_1.__class__.__name__ == 'TanhNorm'
        assert block.ln_2.__class__.__name__ == 'TanhNorm'

        # Check alpha parameter is scalar
        assert block.ln_1.alpha.shape == torch.Size([])
        # Check gamma and beta are per-channel
        assert block.ln_1.gamma.shape == (32,)
        assert block.ln_1.beta.shape == (32,)

        # Test forward pass
        x = torch.randint(0, 50, (2, 32))
        logits, loss = model(x)
        assert logits.shape == (2, 1, 50)

    def test_tanhnorm_alpha_values(self):
        """Test different alpha values for TanhNorm."""
        for alpha in [0.2, 0.5, 0.8, 1.0]:
            config = GPTConfig(
                block_size=64,
                vocab_size=50,
                n_layer=1,
                n_head=2,
                n_embd=32,
                norm_type="tanhnorm",
                norm_alpha=alpha
            )
            model = GPT(config)

            # Check alpha initialization (scalar now)
            block = model.transformer.h[0]
            alpha_value = block.ln_1.alpha.item()
            assert abs(alpha_value - alpha) < 0.001

            # Test forward pass doesn't fail
            x = torch.randint(0, 50, (2, 32))
            logits, loss = model(x)
            assert not torch.isnan(logits).any()


class TestCombinedAblations:
    """Test combinations of ablations."""

    def test_swiglu_with_tanhnorm(self):
        config = GPTConfig(
            block_size=64,
            vocab_size=50,
            n_layer=2,
            n_head=2,
            n_embd=32,
            activation="swiglu",
            norm_type="tanhnorm",
            norm_alpha=0.5
        )
        model = GPT(config)

        # Check both features are present
        block = model.transformer.h[0]
        assert hasattr(block.mlp, 'w1')  # SwiGLU
        assert block.ln_1.__class__.__name__ == 'TanhNorm'

        # Test forward pass
        x = torch.randint(0, 50, (2, 32))
        targets = torch.randint(0, 50, (2, 32))
        logits, loss = model(x, targets)

        assert logits.shape == (2, 32, 50)
        assert loss.item() > 0
        assert not torch.isnan(loss)


if __name__ == "__main__":
    print("Testing Activation Ablations...")
    test_activation = TestActivationAblations()
    test_activation.test_gelu_activation()
    print("✓ GELU activation test passed")
    test_activation.test_swiglu_activation()
    print("✓ SwiGLU activation test passed")
    test_activation.test_parameter_count_difference()
    print("✓ Parameter count difference test passed")

    print("\nTesting Normalization Ablations...")
    test_norm = TestNormalizationAblations()
    test_norm.test_layernorm()
    print("✓ LayerNorm test passed")
    test_norm.test_tanhnorm()
    print("✓ TanhNorm test passed")
    test_norm.test_tanhnorm_alpha_values()
    print("✓ TanhNorm alpha values test passed")

    print("\nTesting Combined Ablations...")
    test_combined = TestCombinedAblations()
    test_combined.test_swiglu_with_tanhnorm()
    print("✓ SwiGLU + TanhNorm combination test passed")

    print("\n✅ All ablation tests passed!")