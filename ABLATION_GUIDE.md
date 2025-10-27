# Ablation Studies Guide

This nanoGPT implementation includes support for architectural ablation studies to compare different design choices.

## Supported Ablations

### 1. Activation Functions
Compare GELU vs SwiGLU activation functions in the MLP layers.

**Configuration:**
- `activation: "gelu"` - Standard GELU activation (default)
- `activation: "swiglu"` - SwiGLU activation (Swish-gated linear unit)

**Key Differences:**
- **GELU**: Single projection → GELU → projection
- **SwiGLU**: Two projections → element-wise multiply with Swish(first) → final projection
- SwiGLU has ~33% more parameters but can be faster on modern hardware

### 2. Normalization Methods
Compare standard LayerNorm vs Dynamic Tanh (DyT) normalization.

**Configuration:**
- `norm_type: "layernorm"` - Standard LayerNorm (default)
- `norm_type: "tanhnorm"` - Dynamic Tanh normalization (DyT)
- `norm_alpha: 0.5` - Alpha parameter for tanh normalization (paper default)

**Key Differences:**
- **LayerNorm**: Normalizes inputs to zero mean and unit variance with learnable scale/shift
- **TanhNorm (DyT)**: Applies γ * tanh(α * x) + β where:
  - α is a learnable scalar that controls the input scale
  - γ and β are learnable per-channel scale and shift (like LayerNorm)
- TanhNorm is computationally simpler but requires careful tuning of α

## Running Ablation Studies

### Quick Activation Comparison
```bash
python ablation_simple.py
```
Compares GELU vs SwiGLU on a small model for 50 iterations.

### Normalization Comparison
```bash
python ablation_normalization.py
```
Compares LayerNorm vs TanhNorm with different alpha values.

### Full Activation Study
```bash
python ablation_activation.py
```
Runs longer training with plots (requires matplotlib).

## Configuration Examples

### Using SwiGLU in Training
```yaml
# configs/swiglu_model.yaml
model:
  activation: swiglu
  n_layer: 6
  n_head: 6
  n_embd: 384
```

### Using TanhNorm
```yaml
# configs/tanhnorm_model.yaml
model:
  norm_type: tanhnorm
  norm_alpha: 0.5  # Paper default, tune between 0.2-1.0
  n_layer: 4
  n_head: 4
  n_embd: 256
```

### Combining Both
```python
from src.model import GPTConfig, GPT

config = GPTConfig(
    activation="swiglu",
    norm_type="tanhnorm",
    norm_alpha=0.1,
    n_layer=4,
    n_head=4,
    n_embd=128
)
model = GPT(config)
```

## Results Summary

From initial testing on Shakespeare dataset:

### Activation Functions
| Metric | GELU | SwiGLU | Difference |
|--------|------|--------|------------|
| Step Time | 12.00ms | 4.67ms | -61% (SwiGLU faster) |
| Final Loss | 3.42 | 3.50 | +2.5% |
| Parameters | 104K | 138K | +33K |

### Normalization Methods
| Metric | LayerNorm | TanhNorm (α=0.05) | Difference |
|--------|-----------|-------------------|------------|
| Step Time | 7.65ms | 5.66ms | -26% (TanhNorm faster) |
| Final Loss | 3.52 | 4.15 | +18% (LayerNorm better) |
| Parameters | 104K | 104K | -320 |

## Observations

1. **SwiGLU**: Faster despite more parameters, slightly worse convergence in early training
2. **TanhNorm**: Faster but significantly slower convergence, may need more iterations or different learning rates
3. Both alternative methods show speed improvements but may require tuning

## Adding New Ablations

To add new ablation options:

1. Add configuration parameter to `GPTConfig` in `src/model.py`
2. Implement the variant in the appropriate class
3. Create ablation script following the pattern in `ablation_simple.py`
4. Document results in this guide

## Tips for Ablation Studies

1. **Control Variables**: Keep all other hyperparameters constant
2. **Multiple Seeds**: Run with different random seeds for robust comparison
3. **Sufficient Training**: Some methods may need more iterations to converge
4. **Hardware Considerations**: Results may vary on different hardware (MPS vs CUDA)
5. **Learning Rate**: Different architectures may need different learning rates