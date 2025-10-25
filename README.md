# nanoGPT for M4 MacBook Air

A complete implementation of nanoGPT optimized for Apple Silicon M4 MacBook Air with 24GB RAM. This project provides a fully functional transformer language model training pipeline with MPS (Metal Performance Shaders) acceleration.

## Features

- ✅ Complete GPT implementation from scratch
- ✅ Optimized for Apple Silicon M4 with MPS support
- ✅ Character-level and BPE tokenization
- ✅ Multiple model configurations (tiny, small, medium, large)
- ✅ Memory-efficient training pipeline
- ✅ Interactive text generation
- ✅ Comprehensive test suite
- ✅ Training visualization and logging

## Quick Start

### 1. Installation

```bash
# Clone the repository (if using git)
cd ~/nanogpt-m4

# Activate virtual environment
source venv/bin/activate

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. Verify MPS Support

```bash
python scripts/verify_mps.py
```

### 3. Quick Training

```bash
# Train a tiny model (fastest, ~5 minutes)
python quick_train.py

# Or train with specific configuration
python train.py --config configs/tiny_shakespeare.yaml --max_iters 1000
```

### 4. Generate Text

```bash
# Interactive generation
python src/generate.py --checkpoint checkpoints/quick_test --interactive

# Generate from prompt
python src/generate.py --checkpoint checkpoints/quick_test \
    --prompt "To be or not to be" --max_tokens 100
```

## Project Structure

```
nanogpt-m4/
├── src/                  # Core implementation
│   ├── model.py         # GPT architecture
│   ├── train.py         # Training pipeline
│   ├── dataset.py       # Data loading
│   ├── tokenizer.py     # Tokenization
│   └── generate.py      # Text generation
├── configs/             # Model configurations
│   ├── tiny_shakespeare.yaml
│   ├── small_model.yaml
│   ├── medium_model.yaml
│   └── large_model.yaml
├── data/                # Datasets
├── checkpoints/         # Saved models
├── tests/               # Test suite
├── scripts/             # Utility scripts
└── logs/                # Training logs
```

## Model Configurations

### Tiny Model (0.4M params)
- **Purpose**: Quick testing and validation
- **Training time**: ~5-10 minutes
- **Memory**: <2GB
- **Config**: `configs/tiny_shakespeare.yaml`

### Small Model (2M params)
- **Purpose**: Experimentation
- **Training time**: ~30 minutes
- **Memory**: ~4GB
- **Config**: `configs/small_model.yaml`

### Medium Model (25M params)
- **Purpose**: Serious training
- **Training time**: ~2 hours
- **Memory**: ~8GB
- **Config**: `configs/medium_model.yaml`

### Large Model (125M params)
- **Purpose**: Maximum capability
- **Training time**: ~8 hours
- **Memory**: ~20GB
- **Config**: `configs/large_model.yaml`

## Training Examples

### Basic Training
```bash
python train.py --config configs/tiny_shakespeare.yaml
```

### Custom Parameters
```bash
python train.py --config configs/small_model.yaml \
    --max_iters 5000 \
    --batch_size 8 \
    --device mps
```

### Resume Training
```bash
python train.py --config configs/tiny_shakespeare.yaml \
    --init_from resume
```

## Text Generation

### Command Line
```bash
# Generate text
python src/generate.py --checkpoint checkpoints/tiny_shakespeare/ckpt.pt \
    --prompt "ROMEO:" --max_tokens 200 --temperature 0.8
```

### Interactive Mode
```bash
python src/generate.py --checkpoint checkpoints/tiny_shakespeare/ckpt.pt \
    --interactive
```

### Python API
```python
from src.generate import Generator

# Load model
gen = Generator("checkpoints/tiny_shakespeare/ckpt.pt")

# Generate text
samples = gen.generate(
    prompt="To be or not to be",
    max_new_tokens=100,
    temperature=0.8,
    top_k=40
)
print(samples[0])
```

## Testing

Run the test suite:
```bash
# All tests
pytest tests/

# Specific tests
python tests/test_model.py
python tests/test_tokenizer.py
```

## Performance Tips

1. **MPS Optimization**: The code automatically uses MPS when available
2. **Batch Size**: Adjust based on available memory
3. **Gradient Accumulation**: Use for effective larger batch sizes
4. **Mixed Precision**: Currently disabled for MPS stability

## Memory Management

For 24GB M4 MacBook Air:
- Tiny model: Use batch_size=12-16
- Small model: Use batch_size=8-12
- Medium model: Use batch_size=4-8
- Large model: Use batch_size=2-4

## Troubleshooting

### Out of Memory
- Reduce batch_size in config
- Increase gradient_accumulation_steps
- Use smaller model configuration

### MPS Issues
- Verify MPS with `scripts/verify_mps.py`
- Fall back to CPU if needed: `--device cpu`

### Slow Training
- Ensure you're using MPS device
- Check Activity Monitor for memory pressure
- Close other applications

## Advanced Features

### Custom Tokenizer
```python
from src.tokenizer import CharacterTokenizer

tokenizer = CharacterTokenizer()
tokenizer.train(your_text)
tokenizer.save("custom_tokenizer.json")
```

### Model Export
```python
# Save for inference
torch.save(model.state_dict(), "model_weights.pt")

# ONNX export (if needed)
torch.onnx.export(model, dummy_input, "model.onnx")
```

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Based on Andrej Karpathy's nanoGPT
- Optimized for Apple Silicon by this implementation
- Uses PyTorch with MPS backend