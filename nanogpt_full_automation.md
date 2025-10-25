# Complete nanoGPT Setup and Training Automation Instructions

## Mission Objective
Fully automate the setup, configuration, coding, training, and deployment of nanoGPT on an M4 MacBook Air with 24GB RAM. Complete all tasks independently without user intervention, creating a working end-to-end transformer training pipeline.

## Core Requirements
- Create a complete, working nanoGPT implementation from scratch
- Optimize for Apple Silicon M4 with MPS acceleration
- Handle all edge cases and errors gracefully
- Generate comprehensive documentation and examples
- Complete all tasks in a single uninterrupted session

## Phase 1: Environment Setup (30 minutes)

### 1.1 Create Project Structure
```
~/nanogpt-m4/
├── src/              # Core implementation
├── data/             # Datasets
├── checkpoints/      # Model saves
├── configs/          # Configuration files
├── scripts/          # Utility scripts
├── tests/            # Test suite
├── logs/             # Training logs
└── examples/         # Demo scripts
```

### 1.2 Python Environment
1. Create virtual environment with Python 3.11
2. Create requirements.txt with all dependencies
3. Install PyTorch with MPS support
4. Verify MPS availability and functionality

### 1.3 System Checks
- Verify available memory
- Check disk space
- Test MPS device allocation
- Create system compatibility report

## Phase 2: Core Implementation (2 hours)

### 2.1 Model Architecture
Create from scratch or adapt from repository:
- `model.py`: Transformer architecture with MPS optimizations
- `attention.py`: Multi-head attention implementation
- `feedforward.py`: FFN layers
- `positional.py`: Positional encodings
- `utils.py`: Helper functions

### 2.2 Training Pipeline
Implement complete training infrastructure:
- `train.py`: Main training loop with:
  - Memory-efficient gradient accumulation
  - Mixed precision training (if supported)
  - Checkpoint saving/loading
  - Early stopping
  - Learning rate scheduling
  - Gradient clipping
  
### 2.3 Data Processing
- `data_loader.py`: Efficient data loading
- `tokenizer.py`: Character and BPE tokenization
- `preprocessing.py`: Data preparation utilities

### 2.4 MPS Optimizations
- Device management for Apple Silicon
- Memory optimization techniques
- Batch size auto-tuning
- Fallback to CPU when needed

## Phase 3: Configuration System (30 minutes)

### 3.1 Create Configuration Files
Generate multiple configs for different scenarios:

`configs/tiny_shakespeare.yaml`:
```yaml
model:
  n_layer: 6
  n_head: 6
  n_embd: 384
  dropout: 0.2
  block_size: 256

training:
  batch_size: 12
  learning_rate: 3e-4
  max_iters: 5000
  warmup_iters: 100
  eval_interval: 500
  gradient_accumulation: 4
  
device:
  type: mps
  mixed_precision: false
  compile: false
```

`configs/small_model.yaml` (for testing)
`configs/medium_model.yaml` (for serious training)
`configs/large_model.yaml` (max for 24GB)

### 3.2 Dynamic Configuration
- Automatic batch size finder
- Memory usage predictor
- Performance profiler

## Phase 4: Dataset Preparation (30 minutes)

### 4.1 Download and Prepare Datasets
1. Shakespeare character-level dataset
2. OpenWebText sample (if feasible)
3. Custom test datasets
4. Create train/val/test splits

### 4.2 Tokenization
- Implement character-level tokenizer
- Implement BPE tokenizer using tiktoken
- Create vocabulary files
- Generate token statistics

## Phase 5: Training Execution (2-4 hours)

### 5.1 Progressive Training Strategy
1. Start with tiny model (5M params) for validation
2. Train small model (50M params) 
3. Train medium model (125M params) if memory allows
4. Document performance metrics for each

### 5.2 Training Monitoring
- Real-time loss plotting
- Memory usage tracking
- Temperature sampling during training
- Perplexity calculation
- Generate sample outputs every N steps

### 5.3 Checkpoint Management
- Save best model
- Save periodic checkpoints
- Implement checkpoint averaging
- Create model cards with metadata

## Phase 6: Evaluation and Testing (1 hour)

### 6.1 Model Evaluation
- Calculate perplexity on test set
- Generate diverse text samples
- Measure inference speed
- Profile memory usage

### 6.2 Test Suite
Create comprehensive tests:
- `test_model.py`: Architecture tests
- `test_training.py`: Training pipeline tests
- `test_data.py`: Data loading tests
- `test_generation.py`: Text generation tests

### 6.3 Benchmarking
- Compare with baseline models
- Document training curves
- Create performance report

## Phase 7: Inference and Deployment (1 hour)

### 7.1 Inference Script
Create `generate.py` with:
- Interactive text generation
- Temperature and top-k/top-p sampling
- Batch generation
- Streaming generation

### 7.2 Model Optimization
- Quantization exploration (if supported)
- Model pruning experiments
- Optimization for inference

### 7.3 Demo Applications
Create example scripts:
- `chat_bot.py`: Interactive chatbot
- `complete.py`: Text completion
- `creative_writing.py`: Story generation

## Phase 8: Documentation and Reporting (30 minutes)

### 8.1 Generate Documentation
- README.md with full instructions
- API documentation
- Training best practices
- Troubleshooting guide

### 8.2 Create Final Report
Generate `training_report.md` with:
- System specifications
- Training statistics
- Performance metrics
- Memory usage analysis
- Sample outputs
- Recommendations for improvement

## Error Handling Strategy

### Common Issues and Solutions
1. **OOM Errors**: Automatically reduce batch size and retry
2. **MPS Errors**: Fall back to CPU with warning
3. **Data Corruption**: Validate and re-download if needed
4. **Training Instability**: Adjust learning rate and clip gradients
5. **Checkpoint Corruption**: Maintain multiple backup checkpoints

## Success Criteria
- [ ] Complete environment setup without errors
- [ ] All code files created and tested
- [ ] At least one model successfully trained
- [ ] Loss decreases over training iterations
- [ ] Model generates coherent text
- [ ] All tests pass
- [ ] Documentation complete
- [ ] Final report generated

## Time Management
- Total estimated time: 6-8 hours
- Use parallel tasks where possible
- Prioritize working implementation over perfection
- If time constrained, focus on small model success

## Final Deliverables
1. Complete codebase in ~/nanogpt-m4/
2. Trained model checkpoints
3. Performance benchmarks
4. Sample generated texts
5. Full documentation
6. Reproducible training scripts

## Notes for Automation
- Log all actions to `automation_log.txt`
- Create progress markers for restart capability
- Use verbose output for debugging
- Test each component before integration
- Commit to git after major milestones (if git initialized)

Remember: The goal is a fully functional, well-documented, and optimized nanoGPT implementation that leverages the M4 MacBook Air's capabilities effectively.