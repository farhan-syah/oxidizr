# Nano-Start Educational Dataset

The nano-start dataset is a small, curated collection designed to help you learn LLM training fundamentals.

## What is nano-start?

A pre-tokenized dataset ready for immediate training:
- Small enough to train quickly (~6K tokens)
- Diverse enough to demonstrate key concepts
- Real data, not random noise

## Quick Start

### Download

**Option A: Using hf CLI**
```bash
pip install huggingface_hub
hf download fs90/nano-start-data-bin --local-dir data/nano-start/tokenized --repo-type dataset
```

**Option B: Direct download**

Download from [HuggingFace](https://huggingface.co/datasets/fs90/nano-start-data-bin) and place in `data/nano-start/tokenized/`.

### Train

```bash
cargo run --release -- -f models/nano.yaml -d data/nano-start/tokenized/combined.bin
```

## Dataset Contents

### combined.bin (Recommended)

All data merged together:
- Size: 25,516 bytes
- Tokens: 6,379

Use this for training.

### Individual Files (Optional)

Experiment with different data types:

| File | Tokens | Content |
|------|--------|---------|
| `completions.bin` | 2,197 | Factual statements |
| `qa.bin` | 2,759 | Q&A pairs |
| `chat.bin` | 1,423 | Multi-turn conversations |

## Data Types

### Completions

Simple factual statements the model learns to complete:

```
The capital of France is Paris.
Water boils at 100 degrees Celsius.
```

### Q&A

Question-answer pairs with special tokens:

```
<|user|>What is the capital of France?<|assistant|>Paris<|endoftext|>
```

### Chat

Multi-turn conversations with system prompts:

```
<|system|>You are a helpful assistant.<|user|>Hello!<|assistant|>Hi there!<|endoftext|>
```

## Training with nano-start

### Basic Training

```bash
cargo run --release -- -f models/nano.yaml -d data/nano-start/tokenized/combined.bin
```

### With Options

```bash
cargo run --release -- \
    -f models/nano.yaml \
    -d data/nano-start/tokenized/combined.bin \
    --batch-size 2 \
    --max-steps 1000 \
    --headless
```

### Watch Training

```bash
# Interactive progress bar
cargo run --release -- -f models/nano.yaml -d data/nano-start/tokenized/combined.bin

# Or JSON output
cargo run --release -- -f models/nano.yaml -d data/nano-start/tokenized/combined.bin --headless
```

## Expected Results

With nano-start on the nano model:

- Initial loss: ~11.5
- After 100 steps: ~9-10
- After 1000 steps: ~6-7
- Training speed: ~3 it/s (CPU), ~50+ it/s (GPU)

The model learns to:
1. Associate related concepts
2. Complete factual statements
3. Follow Q&A patterns

## Why nano-start?

### For Learning

- See how loss decreases during training
- Understand the training loop
- Experiment with hyperparameters
- Quick iteration cycles

### For Testing

- Verify your setup works
- Benchmark performance
- Test new configs
- Debug issues

### Philosophy

The nano project follows these principles:
- **Zero magic** - Full visibility into training
- **Small scale** - Fast experiments
- **Real data** - Not random noise

## Raw Data

Want to see the original text before tokenization?

Visit [fs90/nano-start-data](https://huggingface.co/datasets/fs90/nano-start-data) on HuggingFace.

## Creating Similar Data

To create your own educational dataset:

1. Write examples in text/JSONL format
2. Tokenize with splintr
3. Save as binary

See [Tokenization](tokenization.md) for details.

## Sample Configs

The `models/` directory includes configs optimized for nano-start:

- `nano.yaml` - Standard transformer
- `nano_mamba2.yaml` - Hybrid Mamba2 + MLA
- `nano_mamba2_pure.yaml` - Pure Mamba2

All work with nano-start data.
