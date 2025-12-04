# Nano-Start Dataset

A small educational dataset for learning how to train language models with oxidizr.

## What is Tokenization?

Language models don't process text directly - they work with numbers called **tokens**. Tokenization converts text into token IDs:

```
"Hello world" → [9906, 1917]
```

This dataset is **pre-tokenized** for simplicity - download and start training immediately. To learn how tokenization works and create your own datasets, see the [splintr](https://github.com/farhan-syah/splintr) project.

## Quick Start

### 1. Download

**Option A: Using hf**
```bash
pip install huggingface_hub
hf download fs90/nano-start-data-bin --local-dir tokenized --repo-type dataset
```

**Option B: Direct download**

Download `combined.bin` from [fs90/nano-start-data-bin](https://huggingface.co/datasets/fs90/nano-start-data-bin/tree/main) and place it in `tokenized/`.

### 2. Train

```bash
cargo run --release -- \
    --config models/nano-start.yaml \
    --data data/nano-start/tokenized/combined.bin
```

## Files

The recommended file is `combined.bin` which contains all training data merged together:

| File | Tokens | Description |
|------|--------|-------------|
| **`combined.bin`** | 6,379 | **All data merged (recommended)** |

### Training with Individual Files (Optional)

You can also train on individual subsets for different results:

| File | Tokens | Description |
|------|--------|-------------|
| `completions.bin` | 2,197 | Factual statements only |
| `qa.bin` | 2,759 | Q&A pairs only |
| `chat.bin` | 1,423 | Multi-turn conversations only |

Training on different data produces different model behavior - experiment to see the differences!

## View Raw Data

To see the human-readable text before tokenization:

**Option A: Using hf**
```bash
hf download fs90/nano-start-data --local-dir raw --repo-type dataset
```

**Option B: Direct download**

Browse and download from [fs90/nano-start-data](https://huggingface.co/datasets/fs90/nano-start-data/tree/main)

## Binary Format

The `.bin` files contain raw token IDs:
- u32 little-endian encoding
- No headers or metadata
- Tokenized with `cl100k_base` (vocab size: 100,331)

## Special Tokens

| Token | ID | Purpose |
|-------|------|---------|
| `<\|endoftext\|>` | 100257 | End of document |
| `<\|system\|>` | 100277 | System instructions |
| `<\|user\|>` | 100278 | User input |
| `<\|assistant\|>` | 100279 | Model response |

## Testing Your Model

After training, use blazr for inference:

```bash
blazr generate \
    --model checkpoints/nano-start \
    --prompt "The capital of France is" \
    --max-tokens 20
```

## Learn More

- **Tokenization**: [splintr](https://github.com/farhan-syah/splintr) - Learn how to tokenize your own data
- **Raw data**: [fs90/nano-start-data](https://huggingface.co/datasets/fs90/nano-start-data) - See the original text
