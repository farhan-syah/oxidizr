# Training Data

This directory is for storing training data. You can put any tokenized data here, or use data from anywhere on your machine.

## What is Tokenization?

Language models work with numbers called **tokens**, not raw text. Tokenization converts text into token IDs:

```
"Hello world" → [9906, 1917]
```

To learn how tokenization works, see [splintr](https://github.com/farhan-syah/splintr).

## Data Format

oxidizr expects binary files containing u32 little-endian token IDs:
- No headers or metadata
- Just raw token IDs, 4 bytes each
- Tokenized with `cl100k_base` (vocab size: 100,331)

## Usage Examples

You can use data from anywhere - just pass the path to `--data`:

```bash
# Use data from this directory
cargo run --release -- --config models/nano.yaml --data data/nano-start/tokenized/combined.bin

# Use data from anywhere on your machine
cargo run --release -- --config models/nano.yaml --data /home/alice/datasets/my_data.bin
cargo run --release -- --config models/nano.yaml --data ~/projects/llm-data/wikipedia.bin
cargo run --release -- --config models/nano.yaml --data /mnt/storage/training/large_corpus.bin
```

## Available Datasets

| Dataset | Description | HuggingFace |
|---------|-------------|-------------|
| `nano-start/` | Small educational dataset for learning | [fs90/nano-start-data-bin](https://huggingface.co/datasets/fs90/nano-start-data-bin) |

## Directory Structure

```
data/
├── README.md
├── nano-start/          # Educational dataset
│   ├── README.md
│   └── tokenized/
│       └── combined.bin
└── my-dataset/          # Your own datasets
    └── tokens.bin
```
