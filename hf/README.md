# HuggingFace Model Packages

This directory contains packaged models ready for upload to HuggingFace Hub.

## Structure

```
hf/
├── README.md           # This file
└── <username>/
    └── <model-name>/
        ├── model.safetensors     # Model weights
        ├── config.json           # Inference configuration
        ├── training_config.json  # Training configuration (optional)
        └── README.md             # Auto-generated model card
```

## Usage

### 1. Pack a trained model

```bash
# Interactive mode - select checkpoint from list
oxidizr pack

# Non-interactive - use specific checkpoint
oxidizr pack --checkpoint latest
oxidizr pack --checkpoint final
oxidizr pack --checkpoint 10000  # specific step

# Custom options
oxidizr pack \
  --checkpoint-dir ./checkpoints \
  --checkpoint final \
  --name my-model \
  --username my-hf-username
```

### 2. Push to HuggingFace (requires `--features huggingface`)

```bash
# Build with huggingface feature
cargo build --release --features huggingface

# Interactive mode - select model from list
oxidizr push

# Non-interactive - specify model path
oxidizr push --model hf/username/model-name

# Options
oxidizr push --model hf/username/model-name --private  # Create private repo
```

## Environment Variables

Create a `.env` file in the project root with:

```env
HF_USERNAME=your-username
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Or pass values via CLI flags:
- `--username` for pack command
- `--token` for push command

## Note

This directory is tracked by git, but model contents (`hf/*/`) are ignored.
Only this README.md is committed to the repository.
