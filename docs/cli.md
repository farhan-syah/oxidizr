# CLI Reference

Complete command-line options for Oxidizr.

## Basic Usage

```bash
oxidizr -f <config.yaml> [OPTIONS]
```

Or when running from source:

```bash
cargo run --release -- -f <config.yaml> [OPTIONS]
```

## Required Arguments

| Argument              | Description                     |
| --------------------- | ------------------------------- |
| `-f, --config <FILE>` | Path to YAML configuration file |

## Data Options

| Argument            | Description                                                                 |
| ------------------- | --------------------------------------------------------------------------- |
| `-d, --data <FILE>` | Path to tokenized data file (.bin). If not specified, generates dummy data. |

## Device Options

| Argument                     | Description                                                           |
| ---------------------------- | --------------------------------------------------------------------- |
| `--target-device <cpu\|gpu>` | Override target device. Default: gpu if CUDA available, otherwise cpu |
| `--gpus <IDS>`               | GPU IDs for multi-GPU training (e.g., `0,1,2,3`)                      |
| `--sync-backend <cpu\|nccl>` | Gradient sync backend for multi-GPU. Default: cpu                     |

### CPU Training

No special flags needed. Just build without CUDA:

```bash
cargo build --release
cargo run --release -- -f models/nano.yaml --target-device cpu
```

### GPU Training

Requires CUDA feature:

```bash
cargo build --release --features cuda
cargo run --release --features cuda -- -f models/nano.yaml
```

### Multi-GPU Training

```bash
# 4 GPUs with CPU sync (simple, works everywhere)
cargo run --release --features cuda -- -f models/nano.yaml --gpus 0,1,2,3 --sync-backend cpu

# 4 GPUs with NCCL sync (faster, requires NCCL)
cargo run --release --features cuda,nccl -- -f models/nano.yaml --gpus 0,1,2,3 --sync-backend nccl
```

## Training Overrides

Override config values from command line:

| Argument           | Description                          |
| ------------------ | ------------------------------------ |
| `--seq-len <N>`    | Override sequence length             |
| `--batch-size <N>` | Override batch size                  |
| `--grad-accum <N>` | Override gradient accumulation steps |
| `--max-steps <N>`  | Override maximum training steps      |

Example:

```bash
cargo run --release -- -f models/nano.yaml --batch-size 4 --seq-len 256 --max-steps 1000
```

## Checkpoint Options

| Argument                | Description                                             |
| ----------------------- | ------------------------------------------------------- |
| `--resume <PATH\|auto>` | Resume from checkpoint. Use `auto` to find latest.      |
| `--max-checkpoints <N>` | Maximum checkpoints to keep. 0 = unlimited. Default: 10 |

Example:

```bash
# Resume from specific checkpoint
cargo run --release -- -f models/nano.yaml --resume checkpoints/step_1000.safetensors

# Resume from latest checkpoint
cargo run --release -- -f models/nano.yaml --resume auto
```

## Output Options

| Argument                   | Description                                                      |
| -------------------------- | ---------------------------------------------------------------- |
| `--headless`               | Output JSON metrics only. No TUI progress bar.                   |
| `--dtype <f32\|f16\|bf16>` | Model precision. Default: f32. BF16 recommended for modern GPUs. |

### Interactive Mode (Default)

Shows a TUI progress bar with live metrics:

- Current loss
- Training speed (iterations/second)
- ETA to completion
- Memory usage

### Headless Mode

Use when:

- TUI doesn't render (CI/CD, piped output, remote shells)
- You want to parse metrics programmatically
- Progress bar appears stuck

```bash
cargo run --release -- -f models/nano.yaml --headless
```

Output format (JSON per line):

```json
{"step": 1, "loss": 11.5161, "grad_norm": 0.2542, "learning_rate": 2.000000e-3, "epoch": 0.01, "it/s": 3.24}
{"step": 21, "loss": 11.4948, "grad_norm": 0.1980, "learning_rate": 2.000000e-3, "epoch": 0.21, "it/s": 3.09}
```

## Performance Options

| Argument         | Description                                                     |
| ---------------- | --------------------------------------------------------------- |
| `--prefetch <N>` | Prefetch N batches in background thread. Default: 0 (disabled). |

Prefetching overlaps CPU data loading with GPU compute:

```bash
cargo run --release --features cuda -- -f models/nano.yaml --prefetch 2
```

## Full Example

```bash
# Full training run with all options
cargo run --release --features cuda -- \
    -f models/nano.yaml \
    -d data/my_dataset.bin \
    --target-device gpu \
    --batch-size 8 \
    --seq-len 512 \
    --max-steps 10000 \
    --prefetch 2 \
    --dtype bf16 \
    --resume auto
```

## Environment Variables

| Variable               | Description                                           |
| ---------------------- | ----------------------------------------------------- |
| `RUST_LOG`             | Set log level (e.g., `RUST_LOG=debug`)                |
| `CUDA_VISIBLE_DEVICES` | Limit visible GPUs (e.g., `CUDA_VISIBLE_DEVICES=0,1`) |
