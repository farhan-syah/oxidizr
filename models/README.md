# Model Configurations

Example YAML configs for different architectures.

## Available Configs

| Config | Architecture | Description |
|--------|--------------|-------------|
| `nano-start.yaml` | Transformer | Small GPT for learning (cl100k_base vocab) |
| `nano.yaml` | Hybrid Mamba2 + MLA + MoE | Production-ready hybrid |
| `nano_mamba2.yaml` | Hybrid Mamba2 + MLA | Mamba2 with attention layers |
| `nano_mamba3.yaml` | Pure Mamba3 | All Mamba3 layers |
| `nano_mamba3_hybrid.yaml` | Hybrid Mamba3 + MLA + MoE | Mamba3 with attention layers |
| `nano_mamba3_mimo.yaml` | Pure Mamba3 MIMO | Mamba3 with multi-input/output |

## Quick Start

```bash
# Educational (recommended for beginners)
cargo run --release -- -f models/nano-start.yaml -d data/nano-start/tokenized/combined.bin

# Production hybrid
cargo run --release -- -f models/nano.yaml -d path/to/data.bin

# Pure Mamba3
cargo run --release -- -f models/nano_mamba3.yaml -d path/to/data.bin
```

## Architecture Overview

### Transformer (nano-start.yaml)

Standard GPT-style transformer for learning:

```yaml
model:
  hidden_size: 256
  num_layers: 6
  num_heads: 8
  vocab_size: 100315  # cl100k_base
```

### Mamba2 Hybrid (nano.yaml, nano_mamba2.yaml)

Combines Mamba2 SSM with attention:

```yaml
model:
  mamba_layers: [0, 1, 2, 4, 5, 6]  # Mamba2 layers
  # Layers 3, 7 use MLA + MoE

  mamba2_num_heads: 48
  mamba2_head_dim: 16
  mamba2_state_size: 64
  mamba2_expand: 2
```

### Mamba3 (nano_mamba3*.yaml)

Latest Mamba with three innovations:

```yaml
model:
  mamba3_enabled: true
  mamba3_complex_rope: true   # Complex-valued RoPE
  mamba3_mimo_rank: 0         # 0=SISO, 4=MIMO
  mamba3_use_conv: false      # Trapezoidal discretization

  # Uses Mamba2 base parameters
  mamba2_num_heads: 48
  mamba2_head_dim: 16
  mamba2_state_size: 64
  mamba2_expand: 2
```

## Creating Custom Configs

1. Copy an existing config as template
2. Adjust parameters
3. Ensure constraint: `hidden_size × expand == num_heads × head_dim`

```bash
cargo run --release -- -f path/to/your/config.yaml -d path/to/data.bin
```

## Tips

- Start with `nano-start.yaml` to learn
- Use `--batch-size 1` to test memory usage
- Use `--headless` if progress bar doesn't appear
