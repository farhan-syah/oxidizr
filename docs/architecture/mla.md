# MLA Guide

Multi-Head Latent Attention (MLA) compresses the key-value cache into a lower-dimensional latent space, reducing memory usage while maintaining quality.

## What is MLA?

Standard attention stores full-dimensional K and V tensors in the cache. MLA projects these into a smaller latent space:

```
Standard: KV cache size = 2 × num_layers × seq_len × hidden_size
MLA:      KV cache size = 2 × num_layers × seq_len × kv_latent_dim
```

With `kv_latent_dim < hidden_size`, you get significant memory savings.

## Configuration

Enable MLA by adding these parameters:

```yaml
model:
  hidden_size: 512
  num_layers: 8
  num_heads: 8
  kv_heads: 4

  # MLA parameters
  kv_latent_dim: 192    # Compressed KV dimension
  q_latent_dim: 192     # Compressed query dimension
  d_rope: 16            # RoPE dimension
```

## How It Works

1. **Compression**: K and V are projected to `kv_latent_dim`
2. **Storage**: Only compressed representations are cached
3. **Decompression**: Restored to full dimension for attention
4. **RoPE**: Applied in a separate `d_rope` dimensional space

This allows long sequences without proportional memory growth.

## Benefits

| Aspect | Standard Attention | MLA |
|--------|-------------------|-----|
| KV cache size | Large | Small |
| Long sequences | Memory limited | Feasible |
| Quality | Baseline | Slightly lower |
| Speed | Baseline | Faster (less memory movement) |

## Example Configuration

### Standalone MLA

```yaml
model:
  hidden_size: 512
  num_layers: 8
  num_heads: 8
  kv_heads: 4
  vocab_size: 128354
  max_seq_len: 2048
  rope_theta: 10000.0
  intermediate_size: 2048

  # MLA
  kv_latent_dim: 192
  q_latent_dim: 192
  d_rope: 16

trainer:
  learning_rate: 0.0003
  batch_size: 2
  max_steps: 5000
```

### MLA + MoE

Combine with Mixture of Experts for capacity:

```yaml
model:
  # ... base config ...

  # MLA
  kv_latent_dim: 192
  q_latent_dim: 192
  d_rope: 16

  # MoE
  num_experts: 4
  experts_per_tok: 2
  shared_expert_enabled: true

trainer:
  load_balance_alpha: 0.01
```

### Hybrid Mamba + MLA

Use MLA only in attention layers:

```yaml
model:
  num_layers: 8
  mamba_layers: [0, 1, 2, 4, 5, 6]  # Mamba layers
  # Layers 3, 7 use MLA

  # MLA params
  kv_latent_dim: 192
  q_latent_dim: 192
  d_rope: 16

  # Mamba2 params
  mamba2_num_heads: 48
  # ... etc
```

## Choosing Latent Dimensions

### Guidelines

- `kv_latent_dim` should be smaller than `hidden_size`
- Typical ratio: 0.25-0.5 of `hidden_size`
- Smaller = more compression, potential quality loss
- Larger = less compression, closer to standard attention

### Examples

| hidden_size | kv_latent_dim | Compression |
|-------------|---------------|-------------|
| 512 | 192 | 2.7x |
| 512 | 256 | 2x |
| 768 | 256 | 3x |
| 1024 | 384 | 2.7x |

## When to Use MLA

**Good for:**
- Long sequence models (>2K tokens)
- Memory-constrained inference
- Batch inference with many requests

**Consider alternatives if:**
- Short sequences only
- Memory not a concern
- Maximum quality required

## Technical Details

MLA decomposes attention:

1. **Latent projection**: `KV_latent = W_kv @ input`
2. **Query latent**: `Q_latent = W_q @ input`
3. **RoPE in subspace**: Position encoding in `d_rope` dimensions
4. **Attention**: Standard attention on latent representations
5. **Output projection**: `output = W_out @ attention_output`

The key insight is that attention patterns can be captured in a lower-dimensional space.
