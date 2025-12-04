# Mamba Guide

Mamba is a state space model architecture that provides efficient sequence processing without the quadratic complexity of attention.

## What is Mamba?

Traditional transformers use attention with O(N²) complexity. Mamba uses selective state spaces with O(N) complexity, making it much faster for long sequences.

Key benefits:

- Linear time complexity
- No KV cache needed
- Efficient long-range dependencies
- Fast training and inference

## Mamba Variants

Oxidizr supports three Mamba variants:

### Mamba1

The original selective state space model:

```yaml
model:
  # ... base config ...
  mamba_d_state: 16 # State dimension
  mamba_d_conv: 4 # Conv kernel size
  mamba_expand: 2 # Expansion factor
```

### Mamba2

Improved version with State Space Duality (SSD):

```yaml
model:
  # ... base config ...
  mamba2_num_heads: 48
  mamba2_head_dim: 16
  mamba2_state_size: 64
  mamba2_chunk_size: 64
  mamba2_n_groups: 1
  mamba2_conv_kernel: 4
  mamba2_expand: 2
```

**Important constraint:**

```
hidden_size × mamba2_expand == mamba2_num_heads × mamba2_head_dim
```

Example: `384 × 2 = 768 == 48 × 16` ✓

### Mamba3

Latest variant with three innovations:

1. **Trapezoidal Discretization** - More expressive recurrence (replaces short conv)
2. **Complex-Valued SSM via RoPE** - Data-dependent rotary embeddings for state tracking
3. **MIMO (Multi-Input Multi-Output)** - Increased arithmetic intensity for inference

```yaml
model:
  # Enable Mamba3 features
  mamba3_enabled: true
  mamba3_complex_rope: true # Complex-valued RoPE for state tracking
  mamba3_mimo_rank: 0 # 0 = SISO, 4 = MIMO mode
  mamba3_use_conv: false # false = trapezoidal discretization

  # Mamba2 base parameters (shared with Mamba3)
  mamba2_num_heads: 48
  mamba2_head_dim: 16
  mamba2_state_size: 64
  mamba2_chunk_size: 64
  mamba2_n_groups: 1
  mamba2_conv_kernel: 4 # Only used if mamba3_use_conv: true
  mamba2_expand: 2
```

**MIMO mode** (`mamba3_mimo_rank: 4`) increases GPU utilization during inference by processing multiple inputs/outputs simultaneously.

## Pure Mamba Model

Use Mamba for all layers:

```yaml
model:
  hidden_size: 384
  num_layers: 8
  vocab_size: 128354
  max_seq_len: 1024

  # Mamba2 config
  mamba2_num_heads: 48
  mamba2_head_dim: 16
  mamba2_state_size: 64
  mamba2_chunk_size: 64
  mamba2_n_groups: 1
  mamba2_conv_kernel: 4
  mamba2_expand: 2

  # All layers use Mamba
  mamba_layers: [0, 1, 2, 3, 4, 5, 6, 7]

trainer:
  learning_rate: 0.002
  batch_size: 2
  max_steps: 5000
```

## Hybrid Mamba + Attention

Mix Mamba with attention layers for best of both:

```yaml
model:
  num_layers: 8
  # Layers 0,1,2,4,5,6 use Mamba
  # Layers 3,7 use attention
  mamba_layers: [0, 1, 2, 4, 5, 6]
```

This pattern:

- Mamba handles sequential processing efficiently
- Attention layers provide global context
- Better quality than pure Mamba
- Faster than pure attention

## When to Use Mamba

| Scenario                    | Recommendation           |
| --------------------------- | ------------------------ |
| Long sequences (>2K tokens) | Mamba or hybrid          |
| Memory constrained          | Mamba (no KV cache)      |
| Maximum quality             | Hybrid or pure attention |
| Fast training               | Mamba2                   |
| Simple setup                | Start with attention     |

## Sample Configs

See `models/` directory:

- `nano_mamba2.yaml` - Hybrid Mamba2 + MLA
- `nano_mamba3.yaml` - Pure Mamba3
- `nano_mamba3_hybrid.yaml` - Hybrid Mamba3 + MLA + MoE
- `nano_mamba3_mimo.yaml` - Pure Mamba3 with MIMO

## Technical Details

### State Space Model

Mamba processes sequences through a state space:

```
h(t) = Ah(t-1) + Bx(t)
y(t) = Ch(t) + Dx(t)
```

Where:

- `h` is the hidden state
- `x` is input, `y` is output
- `A, B, C, D` are learned parameters

### Selective Mechanism

Mamba's key innovation is making `A, B, C` input-dependent (selective), allowing the model to focus on relevant information.

### SSD Algorithm (Mamba2)

Mamba2's State Space Duality reformulates the computation to enable:

- Better parallelization during training
- Chunk-based processing
- Faster overall throughput

## Implementation Notes

### Current Status

Our Mamba2/Mamba3 implementation follows the [HuggingFace Transformers](https://huggingface.co/docs/transformers/model_doc/mamba2) reference architecture. We implement the **naive SSD algorithm** using standard tensor operations (similar to HF's `torch_forward` path).

**What this means:**

- ✅ **Correct**: Training loss decreases, inference works properly
- ✅ **Complete**: Full Mamba2/Mamba3 feature set implemented
- ⚠️ **Not optimized**: No custom CUDA kernels (yet)

### Why No Custom Kernels?

The official Mamba implementation uses specialized Triton/CUDA kernels (`mamba_chunk_scan_combined`, `causal_conv1d_fn`) for maximum performance. These kernels are:

- Written in Triton (Python-based GPU compiler)
- Tightly coupled to PyTorch's autograd
- Not available in Candle

Our approach uses **batched matmul** instead of these kernels. Matmul is heavily optimized (cuBLAS), so performance is reasonable but not optimal.

### Performance Comparison

| Implementation            | Speed                  | Notes                   |
| ------------------------- | ---------------------- | ----------------------- |
| HF `cuda_kernels_forward` | Fastest                | Requires Triton kernels |
| HF `torch_forward`        | Fast                   | Naive PyTorch ops       |
| **Oxidizr (current)**     | Comparable to HF torch | Batched matmul approach |

Interestingly, HF docs note: _"torch_forward without compilation is 3-4x faster than cuda_kernels_forward"_ during prefill due to kernel launch overhead. So "naive" isn't always slower.

### Future Optimization

We have two paths forward:

**Option 1: Wait for Candle**

- Candle may add optimized SSM kernels in future releases
- Drop-in replacement when available

**Option 2: Native Rust Kernels**

- Write custom CUDA kernels in Rust using `cudarc`
- Full control over optimization
- Rust's safety guarantees even in GPU code

The current implementation is production-ready for training and inference. Kernel optimization is a performance enhancement, not a correctness fix.
