# CPU Training

Oxidizr supports full training on CPU without any GPU. This is great for learning, prototyping, and systems without GPU access.

## Building for CPU

```bash
# Standard build (no special flags needed)
cargo build --release
```

## Running on CPU

```bash
# Explicit CPU target
cargo run --release -- -f models/nano.yaml --target-device cpu

# Or just run without CUDA feature (defaults to CPU)
cargo run --release -- -f models/nano.yaml
```

## Performance Expectations

CPU training is slower than GPU but fully functional:

| Model | CPU Speed | GPU Speed |
|-------|-----------|-----------|
| nano (~83M params) | ~3 it/s | ~50+ it/s |

Actual speed depends on:
- CPU cores and speed
- Batch size and sequence length
- Model architecture

## Optimizing CPU Training

### Reduce Memory Pressure

```bash
# Smaller batch size
cargo run --release -- -f models/nano.yaml --batch-size 1 --target-device cpu

# Shorter sequences
cargo run --release -- -f models/nano.yaml --seq-len 128 --target-device cpu
```

### Use Gradient Accumulation

Simulate larger batches without more memory:

```yaml
trainer:
  batch_size: 1
  gradient_accumulation: 4  # Effective batch = 4
```

### Enable Prefetching

Overlap data loading with computation:

```bash
cargo run --release -- -f models/nano.yaml --prefetch 2 --target-device cpu
```

## When to Use CPU

- **Learning** - Understand the training process
- **Prototyping** - Test config changes quickly
- **No GPU available** - Cloud instances, laptops
- **Debugging** - Easier to debug without CUDA
- **Small models** - Nano-scale experiments

## Comparing CPU vs GPU

```bash
# Test on CPU
cargo run --release -- -f models/nano.yaml --target-device cpu --max-steps 100 --headless

# Test on GPU (requires CUDA build)
cargo build --release --features cuda
cargo run --release --features cuda -- -f models/nano.yaml --target-device gpu --max-steps 100 --headless
```

## Troubleshooting

**Training seems slow?**
- This is expected. CPU is 10-20x slower than GPU.
- Use `--headless` to see actual iteration speed.

**Out of memory?**
- Reduce batch size: `--batch-size 1`
- Reduce sequence length: `--seq-len 128`

**Want to use GPU?**
- See [GPU Training](gpu.md)
