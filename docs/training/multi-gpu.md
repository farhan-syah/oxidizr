# Multi-GPU Training

Oxidizr supports data-parallel training across multiple GPUs.

## How It Works

1. Dataset is sharded across GPUs
2. Each GPU runs forward/backward on its shard
3. Gradients are synchronized (all-reduced)
4. Averaged gradients applied by optimizer

**Effective batch size** = `batch_size × gradient_accumulation × num_gpus`

## Basic Usage

```bash
# Train on GPUs 0 and 1
cargo run --release --features cuda -- -f models/nano.yaml --gpus 0,1

# Train on 4 GPUs
cargo run --release --features cuda -- -f models/nano.yaml --gpus 0,1,2,3
```

## Sync Backends

### CPU Backend (Default)

Simple, works everywhere:

```bash
cargo run --release --features cuda -- -f models/nano.yaml --gpus 0,1 --sync-backend cpu
```

- Gradients copied to CPU for averaging
- Works with any number of GPUs
- Good for 2-4 GPUs

### NCCL Backend (Faster)

Faster for 4+ GPUs, requires NCCL:

```bash
# Build with NCCL support
cargo build --release --features cuda,nccl

# Run with NCCL
cargo run --release --features cuda,nccl -- -f models/nano.yaml --gpus 0,1,2,3 --sync-backend nccl
```

- Direct GPU-to-GPU communication
- Better scaling for many GPUs
- Requires NCCL library installed

## Effective Batch Size

With multi-GPU, your effective batch size increases:

```
effective_batch = batch_size × gradient_accumulation × num_gpus
```

Example:
- `batch_size=2`
- `gradient_accumulation=4`
- `num_gpus=2`
- **Effective batch = 16**

Adjust learning rate accordingly (often scale with batch size).

## Example Configurations

### 2 GPUs, Simple Setup

```bash
cargo run --release --features cuda -- \
    -f models/nano.yaml \
    --gpus 0,1 \
    --sync-backend cpu
```

### 4 GPUs, NCCL

```bash
cargo run --release --features cuda,nccl -- \
    -f models/nano.yaml \
    --gpus 0,1,2,3 \
    --sync-backend nccl \
    --dtype bf16
```

### 8 GPUs, Full Setup

```bash
cargo run --release --features cuda,nccl -- \
    -f models/large.yaml \
    --gpus 0,1,2,3,4,5,6,7 \
    --sync-backend nccl \
    --dtype bf16 \
    --prefetch 4
```

## Selecting Specific GPUs

```bash
# Use GPUs 2 and 3 only
cargo run --release --features cuda -- -f models/nano.yaml --gpus 2,3

# Or use environment variable
CUDA_VISIBLE_DEVICES=2,3 cargo run --release --features cuda -- -f models/nano.yaml --gpus 0,1
```

## Performance Tips

1. **Use NCCL for 4+ GPUs** - Much faster gradient sync
2. **Match batch sizes** - Keep consistent across GPUs
3. **Enable prefetching** - `--prefetch 2` overlaps I/O
4. **Use BF16** - Faster and less memory
5. **Balance load** - Use same GPU models if possible

## Troubleshooting

**GPUs not detected?**
```bash
# Check visible GPUs
nvidia-smi

# List GPU IDs
nvidia-smi -L
```

**NCCL errors?**
- Ensure NCCL is installed
- Build with `--features cuda,nccl`
- Check all GPUs are same model

**Uneven GPU usage?**
- Data sharding may not be perfectly even
- This is normal for small datasets

**Out of memory on one GPU?**
- Reduce batch size
- Ensure no other processes using GPU memory
