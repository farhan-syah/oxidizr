# Checkpoints

Oxidizr saves training progress to checkpoints, allowing you to resume training or use trained models.

## Checkpoint Format

Checkpoints are saved in SafeTensors format (`.safetensors`):
- Efficient binary format
- Safe to load (no arbitrary code execution)
- Compatible with many ML tools

## Automatic Saving

Configure checkpoint saving in your YAML config:

```yaml
trainer:
  checkpoint_dir: "./checkpoints"
  save_interval: 500  # Save every 500 steps
```

Checkpoints are saved as:
```
checkpoints/
├── step_500.safetensors
├── step_1000.safetensors
├── step_1500.safetensors
└── ...
```

## Checkpoint Management

### Maximum Checkpoints

Control how many checkpoints to keep:

```bash
# Keep only 5 most recent checkpoints
cargo run --release -- -f models/nano.yaml --max-checkpoints 5

# Keep all checkpoints
cargo run --release -- -f models/nano.yaml --max-checkpoints 0
```

Default: 10 checkpoints

### Manual Save Points

The save interval determines when checkpoints are written:

```yaml
trainer:
  save_interval: 1000  # Every 1000 steps
```

Smaller intervals = more recovery points, more disk usage.

## Resuming Training

### From Specific Checkpoint

```bash
cargo run --release -- -f models/nano.yaml --resume checkpoints/step_1000.safetensors
```

### From Latest Checkpoint

```bash
cargo run --release -- -f models/nano.yaml --resume auto
```

The `auto` option finds the most recent checkpoint in the config's `checkpoint_dir`.

## What's Saved

Checkpoints include:
- Model weights
- Optimizer state
- Training step counter
- Random number generator state

This ensures training continues exactly where it left off.

## Resume Behavior

When resuming:
1. Model weights loaded from checkpoint
2. Optimizer state restored
3. Training continues from saved step
4. Metrics resume from checkpoint state

## Best Practices

### Regular Checkpoints

For long training runs:
```yaml
trainer:
  save_interval: 500  # Save frequently
  max_checkpoints: 10  # But don't fill disk
```

### Before Risky Changes

Save a checkpoint before:
- Changing learning rate
- Modifying config
- System maintenance

### Backup Important Checkpoints

Copy important checkpoints outside `checkpoint_dir`:
```bash
cp checkpoints/step_5000.safetensors backups/model_v1.safetensors
```

## Checkpoint Directory Structure

```
my_project/
├── models/
│   └── nano.yaml
├── checkpoints/          # Default location
│   ├── step_500.safetensors
│   ├── step_1000.safetensors
│   └── step_1500.safetensors
└── data/
    └── ...
```

Custom directory:
```yaml
trainer:
  checkpoint_dir: "/mnt/storage/checkpoints"
```

## Troubleshooting

**Checkpoint not found?**
- Check the path is correct
- Verify file exists: `ls checkpoints/`
- Use `--resume auto` to find latest

**Resume produces different results?**
- Ensure same config file
- Check random seed if using one
- Verify data is the same

**Out of disk space?**
- Reduce `save_interval`
- Lower `--max-checkpoints`
- Clean old checkpoints manually

**Checkpoint too large?**
- Normal for large models
- Use lower precision training (`--dtype bf16`)
- Checkpoints scale with model size
