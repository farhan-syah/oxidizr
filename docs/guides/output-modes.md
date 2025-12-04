# Output Modes

Oxidizr supports two output modes for training progress.

## Interactive Mode (Default)

Shows a TUI (Terminal User Interface) progress bar with live metrics:

```
Training ████████████████░░░░░░░░░░░░░░░░ 50% | Step 500/1000 | Loss: 8.234 | 45.2 it/s | ETA: 11s
```

**Features:**

- Real-time loss tracking
- Training speed (iterations/second)
- ETA to completion
- Memory usage indicators
- Visual progress bar

### When Interactive Mode Works

- Interactive terminal sessions
- Direct terminal access (not piped)
- Most local development environments

### When Interactive Mode May Fail

- Non-interactive terminals
- CI/CD environments
- Piped output (`cargo run ... | tee log.txt`)
- Some remote shells (tmux, screen with issues)
- Redirected output (`cargo run ... > output.txt`)

## Headless Mode

Outputs JSON metrics to stdout, one line per log interval:

```bash
cargo run --release -- -f models/nano.yaml --headless
```

Output:

```json
{"step": 1, "loss": 11.5161, "grad_norm": 0.2542, "learning_rate": 2.000000e-3, "epoch": 0.01, "it/s": 3.24}
{"step": 21, "loss": 11.4948, "grad_norm": 0.1980, "learning_rate": 2.000000e-3, "epoch": 0.21, "it/s": 3.09}
{"step": 41, "loss": 11.3234, "grad_norm": 0.1823, "learning_rate": 2.000000e-3, "epoch": 0.41, "it/s": 3.12}
```

### When to Use Headless Mode

1. **TUI doesn't render** - Progress bar appears stuck or garbled
2. **CI/CD pipelines** - Automated training jobs
3. **Logging to file** - Want to capture metrics
4. **Parsing metrics** - Programmatic analysis
5. **Remote execution** - Scripts, cron jobs

### JSON Fields

| Field           | Description                 |
| --------------- | --------------------------- |
| `step`          | Current training step       |
| `loss`          | Training loss               |
| `grad_norm`     | Gradient norm               |
| `learning_rate` | Current learning rate       |
| `epoch`         | Epoch progress (fractional) |
| `it/s`          | Iterations per second       |

## Choosing the Right Mode

| Situation                  | Mode                  |
| -------------------------- | --------------------- |
| Local development          | Interactive (default) |
| Progress bar not appearing | Headless              |
| CI/CD pipeline             | Headless              |
| Logging to file            | Headless              |
| Quick visual check         | Interactive           |
| Long unattended run        | Headless              |

## Parsing Headless Output

### Python

```python
import json

metrics = []
with open("training.log") as f:
    for line in f:
        metrics.append(json.loads(line))

# Plot loss curve
import matplotlib.pyplot as plt
steps = [m['step'] for m in metrics]
losses = [m['loss'] for m in metrics]
plt.plot(steps, losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.savefig('loss_curve.png')
```

### Bash

```bash
# Run training, capture output
cargo run --release -- -f models/nano.yaml --headless > training.log 2>&1

# Monitor in real-time
tail -f training.log
```

## Troubleshooting

### Progress bar stuck?

```bash
# Switch to headless
cargo run --release -- -f models/nano.yaml --headless
```

### No output at all?

Check stderr for errors:

```bash
cargo run --release -- -f models/nano.yaml --headless 2>&1
```

### Want both TUI and log file?

Use `tee` but note TUI may not render properly:

```bash
# Better: use headless for logging
cargo run --release -- -f models/nano.yaml --headless 2>&1 | tee training.log
```

## Log Interval

Control how often metrics are printed:

```yaml
trainer:
  log_interval: 10 # Print every 10 steps
```

Smaller = more output, larger = less output.
