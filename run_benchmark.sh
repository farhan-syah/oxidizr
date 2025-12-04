#!/bin/bash

# Benchmark runner script for oxidizr
# Runs systematic benchmarks across different batch sizes and sequence lengths
#
# Usage: ./run_benchmark.sh [-f config.yaml]
#
# Options:
#   -f config.yaml    Use specific config file (default: uses binary default)
#
# Examples:
#   ./run_benchmark.sh                              # Benchmark with default config
#   ./run_benchmark.sh -f models/nano.yaml          # Benchmark Mamba1 hybrid
#   ./run_benchmark.sh -f models/nano_mamba2.yaml   # Benchmark Mamba2 hybrid
#
# Output:
#   benchmark/<timestamp>/logs.txt      Full output logs
#   benchmark/<timestamp>/results.csv   CSV with metrics per configuration

BINARY="./target/release/oxidizr"
CONFIG_FILE=""

# Parse command line arguments
while getopts "f:" opt; do
    case $opt in
        f)
            CONFIG_FILE="$OPTARG"
            ;;
        \?)
            echo "Usage: $0 [-f config.yaml]"
            exit 1
            ;;
    esac
done

# Create timestamped benchmark directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BENCH_DIR="benchmark/${TIMESTAMP}"
mkdir -p "$BENCH_DIR"

LOGS_FILE="${BENCH_DIR}/logs.txt"
RESULTS_CSV="${BENCH_DIR}/results.csv"

# Test configurations from CSV
# Format: seq_length,batch_size,gradient_accumulation
CONFIGS=(
    # seq=128
    "128,1,1"
    "128,1,2"
    "128,2,1"
    "128,2,2"
    "128,3,1"
    "128,3,2"
    "128,4,1"
    "128,4,2"
    "128,5,1"
    "128,5,2"
    # seq=256
    "256,1,1"
    "256,1,2"
    "256,2,1"
    "256,2,2"
    "256,3,1"
    "256,3,2"
    "256,4,1"
    "256,4,2"
    "256,5,1"
    "256,5,2"
    # seq=384 (nano-mini default)
    "384,1,1"
    "384,1,2"
    "384,2,1"
    "384,2,2"
    "384,3,1"
    "384,3,2"
    "384,4,1"
    "384,4,2"
    "384,5,1"
    "384,5,2"
    # seq=512
    "512,1,1"
    "512,1,2"
    "512,2,1"
    "512,2,2"
    "512,3,1"
    "512,3,2"
    "512,4,1"
    "512,4,2"
    "512,5,1"
    "512,5,2"
    # seq=1024
    "1024,1,1"
    "1024,1,2"
    "1024,2,1"
    "1024,2,2"
    "1024,3,1"
    "1024,3,2"
    "1024,4,1"
    "1024,4,2"
    "1024,5,1"
    "1024,5,2"
)

# Initialize CSV with header
echo "seq_length,batch_size,gradient_accumulation,effective_batch,vram_usage_mb,samples_per_s,notes" > "$RESULTS_CSV"

echo "Starting benchmark run..."
echo "Output directory: $BENCH_DIR"
echo "Logs: $LOGS_FILE"
echo "Results: $RESULTS_CSV"
echo ""

# Build config args
CONFIG_ARGS=""
if [ -n "$CONFIG_FILE" ]; then
    CONFIG_ARGS="-f $CONFIG_FILE"
fi

# Log header to logs file
echo "=== Oxidizr Benchmark Run ===" | tee -a "$LOGS_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$LOGS_FILE"
echo "Binary: $BINARY" | tee -a "$LOGS_FILE"
if [ -n "$CONFIG_FILE" ]; then
    echo "Config: $CONFIG_FILE" | tee -a "$LOGS_FILE"
fi
echo "======================================" | tee -a "$LOGS_FILE"
echo "" | tee -a "$LOGS_FILE"

for config in "${CONFIGS[@]}"; do
    IFS=',' read -r seq batch grad <<< "$config"

    effective_batch=$((batch * grad))

    echo "Running: seq=$seq, batch=$batch, grad_accum=$grad" | tee -a "$LOGS_FILE"

    # Run in headless mode with max 1 step (just to get initial metrics)
    # Use timeout to prevent hanging, and handle errors gracefully
    set +e  # Don't exit on error
    output=$($BINARY $CONFIG_ARGS --headless --seq-len "$seq" --batch-size "$batch" --grad-accum "$grad" --max-steps 1 2>&1)
    exit_code=$?
    set -e  # Re-enable exit on error

    # Write full output to logs
    echo "--- Full Output ---" >> "$LOGS_FILE"
    echo "$output" >> "$LOGS_FILE"
    echo "--- Exit Code: $exit_code ---" >> "$LOGS_FILE"
    echo "" >> "$LOGS_FILE"

    # Check for various error conditions
    if [ $exit_code -ne 0 ]; then
        # Check specific error types
        if echo "$output" | grep -qi "out of memory\|OUT_OF_MEMORY"; then
            echo "  Result: OOM" | tee -a "$LOGS_FILE"
            echo "$seq,$batch,$grad,$effective_batch,,,OOM" >> "$RESULTS_CSV"
        elif echo "$output" | grep -qi "error\|panic\|failed"; then
            echo "  Result: ERROR" | tee -a "$LOGS_FILE"
            error_line=$(echo "$output" | grep -i "error" | head -1)
            if [ -n "$error_line" ]; then
                echo "  Details: $error_line" | tee -a "$LOGS_FILE"
            fi
            echo "$seq,$batch,$grad,$effective_batch,,,ERROR" >> "$RESULTS_CSV"
        else
            echo "  Result: UNKNOWN_ERROR (exit code: $exit_code)" | tee -a "$LOGS_FILE"
            echo "$seq,$batch,$grad,$effective_batch,,,UNKNOWN_ERROR" >> "$RESULTS_CSV"
        fi
    else
        # Success - extract metrics
        last_line=$(echo "$output" | grep "{'step':" | tail -1)

        if [ -n "$last_line" ]; then
            # Parse JSON-like output
            step=$(echo "$last_line" | sed -n "s/.*'step': \([0-9]*\).*/\1/p")
            loss=$(echo "$last_line" | sed -n "s/.*'loss': \([0-9.]*\).*/\1/p")
            it_s=$(echo "$last_line" | sed -n "s/.*'it\/s': \([0-9.]*\).*/\1/p")
            vram=$(echo "$last_line" | sed -n "s/.*'vram': '\([0-9]*\)'.*/\1/p")

            echo "  Result: SUCCESS" | tee -a "$LOGS_FILE"
            echo "  Metrics: it/s=$it_s, vram=${vram}MB, loss=$loss" | tee -a "$LOGS_FILE"

            # Write to CSV: seq_length,batch_size,gradient_accumulation,effective_batch,vram_usage_mb,samples_per_s,notes
            echo "$seq,$batch,$grad,$effective_batch,$vram,$it_s," >> "$RESULTS_CSV"
        else
            echo "  Result: NO_OUTPUT" | tee -a "$LOGS_FILE"
            echo "$seq,$batch,$grad,$effective_batch,,,NO_OUTPUT" >> "$RESULTS_CSV"
        fi
    fi

    echo "" | tee -a "$LOGS_FILE"

    # Small delay between runs to let GPU cool down
    sleep 1
done

echo "Benchmark complete!" | tee -a "$LOGS_FILE"
echo ""
echo "Results saved to:"
echo "  - Full logs: $LOGS_FILE"
echo "  - CSV results: $RESULTS_CSV"
