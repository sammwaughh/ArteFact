#!/bin/bash
#SBATCH --job-name=embed-sentences-gh
#SBATCH --account=bddur51
#SBATCH --partition=gh              # Grace-Hopper partition
#SBATCH --gres=gpu:1                # Full H100 node
#SBATCH --nodes=1                   # Single node
#SBATCH --ntasks=1                  # Single task
#SBATCH --mem=240G                  # Use 240GB (half of 480GB to be safe)
#SBATCH --time=2-00:00:00
#SBATCH --output=/projects/bddur51/artefact-context/pipeline/logs/embed_sentences_gh_%j.out
#SBATCH --error=/projects/bddur51/artefact-context/pipeline/logs/embed_sentences_gh_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJ_ROOT=/projects/bddur51/artefact-context
RUN_ROOT=/nobackup/projects/bddur51/oa_run_914266

echo "=== Grace-Hopper Embedding Generation (OPTIMIZED) ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "RUN_ROOT=$RUN_ROOT"
echo "Architecture: $(uname -m)"
echo "CPU cores: ${SLURM_CPUS_PER_TASK:-'Not set'}"
echo "GPU memory: ${SLURM_MEM_PER_NODE:-'Not set'}"
echo "Job time limit: ${SLURM_TIMELIMIT:-'Not set'}"
date
hostname

# ── Environment Setup ─────────────────────────────────────────────────────────
if ! module load cuda/12.6.2; then
    echo "❌ ERROR: Failed to load CUDA module"
    exit 1
fi

if ! source ~/embedding_env_gh/bin/activate; then
    echo "❌ ERROR: Failed to activate Python environment"
    exit 1
fi

# ── Performance Tuning ────────────────────────────────────────────────────────
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_V8_API_DISABLED=0

# ADD THESE: Grace-Hopper specific optimizations
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export NCCL_IB_DISABLE=1  # Disable InfiniBand for single-node jobs
export NCCL_P2P_DISABLE=1 # Disable P2P for single-node jobs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}

# ADD THIS: Use local scratch space for temporary files
if [ -n "${SLURM_TMPDIR:-}" ]; then
    export TMPDIR="$SLURM_TMPDIR"
    export TEMP="$SLURM_TMPDIR"
    echo "Using local scratch space: $SLURM_TMPDIR"
else
    echo "Warning: SLURM_TMPDIR not set, using default temp location"
fi

# ── Pre-Run Setup ────────────────────────────────────────────────────────────
cd "$PROJ_ROOT"
export RUN_ROOT="$RUN_ROOT"
export SLURM_SUBMIT_DIR="$PROJ_ROOT"  # ADD THIS: Explicit working directory

# Create logs directory
mkdir -p pipeline/logs

# ── Pre-Run Verification ─────────────────────────────────────────────────────
echo "=== Pre-Run Verification ==="
echo "Current directory: $(pwd)"

# Check data files exist
echo "Checking data files..."
if [[ ! -f "$RUN_ROOT/sentences.json" ]]; then
    echo "❌ ERROR: sentences.json not found at $RUN_ROOT/sentences.json"
    exit 1
fi
echo "✅ sentences.json found"

if [[ ! -d "$RUN_ROOT/PaintingCLIP" ]]; then
    echo "⚠️  WARNING: PaintingCLIP directory not found - will skip PaintingCLIP embeddings"
else
    echo "✅ PaintingCLIP directory found"
fi

# Check script exists
if [[ ! -f "pipeline/efficient_batch_embed_sentences.py" ]]; then
    echo "❌ ERROR: Script not found at pipeline/efficient_batch_embed_sentences.py"
    exit 1
fi
echo "✅ Script found"

# ── GPU Verification ──────────────────────────────────────────────────────────
echo "=== GPU Verification ==="
python -c "
import torch
import os
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'CUDA version: {torch.version.cuda}')
print(f'CPU cores available: {os.cpu_count()}')
"

# ── Run Embedding Generation ──────────────────────────────────────────────────
echo "=== Starting OPTIMIZED Embedding Generation ==="
echo "Start time: $(date)"
echo "Running script from: $(pwd)"
echo "Script path: $(pwd)/pipeline/efficient_batch_embed_sentences.py"
echo "CPU cores: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"

time python pipeline/efficient_batch_embed_sentences.py
echo "End time: $(date)"

echo "=== Embedding generation complete ==="
echo "Job completed at: $(date)"
