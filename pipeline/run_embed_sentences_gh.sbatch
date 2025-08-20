#!/bin/bash
#SBATCH --job-name=embed-sentences-gh
#SBATCH --account=bddur51
#SBATCH --partition=gh              # Grace-Hopper partition
#SBATCH --gres=gpu:1                # Full H100 node
#SBATCH --cpus-per-task=72          # Use all 72 ARM64 cores
#SBATCH --mem=96G                   # Increased to match GPU memory
#SBATCH --time=4-00:00:00           # Increased time for large dataset
#SBATCH --output=/projects/bddur51/artefact-context/pipeline/logs/embed_sentences_gh_%j.out
#SBATCH --error=/projects/bddur51/artefact-context/pipeline/logs/embed_sentences_gh_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJ_ROOT=/projects/bddur51/artefact-context
RUN_ROOT=/nobackup/projects/bddur51/oa_run_914266

echo "=== Grace-Hopper Embedding Generation ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "RUN_ROOT=$RUN_ROOT"
echo "Architecture: $(uname -m)"
echo "CPU cores: $SLURM_CPUS_PER_TASK"
echo "GPU memory: $SLURM_MEM_PER_NODE"
date
hostname

# ── Grace-Hopper Environment Setup ────────────────────────────────────────────
# Load CUDA module for optimal performance
module load cuda/12.6.2

# Use existing environment instead of creating new one
source ~/embedding_env_gh/bin/activate

# ── Performance Tuning ────────────────────────────────────────────────────────
# Set environment variables for optimal performance
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_V8_API_DISABLED=0

# ── Run Embedding Generation ──────────────────────────────────────────────────
cd "$PROJ_ROOT"
export RUN_ROOT="$RUN_ROOT"

# Verify GPU access and performance
echo "=== GPU Verification ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB' if torch.cuda.is_available() else 'No GPU')
print(f'CUDA version: {torch.version.cuda}')
"

# Run the script with performance monitoring
echo "=== Starting Embedding Generation ==="
time python pipeline/efficient_batch_embed_sentences.py

echo "=== Embedding generation complete ==="
echo "Job completed at: $(date)"
