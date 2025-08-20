#!/bin/bash
#SBATCH --job-name=embed-sentences-gh
#SBATCH --account=bddur51
#SBATCH --partition=gh              # Grace-Hopper partition
#SBATCH --gres=gpu:1                # Full H100 node
#SBATCH --cpus-per-task=72          # Use all 72 ARM64 cores
#SBATCH --mem=96G                   # Increased to match GPU memory
#SBATCH --time=8-00:00:00           # 8 days for safety
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

# ── Environment Setup ─────────────────────────────────────────────────────────
module load cuda/12.6.2
source ~/embedding_env_gh/bin/activate

# ── Performance Tuning ────────────────────────────────────────────────────────
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_V8_API_DISABLED=0

# ── Pre-Run Setup ────────────────────────────────────────────────────────────
cd "$PROJ_ROOT"
export RUN_ROOT="$RUN_ROOT"

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
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'CUDA version: {torch.version.cuda}')
"

# ── Run Embedding Generation ──────────────────────────────────────────────────
echo "=== Starting Embedding Generation ==="
echo "Start time: $(date)"
echo "Running script from: $(pwd)"
echo "Script path: $(pwd)/pipeline/efficient_batch_embed_sentences.py"

time python pipeline/efficient_batch_embed_sentences.py
echo "End time: $(date)"

echo "=== Embedding generation complete ==="
echo "Job completed at: $(date)"
