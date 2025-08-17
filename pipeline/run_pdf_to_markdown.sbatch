#!/bin/bash
###############################################################################
# PDF to Markdown conversion — sharded processing
###############################################################################
#SBATCH --job-name=pdf2md
#SBATCH --account=bddur51
#SBATCH --partition=gpu              # policy: must use GPU partition
#SBATCH --gres=half:1                # half-GPU slice → 32 CPUs available
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/projects/bddur51/artefact-context/pipeline/logs/pdf2md_%j.out
#SBATCH --error=/projects/bddur51/artefact-context/pipeline/logs/pdf2md_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=samjmwaugh@gmail.com

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJ_ROOT=/projects/bddur51/artefact-context/pipeline     # code repo
# IMPORTANT: point this to the directory that already contains shards/shard_XX/PDF_Bucket
RUN_ROOT=/nobackup/projects/bddur51/oa_run_914266

# Create dirs before cd (fixes the previous failure)
mkdir -p "$PROJ_ROOT/logs"

echo "=== Bede PDF to Markdown run context ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "RUN_ROOT=$RUN_ROOT"
echo "PROJ_ROOT=$PROJ_ROOT"
date
hostname

# ── Conda env ─────────────────────────────────────────────────────────────────
source /projects/bddur51/$USER/miniconda/etc/profile.d/conda.sh
conda activate pdf_processing_env  # ← Changed from download_works_env
python -V
which python

# Ensure deps are installed (once per env)
# pip install --upgrade pip
# pip install "pymupdf>=1.24" "Pillow>=10.0.0"

# ── Sharding knobs (read by your Python code) ─────────────────────────────────
export RUN_ROOT="$RUN_ROOT"
export OA_SHARDS=32

# ── Execute PDF to Markdown conversion ────────────────────────────────────────
cd "$PROJ_ROOT"

echo "Starting PDF to Markdown conversion..."
echo "Working from: $PROJ_ROOT"
echo "Shard data in: $RUN_ROOT"
python batch_pdf_to_markdown_sharded.py --max-workers ${SLURM_CPUS_PER_TASK:-16} --output-dir "$RUN_ROOT"

# ── Final summary ─────────────────────────────────────────────────────────────
echo "----------------------------------------------------------------"
echo " PDF to Markdown conversion complete"
echo " Check logs for detailed results"
echo " Output: $RUN_ROOT/Marker_Output/"
echo "----------------------------------------------------------------"
echo "Done."
###############################################################################
