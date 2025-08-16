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
LUSTRE_ROOT=/nobackup/projects/bddur51                    # 10 TB Lustre
RUN_DIR=$LUSTRE_ROOT/oa_run_${SLURM_JOB_ID}               # per-job run folder

# Create dirs before cd (fixes the previous failure)
mkdir -p "$PROJ_ROOT/logs"

echo "=== Bede PDF to Markdown run context ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "RUN_DIR=$RUN_DIR"
echo "PROJ_ROOT=$PROJ_ROOT"
date
hostname

# ── Conda env ─────────────────────────────────────────────────────────────────
source /projects/bddur51/$USER/miniconda/etc/profile.d/conda.sh
conda activate pdf_processing_env  # ← Changed from download_works_env
python -V
which python

# Verify marker_single is available
echo "Checking marker_single availability..."
which marker_single || echo "WARNING: marker_single not found in PATH"
marker_single --version || echo "WARNING: marker_single version check failed"

# ── Sharding knobs (read by your Python code) ─────────────────────────────────
export RUN_ROOT="$RUN_DIR"
export OA_SHARDS=32

# ── Execute PDF to Markdown conversion ────────────────────────────────────────
cd "$PROJ_ROOT"

echo "Starting PDF to Markdown conversion..."
echo "Working from: $PROJ_ROOT"
echo "Shard data in: $RUN_DIR"
python batch_pdf_to_markdown_sharded.py --max-workers 16 --output-dir "$RUN_DIR"

# ── Final summary ─────────────────────────────────────────────────────────────
echo "----------------------------------------------------------------"
echo " PDF to Markdown conversion complete"
echo " Check logs for detailed results"
echo " Output: $RUN_DIR/Marker_Output/"
echo " Logs: /projects/bddur51/artefact-context/pipeline/logs/pdf2md_${SLURM_JOB_ID}.{out,err}"
echo "----------------------------------------------------------------"
echo "Done."
###############################################################################
