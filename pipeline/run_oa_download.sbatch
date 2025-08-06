#!/bin/bash
#SBATCH --job-name=oa_download
#SBATCH --account=bddur51
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=/projects/bddur51/artefact-context/pipeline/logs/oa_%j.out
#SBATCH --error=/projects/bddur51/artefact-context/pipeline/logs/oa_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=samjmwaugh@gmail.com
set -euo pipefail

# ─── Paths ──────────────────────────────────────────────────────────────
PROJ_ROOT=/projects/bddur51/artefact-context/pipeline
SCRATCH_JOB=/scratch/$USER/oa_$SLURM_JOB_ID          # whole sandbox
SCRATCH_BUCKET=$SCRATCH_JOB/PDF_Bucket               # PDFs land here
PROJ_BUCKET=$PROJ_ROOT/PDF_Bucket

mkdir -p "$SCRATCH_BUCKET" "$PROJ_ROOT/logs"

# ─── Activate Conda ─────────────────────────────────────────────────────
source /projects/bddur51/$USER/miniconda/etc/profile.d/conda.sh
conda activate download_works_env

# ─── Tell the downloader where to write PDFs ────────────────────────────
export PDF_BUCKET="$SCRATCH_BUCKET"      # see code patch below

# ─── Run ────────────────────────────────────────────────────────────────
cd "$SCRATCH_JOB"
python $PROJ_ROOT/batch_download_works.py

# ─── Copy results back (fast) ───────────────────────────────────────────
rsync -av --delete "$SCRATCH_BUCKET"/  "$PROJ_BUCKET"/

echo "✔ PDFs copied to $PROJ_BUCKET"
