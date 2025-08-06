#!/bin/bash
#SBATCH --job-name=oa_download
#SBATCH --account=bddur51
#SBATCH --partition=gpu          # policy: must request a GPU
#SBATCH --cpus-per-task=20
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=/projects/bddur51/artefact-context/pipeline/logs/oa_%j.out
#SBATCH --error=/projects/bddur51/artefact-context/pipeline/logs/oa_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=samjmwaugh@gmail.com
set -euo pipefail

PROJ_ROOT=/projects/bddur51/artefact-context/pipeline
SCRATCH_JOB=/scratch/$USER/oa_$SLURM_JOB_ID
SCRATCH_BUCKET=$SCRATCH_JOB/PDF_Bucket
PROJ_BUCKET=$PROJ_ROOT/PDF_Bucket

mkdir -p "$SCRATCH_BUCKET" "$PROJ_ROOT/logs"

source /projects/bddur51/$USER/miniconda/etc/profile.d/conda.sh
conda activate download_works_env

export PDF_BUCKET="$SCRATCH_BUCKET"      # ← picked up by download_single_work.py

cd "$SCRATCH_JOB"
python $PROJ_ROOT/batch_download_works.py

rsync -av --delete "$SCRATCH_BUCKET"/ "$PROJ_BUCKET"/     # ~15 s for 10 GB
echo "✔ PDFs copied to $PROJ_BUCKET"
