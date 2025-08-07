#!/bin/bash
###############################################################################
#  OpenAlex bulk PDF download – direct to 10 TB project Lustre
###############################################################################
#SBATCH --job-name=oa_download
#SBATCH --account=bddur51
#SBATCH --partition=gpu          # policy: must request a GPU
#SBATCH --gres=half:1            # cheapest token that satisfies the partition
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00        # wall-clock limit (48 h)
#SBATCH --output=/projects/bddur51/artefact-context/pipeline/logs/oa_%j.out
#SBATCH --error=/projects/bddur51/artefact-context/pipeline/logs/oa_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=samjmwaugh@gmail.com
set -euo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJ_ROOT=/projects/bddur51/artefact-context/pipeline            # code repo
LUSTRE_ROOT=/nobackup/projects/bddur51                            # 10 TB space
RUN_DIR=$LUSTRE_ROOT/oa_run_$SLURM_JOB_ID                         # per-job dir
PDF_BUCKET=$RUN_DIR/PDF_Bucket

# create run-specific directory & logs folder
mkdir -p "$PDF_BUCKET" "$PROJ_ROOT/logs"
echo "Running in $RUN_DIR"

# ─── Conda environment ────────────────────────────────────────────────────────
source /projects/bddur51/$USER/miniconda/etc/profile.d/conda.sh
conda activate download_works_env

# tell the python code where to put PDFs
export PDF_BUCKET

# ─── Execute the harvester ────────────────────────────────────────────────────
cd "$RUN_DIR"
python $PROJ_ROOT/batch_download_works.py

echo "✔ All PDFs stored in $PDF_BUCKET"
###############################################################################
