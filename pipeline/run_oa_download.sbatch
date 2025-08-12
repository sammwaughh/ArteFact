#!/bin/bash
###############################################################################
#  OpenAlex bulk PDF download – sharded to 10 TB project Lustre
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

# PATCH: Remove single PDF_BUCKET - sharding will create its own structure
# PDF_BUCKET=$RUN_DIR/PDF_Bucket

# create run-specific directory & logs folder
mkdir -p "$PROJ_ROOT/logs"
echo "Running in $RUN_DIR"

# ─── Conda environment ────────────────────────────────────────────────────────
source /projects/bddur51/$USER/miniconda/etc/profile.d/conda.sh
conda activate download_works_env

# PATCH: Set sharding environment variables
export RUN_ROOT="$RUN_DIR"
export OA_SHARDS=32

# PATCH: Remove PDF_BUCKET export - sharding handles this
# export PDF_BUCKET

# ─── Execute the harvester ────────────────────────────────────────────────────
cd "$RUN_DIR"
python $PROJ_ROOT/batch_download_works.py

# PATCH: After downloads complete, merge the sharded data
echo " Download complete. Merging sharded metadata..."
python $PROJ_ROOT/merge_works_and_artists.py \
    --run-root "$RUN_DIR" \
    --artist-json-dir "$PROJ_ROOT/Artist-JSONs"

# PATCH: Show final results
echo " Final output files:"
echo "  - works.json: $(wc -l < "$RUN_DIR/works.json" || echo "0") entries"
echo "  - artists.json: $(wc -l < "$RUN_DIR/artists.json" || echo "0") entries"
echo "  - Sharded PDFs: $(find "$RUN_DIR/shards" -name "*.pdf" | wc -l) files"

echo "✔ All PDFs stored in sharded directories under $RUN_DIR/shards/"
echo "✔ Consolidated metadata in $RUN_DIR/works.json and $RUN_DIR/artists.json"
###############################################################################
