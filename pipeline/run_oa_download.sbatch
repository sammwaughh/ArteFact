#!/bin/bash
###############################################################################
# OpenAlex bulk PDF download — sharded to 10 TB project Lustre
###############################################################################
#SBATCH --job-name=oa_download
#SBATCH --account=bddur51
#SBATCH --partition=gpu              # policy: must use GPU partition
#SBATCH --gres=half:1                # half-GPU slice → 32 CPUs available
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=/projects/bddur51/artefact-context/pipeline/logs/oa_%j.out
#SBATCH --error=/projects/bddur51/artefact-context/pipeline/logs/oa_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=samjmwaugh@gmail.com

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJ_ROOT=/projects/bddur51/artefact-context/pipeline     # code repo
LUSTRE_ROOT=/nobackup/projects/bddur51                    # 10 TB Lustre
RUN_DIR=$LUSTRE_ROOT/oa_run_${SLURM_JOB_ID}               # per-job run folder

# Create dirs before cd (fixes the previous failure)
mkdir -p "$RUN_DIR" "$PROJ_ROOT/logs"

echo "=== Bede run context ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "RUN_DIR=$RUN_DIR"
echo "PROJ_ROOT=$PROJ_ROOT"
date
hostname

# ── Conda env ─────────────────────────────────────────────────────────────────
source /projects/bddur51/$USER/miniconda/etc/profile.d/conda.sh
conda activate download_works_env
python -V
which python

# ── Sharding knobs (read by your Python code) ─────────────────────────────────
export RUN_ROOT="$RUN_DIR"
export OA_SHARDS=32

# Pre-create shard folders incl. per-shard PDF buckets
for i in $(seq -w 0 $((OA_SHARDS-1))); do
  mkdir -p "$RUN_DIR/shards/shard_${i}/PDF_Bucket"
done

# ── Execute downloader ────────────────────────────────────────────────────────
cd "$RUN_DIR"
python "$PROJ_ROOT/batch_download_works.py"

# ── Merge shards → works.json + artists.json ──────────────────────────────────
echo "Merging sharded metadata..."
python "$PROJ_ROOT/merge_works_and_artists.py" \
  --run-root "$RUN_DIR" \
  --artist-json-dir "$PROJ_ROOT/Artist-JSONs"

# ── Final summary ─────────────────────────────────────────────────────────────
PDFS=$(find "$RUN_DIR/shards" -type f -name '*.pdf' | wc -l || echo 0)
WCOUNT=$(python - <<'PY'
import json,sys; 
from pathlib import Path
p=Path(sys.argv[1])
print(len(json.loads(p.read_text())) if p.exists() else 0)
PY
"$RUN_DIR/works.json")

ACOUNT=$(python - <<'PY'
import json,sys; 
from pathlib import Path
p=Path(sys.argv[1])
print(len(json.loads(p.read_text())) if p.exists() else 0)
PY
"$RUN_DIR/artists.json")

echo "----------------------------------------------------------------"
echo " PDFs (sharded): $PDFS"
echo " works.json entries: $WCOUNT"
echo " artists.json (artists): $ACOUNT"
echo " Output root: $RUN_DIR"
echo " Logs: /projects/bddur51/artefact-context/pipeline/logs/oa_${SLURM_JOB_ID}.{out,err}"
echo "----------------------------------------------------------------"
echo "Done."
###############################################################################
