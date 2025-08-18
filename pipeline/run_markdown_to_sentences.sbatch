#!/bin/bash
###############################################################################
# Markdown → English sentences (Bede, uses RUN_ROOT from oa_run_914266)
###############################################################################
#SBATCH --job-name=md2sent
#SBATCH --account=bddur51
#SBATCH --partition=gpu
#SBATCH --gres=half:1
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=/projects/bddur51/artefact-context/pipeline/logs/md2sent_%j.out
#SBATCH --error=/projects/bddur51/artefact-context/pipeline/logs/md2sent_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=samjmwaugh@gmail.com

set -euo pipefail

# Paths
PROJ_ROOT=/projects/bddur51/artefact-context/pipeline
RUN_ROOT=/nobackup/projects/bddur51/oa_run_914266   # from download job 914266
export RUN_ROOT

echo "=== Bede Markdown→Sentences context ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "RUN_ROOT=$RUN_ROOT"
echo "PROJ_ROOT=$PROJ_ROOT"
date
hostname

# Conda
source /projects/bddur51/$USER/miniconda/etc/profile.d/conda.sh
conda env list || true
conda activate sentences_env
python -V
which python

# NLTK data (cache into RUN_ROOT to reduce repeated downloads)
export NLTK_DATA="$RUN_ROOT/nltk_data"
mkdir -p "$NLTK_DATA"
python - <<'PY'
import nltk, os
for res in ("punkt","punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{res}/english")
    except LookupError:
        try:
            nltk.download(res, quiet=False)
        except Exception as e:
            print(f"WARNING: failed to download {res}: {e}")
PY

# Sanity check inputs
MD_COUNT=$(find "$RUN_ROOT/Marker_Output" -type f -name "*.md" | wc -l || echo 0)
echo "Found $MD_COUNT markdown files under $RUN_ROOT/Marker_Output"

# Run batch conversion
cd "$PROJ_ROOT"
python batch_markdown_file_to_english_sentences.py

# Summary
SENTENCES_JSON="$RUN_ROOT/sentences.json"
WORKS_JSON="$RUN_ROOT/works.json"

SCOUNT=$(python - <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
print(len(json.loads(p.read_text())) if p.exists() else 0)
PY
"$SENTENCES_JSON")

WCOUNT=$(python - <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
print(len(json.loads(p.read_text())) if p.exists() else 0)
PY
"$WORKS_JSON")

echo "----------------------------------------------------------------"
echo " Markdown → Sentences complete"
echo " Markdown files seen: $MD_COUNT"
echo " sentences.json entries: $SCOUNT"
echo " works.json entries:     $WCOUNT"
echo " Output root:            $RUN_ROOT"
echo " Logs:                   $RUN_ROOT/logs/Markdown-To-Sentences-Logs/"
echo "----------------------------------------------------------------"
echo "Done."
###############################################################################