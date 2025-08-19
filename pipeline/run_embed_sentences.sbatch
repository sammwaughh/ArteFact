#!/bin/bash
###############################################################################
# Efficient Batch Sentence Embedding Generation (Bede)
###############################################################################
#SBATCH --job-name=embed-sentences
#SBATCH --account=bddur51
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/projects/bddur51/artefact-context/pipeline/logs/embed_sentences_%j.out
#SBATCH --error=/projects/bddur51/artefact-context/pipeline/logs/embed_sentences_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=samjmwaugh@gmail.com

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJ_ROOT=/projects/bddur51/artefact-context/pipeline
RUN_ROOT=/nobackup/projects/bddur51/oa_run_914266   # from sentence extraction job
export RUN_ROOT

echo "=== Bede Sentence Embedding Generation ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "RUN_ROOT=$RUN_ROOT"
echo "PROJ_ROOT=$PROJ_ROOT"
date
hostname

# ── Conda environment ─────────────────────────────────────────────────────────
source /projects/bddur51/$USER/miniconda/etc/profile.d/conda.sh
conda activate sentences_env
python -V
which python

# ── Verify inputs ─────────────────────────────────────────────────────────────
echo "Verifying input data..."
SENTENCES_COUNT=$(python - <<'PY'
import json,os,pathlib
rr=os.environ["RUN_ROOT"]
p=pathlib.Path(rr)/"sentences.json"
if p.exists():
    d=json.loads(p.read_text())
    print(len(d))
else:
    print(0)
PY
)

if [ "$SENTENCES_COUNT" -eq 0 ]; then
    echo "❌ No sentences found in $RUN_ROOT/sentences.json"
    exit 1
fi

echo "✅ Found $SENTENCES_COUNT sentences to process"

# ── Check PaintingCLIP adapter ───────────────────────────────────────────────
if [ -d "$RUN_ROOT/PaintingCLIP" ]; then
    echo "✅ PaintingCLIP adapter found"
else
    echo "⚠️  PaintingCLIP adapter not found - will skip PaintingCLIP embeddings"
fi

# ── Run embedding generation ─────────────────────────────────────────────────
cd "$PROJ_ROOT"
echo "Starting embedding generation..."
python efficient_batch_embed_sentences.py

# ── Final summary ─────────────────────────────────────────────────────────────
echo "----------------------------------------------------------------"
echo " Embedding generation complete"
echo " Check logs for detailed results"
echo " Output: $RUN_ROOT/Embeddings/"
echo " Backend: $PROJ_ROOT/../data/embeddings/"
echo "----------------------------------------------------------------"

# ── Verify outputs ───────────────────────────────────────────────────────────
echo "Verifying outputs..."
python - <<'PY'
import json,os,pathlib
rr=os.environ["RUN_ROOT"]
emb_dir=pathlib.Path(rr)/"Embeddings"

# Check safetensors files
clip_st=emb_dir/"CLIP_embeddings.safetensors"
painting_st=emb_dir/"PaintingCLIP_embeddings.safetensors"

print("CLIP embeddings:", "✅" if clip_st.exists() else "❌")
if painting_st.exists():
    print("PaintingCLIP embeddings:", "✅")
else:
    print("PaintingCLIP embeddings:", "⚠️ (not generated)")

# Check sentence ID files
clip_ids=emb_dir/"CLIP_embeddings_sentence_ids.json"
painting_ids=emb_dir/"PaintingCLIP_embeddings_sentence_ids.json"

if clip_ids.exists():
    with open(clip_ids) as f:
        clip_count=len(json.load(f))
    print(f"CLIP sentence IDs: {clip_count}")
if painting_ids.exists():
    with open(painting_ids) as f:
        painting_count=len(json.load(f))
    print(f"PaintingCLIP sentence IDs: {painting_count}")
PY

echo "Done."
###############################################################################