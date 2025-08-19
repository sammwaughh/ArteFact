#!/bin/bash
#SBATCH --job-name=enrich_works
#SBATCH --output=slurm_enrich_works_%j.out
#SBATCH --error=slurm_enrich_works_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --partition=standard

# Load conda environment
source /projects/bddur51/samjmwaugh/miniconda/etc/profile.d/conda.sh
conda activate embedding_env

# Set RUN_ROOT (update this path as needed)
export RUN_ROOT=/nobackup/projects/bddur51/oa_run_914266

echo "🚀 Starting enrich_works_metadata.py job"
echo "�� Job started at: $(date)"
echo "🖥️  Running on: $(hostname)"
echo "📁 RUN_ROOT: $RUN_ROOT"
echo "�� Current directory: $(pwd)"

# Check if works.json exists
if [ ! -f "$RUN_ROOT/works.json" ]; then
    echo "❌ Error: works.json not found at $RUN_ROOT/works.json"
    exit 1
fi

echo "📖 Found works.json, starting metadata enrichment..."

# Run the script
cd /projects/bddur51/artefact-context
python pipeline/enrich_works_metadata.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ enrich_works_metadata.py completed successfully"
    echo "📁 Enriched works file created at: $RUN_ROOT/works_enriched.json"
else
    echo "❌ enrich_works_metadata.py failed"
    exit 1
fi

echo "📅 Job completed at: $(date)"