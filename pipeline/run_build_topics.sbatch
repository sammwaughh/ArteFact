#!/bin/bash
#SBATCH --job-name=build_topics
#SBATCH --output=slurm_build_topics_%j.out
#SBATCH --error=slurm_build_topics_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --partition=standard
#SBATCH --account=bddur51  # ← Add this line

# Load conda environment
source /projects/bddur51/samjmwaugh/miniconda/etc/profile.d/conda.sh
conda activate embedding_env

# Set RUN_ROOT (update this path as needed)
export RUN_ROOT=/nobackup/projects/bddur51/oa_run_914266

echo "�� Starting build_topics_json.py job"
echo "�� Job started at: $(date)"
echo "🖥️  Running on: $(hostname)"
echo "📁 RUN_ROOT: $RUN_ROOT"
echo "�� Current directory: $(pwd)"

# Check if works.json exists
if [ ! -f "$RUN_ROOT/works.json" ]; then
    echo "❌ Error: works.json not found at $RUN_ROOT/works.json"
    exit 1
fi

echo "📖 Found works.json, starting topic index building..."

# Run the script
cd /projects/bddur51/artefact-context
python pipeline/build_topics_json.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ build_topics_json.py completed successfully"
    echo "📁 Topics file created at: $RUN_ROOT/topics.json"
else
    echo "❌ build_topics_json.py failed"
    exit 1
fi

echo "📅 Job completed at: $(date)"