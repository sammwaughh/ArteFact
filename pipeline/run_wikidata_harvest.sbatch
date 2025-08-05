#!/bin/bash
# -------- Slurm Directives --------
#SBATCH --job-name=wikidata_harvest
#SBATCH --account=bddur51
#SBATCH --partition=gpu          # gpu | infer | test
#SBATCH --gres=gpu:1             # policy: need ≥1 GPU on these partitions
#SBATCH --time=06:00:00          # 6 hours wallclock
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=logs/harvest_%j.out
#SBATCH --error=logs/harvest_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=samjmwaugh@gmail.com

set -euo pipefail

# -------- Conda env --------
source /projects/bddur51/$USER/miniconda/etc/profile.d/conda.sh
conda activate wikidata_harvest

# -------- Optional scratch dir --------
mkdir -p /scratch/$USER/harvest_$SLURM_JOB_ID
cd       /projects/bddur51/artefact-context/pipeline

# -------- Run the harvester --------
python batch_harvest_wikidata.py          # LIMIT = 32 000 inside the script
