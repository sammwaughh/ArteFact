#!/bin/bash
# -------- Slurm Directives --------
#SBATCH --job-name=wikidata_harvest          # appears in squeue
#SBATCH --partition=cpu                      # Bede’s CPU queue
#SBATCH --time=01:00:00                      # hh:mm:ss (enough for 1k rows)
#SBATCH --ntasks=1                           # one process
#SBATCH --cpus-per-task=4                    # 4 threads for requests/pandas
#SBATCH --mem=4G                             # RAM (adjust if needed)
#SBATCH --output=logs/harvest_%j.out         # stdout (%j = job-ID)
#SBATCH --error=logs/harvest_%j.err          # stderr

set -euo pipefail                            # fail fast on any error

# -------- Load your Conda on the compute node --------
source /projects/bddur51/$USER/miniconda/etc/profile.d/conda.sh
conda activate wikidata_harvest

# (Optional) put the run in a dated scratch dir
mkdir -p /scratch/$USER/harvest_$SLURM_JOB_ID
cd       /projects/bddur51/artefact-context/pipeline   # repo root

# -------- Run the Python harvester --------
python batch_harvest_wikidata.py
