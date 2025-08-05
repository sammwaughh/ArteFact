#!/bin/bash
# -------- Slurm Directives --------
#SBATCH --job-name=wikidata_harvest           # shown by squeue
#SBATCH --account=bddur51                     # charge to your project
#SBATCH --partition=gpu                       # gpu | infer | test  (all need GPUs)
#SBATCH --gres=gpu:1                          # MUST request ≥1 GPU on these partitions
#SBATCH --time=01:00:00                       # hh:mm:ss  (<= 2-00:00:00 on gpu)  :contentReference[oaicite:3]{index=3}
#SBATCH --ntasks=1                            # one process
#SBATCH --cpus-per-task=4                     # 4 CPU cores for requests/pandas
#SBATCH --mem=4G                              # adjust if necessary
#SBATCH --output=logs/harvest_%j.out          # stdout  (%j = job-ID)
#SBATCH --error=logs/harvest_%j.err           # stderr
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=samjmwaugh@gmail.com

set -euo pipefail                             # fail fast on any error

# -------- Load your Conda on the compute node --------
source /projects/bddur51/$USER/miniconda/etc/profile.d/conda.sh
conda activate wikidata_harvest

# -------- Optional: work in job-specific scratch dir --------
mkdir -p /scratch/$USER/harvest_$SLURM_JOB_ID
cd       /projects/bddur51/artefact-context/pipeline      # repo root

# -------- Run the Python harvester --------
python batch_harvest_wikidata.py
