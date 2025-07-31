## Using Bede for the ArtContext Pipeline

After analyzing your pipeline and the Bede documentation, here's a comprehensive guide for running your ArtContext pipeline on Bede's HPC infrastructure.

### Overview

Your pipeline has six main stages that can benefit from Bede's resources:
1. **Wikidata Harvest** - Network I/O bound, minimal compute
2. **OpenAlex Queries** - Network I/O bound, minimal compute  
3. **PDF Downloads** - Network I/O bound, high storage
4. **PDF→Markdown** - CPU intensive, high storage
5. **Sentence Extraction** - CPU intensive, moderate memory
6. **Embedding Generation** - GPU intensive (H100 ideal), high storage

Given ~100GB PDFs and ~100GB embeddings expected, you'll want to use `/nobackup/projects/$PROJECT` for bulk storage.

### Initial Setup on Bede

```bash
# After logging in to Bede
cd /nobackup/projects/$PROJECT  # Replace $PROJECT with your project name
git clone https://github.com/sammwaughh/ArtContext.git
cd ArtContext

# Create a virtual environment
module load python/3.11  # or latest available
python -m venv artcontext_env
source artcontext_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create a .env file for your email (don't commit this!)
echo "OPENALEX_EMAIL=samjmwaugh@gmail.com" > .env
```

### Stage-by-Stage Implementation

#### Stage 1-3: Data Collection (CPU-only)
These stages are I/O bound and can run on a single CPU node:

**`slurm/data_collection.slurm`**:
```bash
#!/bin/bash
#SBATCH --job-name=artcontext_collect
#SBATCH --account=<your_project>
#SBATCH --partition=gpu  # Even though we don't need GPU, this gives us more resources
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --output=/nobackup/projects/<your_project>/ArtContext/logs/collect_%j.out
#SBATCH --error=/nobackup/projects/<your_project>/ArtContext/logs/collect_%j.err

cd /nobackup/projects/<your_project>/ArtContext
source artcontext_env/bin/activate

echo "=== Stage 1: Wikidata Harvest ==="
python batch_harvest_wikidata.py

echo "=== Stage 2: OpenAlex Queries ==="
python batch_query_open_alex.py

echo "=== Stage 3: PDF Downloads ==="
python batch_download_works.py

echo "=== Data collection complete ==="
```

#### Stage 4: PDF to Markdown (CPU-intensive)
This benefits from parallel processing:

**`slurm/pdf_to_markdown.slurm`**:
```bash
#!/bin/bash
#SBATCH --job-name=artcontext_pdf2md
#SBATCH --account=<your_project>
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16  # Use multiple cores
#SBATCH --mem=64G
#SBATCH --output=/nobackup/projects/<your_project>/ArtContext/logs/pdf2md_%j.out
#SBATCH --error=/nobackup/projects/<your_project>/ArtContext/logs/pdf2md_%j.err

cd /nobackup/projects/<your_project>/ArtContext
source artcontext_env/bin/activate

# Marker-pdf can use multiple threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python batch_pdf_to_markdown.py
```

#### Stage 5: Sentence Extraction (CPU)
**`slurm/extract_sentences.slurm`**:
```bash
#!/bin/bash
#SBATCH --job-name=artcontext_sentences
#SBATCH --account=<your_project>
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --output=/nobackup/projects/<your_project>/ArtContext/logs/sentences_%j.out
#SBATCH --error=/nobackup/projects/<your_project>/ArtContext/logs/sentences_%j.err

cd /nobackup/projects/<your_project>/ArtContext
source artcontext_env/bin/activate

python batch_markdown_file_to_english_sentences.py
```

#### Stage 6: Embedding Generation (GPU-intensive)
This is where Bede's GPUs shine. For best performance, use the Grace-Hopper H100:

**`slurm/generate_embeddings_gh.slurm`**:
```bash
#!/bin/bash
#SBATCH --job-name=artcontext_embed
#SBATCH --account=<your_project>
#SBATCH --partition=gh
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1  # Full H100 node
#SBATCH --output=/nobackup/projects/<your_project>/ArtContext/logs/embed_%j.out
#SBATCH --error=/nobackup/projects/<your_project>/ArtContext/logs/embed_%j.err

cd /nobackup/projects/<your_project>/ArtContext
source artcontext_env/bin/activate

# Set cache directories to avoid quota issues
export HF_HOME=/nobackup/projects/<your_project>/huggingface_cache
export TORCH_HOME=/nobackup/projects/<your_project>/torch_cache
mkdir -p $HF_HOME $TORCH_HOME

python batch_embed_sentences.py
```

### Modified Scripts for HPC

You'll need to make minor modifications to handle Bede's environment:

**Update batch_embed_sentences.py** to handle Bede's GPU detection:
```python
# Replace the DEVICE detection with:
DEVICE = (
    "cuda" if torch.cuda.is_available() 
    else "cpu"
)
# Remove MPS check as it's Mac-specific
```

**Update scripts using hardcoded email** to read from environment:
```python
import os
from dotenv import load_dotenv
load_dotenv()

CONTACT_EMAIL = os.getenv('OPENALEX_EMAIL', 'your-email@example.com')
```

### Running the Complete Pipeline

Create a master script **`slurm/run_pipeline.sh`**:
```bash
#!/bin/bash
# Submit jobs with dependencies

# Data collection
COLLECT_JOB=$(sbatch --parsable slurm/data_collection.slurm)
echo "Data collection job: $COLLECT_JOB"

# PDF to Markdown (depends on collection)
PDF_JOB=$(sbatch --parsable --dependency=afterok:$COLLECT_JOB slurm/pdf_to_markdown.slurm)
echo "PDF conversion job: $PDF_JOB"

# Sentence extraction (depends on PDF conversion)
SENT_JOB=$(sbatch --parsable --dependency=afterok:$PDF_JOB slurm/extract_sentences.slurm)
echo "Sentence extraction job: $SENT_JOB"

# Embedding generation (depends on sentence extraction)
EMBED_JOB=$(sbatch --parsable --dependency=afterok:$SENT_JOB slurm/generate_embeddings_gh.slurm)
echo "Embedding generation job: $EMBED_JOB"

echo "Pipeline submitted. Monitor with: squeue -u $USER"
```

Run with:
```bash
chmod +x slurm/run_pipeline.sh
./slurm/run_pipeline.sh
```

### Data Transfer Back to Local

After completion, create **`transfer_results.sh`**:
```bash
#!/bin/bash
# Run this from your local machine

PROJECT="your_project_name"
REMOTE_DIR="bede.dur.ac.uk:/nobackup/projects/$PROJECT/ArtContext"
LOCAL_DIR="~/Desktop/ArtContext_Results"

mkdir -p $LOCAL_DIR

# Transfer JSON files
rsync -avz --progress \
  $REMOTE_DIR/*.json \
  $LOCAL_DIR/

# Transfer embeddings (this will be ~100GB)
rsync -avz --progress \
  $REMOTE_DIR/CLIP_Embeddings/ \
  $LOCAL_DIR/CLIP_Embeddings/

rsync -avz --progress \
  $REMOTE_DIR/PaintingCLIP_Embeddings/ \
  $LOCAL_DIR/PaintingCLIP_Embeddings/

# Optionally transfer logs
rsync -avz --progress \
  $REMOTE_DIR/logs/ \
  $LOCAL_DIR/logs/
```

### Best Practices for Your Pipeline

1. **Storage Management**:
   - Use `/nobackup/projects/$PROJECT` for all large files
   - Set cache directories for HuggingFace and PyTorch models
   - Clean up intermediate files (Marker_Output) after embedding generation

2. **Resource Requests**:
   - Stages 1-3: Low memory, long walltime for network I/O
   - Stage 4: High CPU count for parallel PDF processing
   - Stage 6: Use Grace-Hopper (H100) for fastest embedding generation

3. **Monitoring**:
   ```bash
   squeue -u $USER  # Check job status
   tail -f logs/embed_*.out  # Monitor progress
   ```

4. **Checkpointing**:
   - Your pipeline already handles resumption well
   - Consider adding periodic saves in embedding generation

5. **Testing First**:
   - Test with a small subset using `--partition=ghtest` (30 min limit)
   - Verify GPU detection and model loading before full runs

### Example Test Run

Before running the full pipeline:
```bash
# Create test subset
head -10 painters.xlsx > painters_test.xlsx
mv painters.xlsx painters_full.xlsx
mv painters_test.xlsx painters.xlsx

# Submit test job
sbatch --partition=ghtest --time=00:30:00 slurm/generate_embeddings_gh.slurm

# Restore full list after test
mv painters.xlsx painters_test.xlsx
mv painters_full.xlsx painters.xlsx
```

This approach gives you a robust, resumable pipeline that leverages Bede's resources effectively without over-engineering. The modular structure allows you to re-run individual stages as needed, and the dependency chain ensures proper ordering.