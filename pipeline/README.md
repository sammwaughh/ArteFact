# ArtContext

A computational pipeline for harvesting, processing, and analysing academic literature about visual art using OpenAlex, Wikidata, and multimodal embeddings. **Now featuring massive-scale processing on HPC clusters with 3.1M+ sentence embeddings.**

## Overview

ArtContext automates the collection and analysis of scholarly articles about painters and their works. The pipeline:
1. Harvests painter metadata from Wikidata
2. Queries OpenAlex for academic papers about each painter
3. Downloads available PDFs
4. Converts PDFs to Markdown for text processing
5. Extracts sentences and generates **large-scale multimodal embeddings using CLIP and PaintingCLIP**

## 🚀 New: HPC Processing at Scale

**Successfully processed 3.1 million sentences** on Durham University's Bede HPC cluster using **Grace Hopper GPUs**:

- **Total sentences**: 3,119,199
- **Processing time**: ~12 minutes on Grace Hopper
- **Total data generated**: ~33GB
- **GPU**: NVIDIA H100 with 32GB memory
- **Batch size**: 1,024 sentences
- **Processing speed**: ~9 batches/second

## Pipeline Workflow

The pipeline consists of batch scripts that should be run in the following order:

### 1. Wikidata Harvest
```bash
python batch_harvest_wikidata.py
```
- Queries Wikidata for painter information
- Populates `artists.json` with painter metadata
- Creates initial painter entries in `paintings.xlsx`

### 2. Query OpenAlex
```bash
python batch_query_open_alex.py
```
- Reads painter names from `painters.xlsx`
- Queries OpenAlex API for academic works mentioning each painter
- Saves results to `Artist-JSONs/<painter_name>.json`
- Uses helper: `query_open_alex_with.py`

### 3. Download Works
```bash
python batch_download_works.py
```
- Processes all artist JSON files in `Artist-JSONs/`
- Downloads available PDFs to `PDF_Bucket/`
- Updates `works.json` with download metadata
- Uses helpers: `download_works_on.py`, `download_single_work.py`

### 4. PDF to Markdown
```bash
python batch_pdf_to_markdown.py
```
- Converts downloaded PDFs to Markdown format
- Outputs to `Marker_Output/<work_id>/`
- Uses helper: `single_pdf_to_markdown.py`

### 5. Extract Sentences
```bash
python batch_markdown_file_to_english_sentences.py
```
- Extracts English sentences from Markdown files
- Updates `sentences.json` with extracted sentences
- Updates `works.json` with sentence counts
- Uses helper: `markdown_file_to_english_sentences.py`

### 6. Generate Embeddings (UPDATED)

#### **Local Processing (Small Scale)**
```bash
python batch_embed_sentences.py
```
- Generates CLIP embeddings for all sentences
- Generates PaintingCLIP embeddings (fine-tuned model)
- Saves embeddings to `CLIP_Embeddings/` and `PaintingCLIP_Embeddings/`
- Uses helper: `embed_sentence_with_clip.py`

#### **HPC Processing (Large Scale - RECOMMENDED)**
```bash
# Grace Hopper partition for GPU acceleration
sbatch run_embed_sentences_gh.sbatch
```
- **Processes 3.1M+ sentences** efficiently on HPC
- **Uses Grace Hopper GPUs** for maximum performance
- **Generates large-scale embeddings** (12.4GB total)
- **Outputs optimized safetensors format** for production use

## Additional Scripts

### Topic Analysis
```bash
python build_topics_json.py
```
- Creates reverse index of OpenAlex topics to works
- Generates `topics.json` from `works.json` data

### Painter List Generation
```bash
python generate_painter_list.py
```
- Generates/updates the painter list for processing

### Enriching Works Metadata
```bash
python enrich_works_metadata.py
```
- Adds BibTeX, Author, Year, Work info to each work object

## Directory Structure

### Data Directories
- `Artist-JSONs/` - OpenAlex query results per artist
- `PDF_Bucket/` - Downloaded PDF files organised by artist
- `Marker_Output/` - Converted Markdown files from PDFs
- **`Embeddings/` - NEW: Large-scale embeddings (12.4GB total)**
  - `clip_embeddings.safetensors` (6.2GB)
  - `paintingclip_embeddings.safetensors` (6.2GB)
  - `clip_embeddings_sentence_ids.json`
  - `paintingclip_embeddings_sentence_ids.json`
- `Excel-Files/` - Excel outputs and working files
- `logs/` - Execution logs for debugging

### Configuration & Models
- `PaintingCLIP/` - Fine-tuned CLIP adapter (LoRA weights)
- `Helper Scripts/` - Utility scripts for data cleaning and analysis
- `Archive/` - Previous versions and documentation
- `Scripts from Project/` - Legacy scripts (excluded from quality checks)

### Data Files
- `artists.json` - Artist metadata and associated work IDs
- `works.json` - Work metadata including URLs, sentences, topics
- **`sentences.json` - UPDATED: 3.1M sentences with embedding status**
- `topics.json` - Topic to work ID mapping
- `painters.xlsx` - Master list of painters to process
- `paintings.xlsx` - Painting metadata

## �� New Output Formats

### **Large-Scale Embeddings**
- **Format**: PyTorch safetensors (more efficient than .pt files)
- **CLIP embeddings**: 6.2GB for 3.1M sentences × 512 dimensions
- **PaintingCLIP embeddings**: 6.2GB for 3.1M sentences × 512 dimensions
- **Sentence ID mappings**: JSON files linking tensor indices to sentence IDs

### **Updated Metadata**
- **sentences.json**: Now contains 3.1M sentences with embedding status flags
- **Marker output**: Comprehensive document analysis results
- **Enhanced works.json**: Rich bibliographic metadata

## Requirements

See `requirements.txt` for Python dependencies. Key requirements:
- Python 3.8+
- OpenAlex API access (free, requires email)
- Wikidata SPARQL access
- **GPU recommended for embedding generation**
- **HPC access recommended for large-scale processing**

## Configuration

1. Set your email in scripts that query OpenAlex (required by their API)
2. Adjust batch sizes and concurrency settings based on your system
3. Configure logging levels in individual scripts
4. **For HPC processing**: Configure SLURM scripts in `slurm/` directory

## Usage Notes

- The pipeline is designed to be resumable - it checks for existing data before reprocessing
- Most scripts support filtering by artist name or work ID for selective processing
- **Embedding generation is computationally intensive** - consider running on GPU or HPC
- OpenAlex API has rate limits - the scripts include automatic retry logic
- **Large-scale processing**: Use HPC scripts for datasets >100K sentences

## HPC Processing Guide

### **Bede Cluster Setup**
```bash
# Access Grace Hopper partition
ghlogin -A <project> --time=4:00:00 -c 8 --mem=32G

# Submit embedding generation job
sbatch run_embed_sentences_gh.sbatch
```

### **SLURM Scripts Available**
- `run_embed_sentences_gh.sbatch` - Grace Hopper embedding generation
- `run_build_topics.sbatch` - Topic index building
- `run_pdf_to_markdown.sbatch` - PDF conversion pipeline

### **Performance on HPC**
- **Grace Hopper**: ~9 batches/second, 12 minutes total for 3.1M sentences
- **Memory efficient**: 32GB RAM sufficient for large-scale processing
- **GPU optimized**: Automatic mixed precision and memory management

## Output Format

The pipeline produces structured data optimized for production use:
- **Sentence-level embeddings** for semantic search (3.1M+ sentences)
- **Work metadata** for bibliometric analysis
- **Topic associations** for thematic studies
- **Large-scale safetensors** for efficient loading in production systems

## 🎯 Performance Improvements

With the new large-scale processing:
- **Corpus size**: Increased from development scale to production scale
- **Processing efficiency**: 12 minutes vs. hours on local systems
- **Storage format**: Safetensors for faster loading and better compression
- **Scalability**: Ready for even larger datasets

## Troubleshooting

### **HPC-Specific Issues**
- **"Requested node configuration not available"**: Submit jobs from Grace Hopper partition, not login nodes
- **Memory errors**: Ensure sufficient RAM allocation in SLURM scripts
- **GPU errors**: Verify CUDA environment setup

### **General Issues**
- **Missing dependencies**: Check requirements.txt and environment setup
- **API rate limits**: Scripts include automatic retry logic
- **Large file handling**: Use Git LFS for version control of large files

## Acknowledgements

**Special thanks to Durham University's Bede HPC cluster** for providing the computational resources needed to process this large-scale art history corpus using Grace Hopper GPUs.

This work made use of the facilities of the **N8 Centre of Excellence in Computationally Intensive Research (N8 CIR)** provided and funded by the N8 research partnership and EPSRC (Grant No. EP/T022167/1). The Centre is co-ordinated by the Universities of Durham, Manchester and York.

I also gratefully acknowledge the supervision and guidance of **Dr Stuart James (Department of Computer Science, Durham University).**

_Note: In line with N8 CIR policy, details of any publication or other public output arising from this project will be sent to **enquiries@n8cir.org.uk** on release._