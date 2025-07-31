# ArtContext

A computational pipeline for harvesting, processing, and analyzing academic literature about visual art using OpenAlex, Wikidata, and multimodal embeddings.

## Overview

ArtContext automates the collection and analysis of scholarly articles about painters and their works. The pipeline:
1. Harvests painter metadata from Wikidata
2. Queries OpenAlex for academic papers about each painter
3. Downloads available PDFs
4. Converts PDFs to Markdown for text processing
5. Extracts sentences and generates multimodal embeddings using CLIP

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

### 6. Generate Embeddings
```bash
python batch_embed_sentences.py
```
- Generates CLIP embeddings for all sentences
- Generates PaintingCLIP embeddings (fine-tuned model)
- Saves embeddings to `CLIP_Embeddings/` and `PaintingCLIP_Embeddings/`
- Uses helper: `embed_sentence_with_clip.py`

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

## Directory Structure

### Data Directories
- `Artist-JSONs/` - OpenAlex query results per artist
- `PDF_Bucket/` - Downloaded PDF files organized by artist
- `Marker_Output/` - Converted Markdown files from PDFs
- `CLIP_Embeddings/` - Standard CLIP text embeddings
- `PaintingCLIP_Embeddings/` - Fine-tuned CLIP embeddings
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
- `sentences.json` - Extracted sentences with embedding status
- `topics.json` - Topic to work ID mapping
- `painters.xlsx` - Master list of painters to process
- `paintings.xlsx` - Painting metadata

## Requirements

See `requirements.txt` for Python dependencies. Key requirements:
- Python 3.8+
- OpenAlex API access (free, requires email)
- Wikidata SPARQL access
- GPU recommended for embedding generation

## Configuration

1. Set your email in scripts that query OpenAlex (required by their API)
2. Adjust batch sizes and concurrency settings based on your system
3. Configure logging levels in individual scripts

## Usage Notes

- The pipeline is designed to be resumable - it checks for existing data before reprocessing
- Most scripts support filtering by artist name or work ID for selective processing
- Embedding generation is computationally intensive - consider running on GPU
- OpenAlex API has rate limits - the scripts include automatic retry logic

## Output Format

The pipeline produces structured JSON files that can be used for downstream analysis:
- Sentence-level embeddings for semantic search
- Work metadata for bibliometric analysis
- Topic associations for thematic studies