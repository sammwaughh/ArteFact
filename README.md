# ArteFact

A web-based application for analyzing artwork images using PaintingCLIP, a fine-tuned CLIP model specialized for art-historical content. The system matches uploaded artwork images against a corpus of pre-computed sentence embeddings from art history texts, returning the most semantically similar passages.

## Overview

This application consists of:
- A Flask backend API that handles image uploads and runs ML inference
- A vanilla JavaScript frontend for image upload and result display
- A PaintingCLIP model for specialized art analysis
- Pre-computed embeddings from art-historical texts

## Prerequisites

- Python 3.8+
- pip
- A modern web browser

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd artefact-context
```

### 2. Set Up Python Environment

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -e backend/          # installs from backend/pyproject.toml
```

### 3. Verify Model Files

Ensure you have the following directories with their required files:
- PaintingCLIP - LoRA adapter files for the fine-tuned model
- PaintingCLIP_Embeddings - Pre-computed sentence embeddings (`.pt` files)

## Running the Application

### Quick start (two terminals)

1. **Backend**

```bash
# First time only
chmod +x backend/run_backend.sh
./backend/run_backend.sh          # → http://localhost:8000
```

2. **Frontend**

```bash
# First time only
chmod +x frontend/run_frontend.sh
./frontend/run_frontend.sh        # → http://localhost:8080
```

Then open `http://localhost:8080` in your browser.

### Manual start

```bash
# Backend
python -m backend.runner.app      # or: python backend/runner/app.py
# Frontend (simple HTTP server)
cd frontend && python -m http.server 8080
```

## Usage

1. **Choose or Upload an Artwork**

   • Drag & drop an image, click “Upload Image”, or pick one of the example
   thumbnails.  
   • The chosen image appears in the centre panel and the working overlay
   pops up.

2. **Select Model (Optional)**

   Use the “Model: …” dropdown in the navbar to switch between *CLIP* and
   *PaintingCLIP* before you run the query.

3. **Apply Filters (Optional)**

   • Click topic pills to restrict retrieval to specific subject codes.  
   • Search for creators in the right-hand panel or in the header search
   box and add them as tags.  
   • Selected topics/creators are shown under the image.

4. **Process the Image**

   The backend receives the upload, computes the embedding and returns the
   most similar passages. Progress is streamed in the debug overlay; the
   sidebar stays hidden until results arrive.

5. **Explore the Results**

   • Top 10 sentences appear in the left sidebar.  
   • Click the magnifying-glass icon to fetch full metadata / DOI;
     a banner with BibTeX and an iframe preview opens at the top.

6. **Region-Specific Search (Optional)**

   • Open *Settings → View Grid* to overlay a 7 × 7 grid.  
   • Click any cell to rerank the corpus for that region only; sentences
     refresh instantly and the clicked area flashes briefly.

7. **Image Tools**

   • **Crop** – draw a rectangle to analyse a sub-region.  
   • **Undo** – revert to the previous image state.  
   • **Rerun** – re-execute the pipeline with the current image (useful
     after changing model or filters).  
   • All previous versions are kept in the Image History strip; click any
     thumbnail to restore it.

8. **Debug Panel**

   Click the circled “i” button to show/hide a real-time log of network
   requests, run-IDs and backend status.

Uploaded images are stored in `data/artifacts/`, JSON results in
`data/outputs/`; both folders are auto-created at runtime.

## Project Structure

```
artefact-context/
├── backend/
│   ├── runner/                 # Flask backend source
│   │   ├── app.py              # API server
│   │   ├── tasks.py            # Background job runner
│   │   ├── inference.py        # PaintingCLIP inference pipeline
│   │   └── …                   # heatmap.py, patch_inference.py, …
│   ├── run_backend.sh          # Helper script → starts the backend
│   └── tests/                  # Unit tests
├── data/
│   ├── artifacts/              # Uploaded images  (created at runtime)
│   ├── outputs/                # JSON results     (created at runtime)
│   ├── embeddings/
│   │   ├── CLIP_Embeddings/
│   │   └── PaintingCLIP_Embeddings/
│   ├── json_info/              # Metadata JSONs: sentences.json, works.json, …
│   └── models/
│       └── PaintingCLIP/       # LoRA adapter files
├── frontend/
│   ├── index.html              # Single-page UI
│   ├── js/artefact-context.js  # Front-end logic
│   ├── css/artefact-context.css
│   └── run_frontend.sh         # Simple static-server wrapper
└── README.md
```

## Data File Structures

### sentences.json
Maps sentence IDs to their metadata. Currently contains minimal metadata:

```json
{
  "W1982215463_s0001": {
    "English Original": "The actual sentence text...",
    "Has PaintingCLIP Embedding": true
  }
}
```

**Note**: The inference pipeline adds the "English Original" field dynamically from other sources if not present in this file.

### works.json
Contains rich metadata for every source work (book, article, catalogue …).

```json
{
  "W4206160935": {
    "Author_Name": "John Guille Millais",
    "Work_Title": "The Life and Letters of Sir John Everett Millais",
    "Year": "2012",
    "Link": "https://archive.org/download/lifelettersofsir01milliala/lifelettersofsir01milliala_bw.pdf",
    "DOI": "https://doi.org/10.1017/cbo9781139343862",
    "Number of Sentences": 4874,
    "ImageIDs": [],
    "TopicIDs": [
      "C2778983918", "C520712124", "C205783811", "C554144382",
      "C142362112", "C52119013", "C2780586970", "C543847140",
      "C95457728",  "C153349607", "C199539241", "C138885662",
      "C27206212",  "C17744445"
    ],
    "Relevance": 3.7782803,
    "BibTeX": "@misc{W4206160935,\\n  author = \"John Guille Millais\",\\n  title = \"{The Life and Letters of Sir John Everett Millais}\",\\n  year = \"2012\",\\n  publisher = \"{Cambridge University Press}\",\\n  doi = \"{10.1017/cbo9781139343862}\",\\n  url = \"{https://doi.org/10.1017/cbo9781139343862}\",\\n}"
  }
}
```

Key fields  
• `Author_Name`, `Work_Title`, `Year` – human-readable bibliographic data  
• `DOI` / `Link` – external identifiers for look-ups and previews  
• `Number of Sentences` – total sentences extracted for the corpus  
• `TopicIDs` – subject codes that reference this work  
• `ImageIDs` – images associated with the work (can be empty)  
• `Relevance` – pre-computed relevance score used for ranking  
• `BibTeX` – ready-to-copy

### topics.json
Maps topic IDs to lists of work IDs that cover that topic:

```json
{
  "C2778983918": ["W4206160935"],
  "C520712124": ["W4206160935", "W1234567890"]
}
```

### topic_names.json
Human-readable names for topic IDs:

```json
{
  "C52119013": "Art History",
  "C204034006": "Art Criticism",
  "C501303744": "Iconography"
}
```

### creators.json
Maps artist/creator names to their associated works:

```json
{
  "arthur_hughes": ["W4206160935", "W2029124454", ...],
  "francesco_hayez": ["W1982215463", "W4388661114", ...],
  "george_stubbs": ["W2020798572", "W2021094421", ...]
}
```

## API Endpoints

- `POST /presign` - Request upload credentials
- `POST /upload/<runId>` - Upload image file
- `POST /runs` - Start inference job
- `GET /runs/<runId>` - Check job status
- `GET /outputs/<filename>` - Retrieve inference results
- `GET /work/<id>` - Get work metadata for DOI lookup

## Key Components

### backend/runner

| File | Purpose |
|------|---------|
| **app.py** | Flask API gateway. Handles image upload, run creation, status queries, file serving and auxiliary routes (topics, creators, heatmap, etc.). |
| **tasks.py** | Lightweight job runner executed via `ThreadPoolExecutor`. Pulls images from `data/artifacts`, calls `inference.run_inference`, stores JSON to `data/outputs`, and updates in-memory run status. |
| **inference.py** | Core CLIP / PaintingCLIP retrieval pipeline. Loads the model (with optional LoRA adapter), reads pre-computed embeddings, performs similarity search and optional Grad-ECLIP heat-map generation. |
| **patch_inference.py** | Region-aware extension that converts ViT patch tokens into a 2-D grid and re-ranks sentences for a user-clicked cell (7 × 7 by default). |
| **heatmap.py** | Implements Grad-ECLIP for visualising which image regions contribute most to a given sentence match; returns PNG overlays. |
| **filtering.py** | Utility layer that filters sentences by selected topic IDs or creator names using the metadata in `data/json_info/*.json`. |
| **\_\_init\_\_.py** | Marks the folder as a Python package; no runtime logic. |

### frontend

*`frontend/js/artefact-context.js`* – jQuery/Bootstrap SPA that handles image input, filter selection, grid overlay, cropping/undo, polling, and result rendering.

### Data folders

* `data/artifacts/` – uploaded images  
* `data/outputs/` – JSON results  
* `data/embeddings/` – sentence vectors for CLIP / PaintingCLIP  
* `data/json_info/` – metadata JSONs (`sentences.json`, `works.json`, …)  
* `data/models/PaintingCLIP/` – LoRA adapter weights

## Troubleshooting

1. **Port Already in Use**: If ports 8000 or 8080 are occupied, modify the port numbers in:
   - app.py (line with `app.run()`)
   - `run.sh` 
   - artefact-context.js (API_BASE_URL)

2. **Missing Dependencies**: Ensure all packages are installed:
```bash
pip install -e backend/
```

3. **Model Loading Errors**: Verify that:
   - PaintingCLIP directory contains LoRA adapter files
   - PaintingCLIP_Embeddings contains `.pt` files

4. **CORS Issues**: The backend is configured to accept requests from any origin. For production, update CORS settings in app.py.

## Development Notes

- The system uses an in-memory store for job tracking (resets on server restart)
- Uploaded images are saved to **data/artifacts**
- Inference results are saved to **data/outputs**
- The frontend uses jQuery and Bootstrap for UI components
- Debug panel available via the (i) button in the bottom-right corner