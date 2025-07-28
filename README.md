# Artefact Context Viewer

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
pip install -r requirements.txt
```

### 3. Verify Model Files

Ensure you have the following directories with their required files:
- PaintingCLIP - LoRA adapter files for the fine-tuned model
- PaintingCLIP_Embeddings - Pre-computed sentence embeddings (`.pt` files)

## Running the Application

### Option 1: Using the Run Script (Recommended)

```bash
# Make the script executable (first time only)
chmod +x run.sh

# Run both frontend and backend
./run.sh
```

This will:
1. Start the Flask backend on `http://localhost:8000`
2. Start a simple HTTP server for the frontend on `http://localhost:8080`
3. Open your default browser to the application

### Option 2: Manual Start

#### Start the Backend

```bash
# Activate virtual environment if not already active
source .venv/bin/activate

# Run the Flask application
python -m hc_services.runner.app
```

The backend will start on `http://localhost:8000`

#### Start the Frontend

In a new terminal:

```bash
# Navigate to the viewer directory
cd viewer_js

# Start a simple HTTP server
python -m http.server 8080
```

Then open `http://localhost:8080` in your browser.

## Usage

1. **Upload an Image**: 
   - Click "Upload an Image" or drag-drop an image
   - Select from provided historical examples
   
2. **Process**: The system will:
   - Upload the image to the backend
   - Compute image embeddings using PaintingCLIP
   - Find the most similar sentences from the corpus
   
3. **View Results**: 
   - See the top 10 most relevant text passages
   - Click the search icon next to any result to view source metadata
   
4. **Image Tools**:
   - **Crop**: Select a region of interest
   - **Undo**: Revert to previous image
   - **Rerun**: Process the current image again

## Project Structure

```
artefact-context/
├── hc_services/
│   └── runner/
│       ├── app.py           # Flask API server
│       ├── tasks.py         # Background task processing
│       ├── inference.py     # PaintingCLIP inference pipeline
│       └── data/           # JSON data files
│           ├── sentences.json
│           ├── works.json
│           ├── topics.json
│           ├── topic_names.json
│           └── creators.json
├── viewer_js/              # Frontend application
│   ├── index.html
│   ├── js/
│   │   └── artefact-context.js
│   └── css/
│       └── artefact-context.css
├── PaintingCLIP/          # LoRA adapter files
├── PaintingCLIP_Embeddings/ # Pre-computed embeddings
├── artifacts/             # Uploaded images (created at runtime)
├── outputs/               # Inference results (created at runtime)
└── requirements.txt
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
Contains metadata about source works (papers, books, etc.):

```json
{
  "W4206160935": {
    "Artist": "arthur_hughes",
    "Link": "https://...",              // Direct PDF/document link
    "Number of Sentences": 4874,
    "DOI": "https://doi.org/...",      // Digital Object Identifier
    "ImageIDs": [],                     // Associated image IDs
    "TopicIDs": ["C2778983918", ...],  // Related topic IDs
    "Relevance": 3.7782803              // Relevance score
  }
}
```

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

### Backend (app.py)
- Flask server handling HTTP requests
- Manages file uploads and storage
- Coordinates inference jobs via thread pool

### Task Processing (tasks.py)
- Handles background inference jobs
- Updates job status in memory
- Writes results to JSON files

### Inference Pipeline (inference.py)
- Loads PaintingCLIP model with LoRA adapters
- Computes image embeddings
- Performs similarity search against sentence corpus
- Returns top-K most similar passages

### Frontend (artefact-context.js)
- Handles image upload and display
- Polls backend for job status
- Displays results and metadata
- Provides image manipulation tools

## Troubleshooting

1. **Port Already in Use**: If ports 8000 or 8080 are occupied, modify the port numbers in:
   - app.py (line with `app.run()`)
   - `run.sh` 
   - artefact-context.js (API_BASE_URL)

2. **Missing Dependencies**: Ensure all packages are installed:
   ```bash
   pip install -r requirements.txt
   ```

3. **Model Loading Errors**: Verify that:
   - PaintingCLIP directory contains LoRA adapter files
   - PaintingCLIP_Embeddings contains `.pt` files

4. **CORS Issues**: The backend is configured to accept requests from any origin. For production, update CORS settings in app.py.

## Development Notes

- The system uses an in-memory store for job tracking (resets on server restart)
- Uploaded images are saved to artifacts
- Inference results are saved to outputs
- The frontend uses jQuery and Bootstrap for UI components
- Debug panel available via the (i) button in the bottom-right corner