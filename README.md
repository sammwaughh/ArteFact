# ArteFact v2 (Local Edition) &nbsp;&nbsp;<img src="viewer/public/images/logo-16-9.JPEG" alt="ArteFact Logo" width="320">

Modern web application for analysing artwork with machine-learning models.  
Upload paintings, then receive academic annotations, contextual labels and confidence scores – all processed locally on your machine.

---

## Overview

This is a simplified, local-only version of ArteFact that runs entirely on your machine without any cloud services. Perfect for development, testing, or running the application offline.

## Architecture

| Component     | Technology          | Port | Role |
|---------------|---------------------|------|------|
| **Frontend**  | React + Vite SPA    | 5173 | Upload UI, polling, image viewer with label overlays |
| **Backend**   | Flask API           | 8000 | File uploads, run management, serving files |
| **Processing**| Python threads      | N/A  | Background ML inference (currently using dummy data) |
| **Storage**   | Local filesystem    | N/A  | `artifacts/` for images, `outputs/` for results |

---

## 🛠 How it Works

<details>
<summary>Click to expand the flow diagram ▶</summary>

```text
User selects image
└─► (1) POST /presign ……………………………… Flask API
        ↳ returns local upload URL + runId

Browser uploads file (2)
└─► POST /upload/<runId> ……………………… Flask API
        ↳ saves to artifacts/<runId>.jpg

Browser creates run (3)
└─► POST /runs ………………………………………… Flask API
        • Creates run in memory (status=queued)
        • Starts background thread for processing

Background thread (4)
        • Updates status → processing
        • Runs inference (currently dummy data)
        • Saves outputs/<runId>.json
        • Updates status → done

Browser polls every 3s (5)
└─► GET /runs/<runId> …………………………… Flask API
        ↳ returns current status

When status=done:
└─► GET /outputs/<runId>.json (6) … Flask serves file
└─► GET /artifacts/<runId>.jpg (7)… Flask serves file
        ↳ React renders image + overlays ✅
```
</details>

### Request Flow

| Step | Endpoint | Description |
|------|----------|-------------|
| 1. **Get upload URL** | `POST /presign` | Returns a unique `runId` and local upload endpoint |
| 2. **Upload image** | `POST /upload/<runId>` | Saves image to `artifacts/` directory |
| 3. **Start processing** | `POST /runs` | Creates run record and starts background thread |
| 4. **Check status** | `GET /runs/<runId>` | Returns current status (queued → processing → done) |
| 5. **Get results** | `GET /outputs/<runId>.json` | Returns inference results when ready |
| 6. **Get image** | `GET /artifacts/<runId>.jpg` | Returns the uploaded image |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+
- 2GB free disk space (for ML dependencies)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/[your-repo]/artefact-context.git
   cd artefact-context
   ```

2. **Set up Python environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install --upgrade pip
   pip install -e .
   ```

3. **Set up frontend:**
   ```bash
   cd viewer
   npm install
   echo "VITE_API_URL=http://localhost:8000" > .env
   cd ..
   ```

### Running the Application

1. **Start the backend (in one terminal):**
   ```bash
   source venv/bin/activate
   python -m hc_services.runner.app
   ```
   You should see: `Running on http://127.0.0.1:8000`

2. **Start the frontend (in another terminal):**
   ```bash
   cd viewer
   npm run dev
   ```
   You should see: `Local: http://localhost:5173/`

3. **Use the app:**
   - Open http://localhost:5173 in your browser
   - Click "Select an image" and upload a JPEG or PNG
   - Wait for processing (3-5 seconds with dummy inference)
   - View your image with AI-generated labels!

---

## 📁 Project Structure

```
artefact-context/
├── hc_services/
│   └── runner/
│       ├── app.py          # Flask API server
│       ├── tasks.py        # Background processing logic
│       └── inference.py    # ML inference (currently dummy data)
├── viewer/                 # React frontend
│   ├── src/
│   └── package.json
├── artifacts/             # Uploaded images (created automatically)
├── outputs/               # Processing results (created automatically)
└── pyproject.toml         # Python package configuration
```

---

## 🔧 Configuration

### Environment Variables

- `SLEEP_SECS`: Simulate slow processing (e.g., `SLEEP_SECS=10`)
- `FORCE_ERROR`: Force processing to fail for testing (e.g., `FORCE_ERROR=1`)

Example:
```bash
SLEEP_SECS=5 python -m hc_services.runner.app
```

### Frontend Configuration

The frontend reads from `viewer/.env`:
- `VITE_API_URL`: Backend URL (default: `http://localhost:8000`)

---

## 🧪 Testing

1. **Test successful processing:**
   ```bash
   python -m hc_services.runner.app
   ```
   Upload an image - should complete in 3-5 seconds.

2. **Test slow processing:**
   ```bash
   SLEEP_SECS=10 python -m hc_services.runner.app
   ```
   Upload an image - should take ~10 seconds.

3. **Test error handling:**
   ```bash
   FORCE_ERROR=1 python -m hc_services.runner.app
   ```
   Upload an image - should show an error message.

---

## 🎯 Adding Real ML Inference

Currently, the app returns dummy labels. To add real ML inference:

1. Edit `hc_services/runner/inference.py`
2. Load your model in the `run_inference()` function
3. Process the image and return results in the expected format:
   ```python
   [
       {
           "label": "detected object/style",
           "score": 0.95,
           "evidence": {"note": "explanation"}
       }
   ]
   ```

---

## 📝 Notes

- All data is stored locally in `artifacts/` and `outputs/` directories
- The application uses in-memory storage for run metadata (resets on restart)
- No external services or cloud resources are required
- Perfect for development, testing, or offline use

---

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

## 📄 License

[Your license here]
