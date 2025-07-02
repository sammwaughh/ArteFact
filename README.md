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

## 🚀 Quick Start - Complete Setup Instructions

### Prerequisites
- Python 3.9+ (check with `python3 --version`)
- Node.js 16+ (check with `node --version`)
- Git (check with `git --version`)
- 2GB free disk space (for ML dependencies)

### Step 1: Clone and Checkout

```bash
# Clone the repository
git clone https://github.com/[your-repo]/artefact-context.git
cd artefact-context

# IMPORTANT: Switch to the 'local' branch
git checkout local

# Verify you're on the correct branch
git branch --show-current
# Should output: local
```

### Step 2: Set up Python Backend

```bash
# Create a Python virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# You should now see (venv) at the start of your terminal prompt

# Upgrade pip
pip install --upgrade pip

# Install the project and all dependencies
pip install -e .

# This will install Flask, torch, pillow, and other dependencies
# Note: torch is large (2GB+) so this may take a few minutes
```

### Step 3: Set up React Frontend

```bash
# Navigate to the frontend directory
cd viewer

# Install Node dependencies
npm install

# Create the environment file for the frontend
echo "VITE_API_URL=http://localhost:8000" > .env

# Return to the project root
cd ..
```

### Step 4: Run the Application

You'll need **two terminal windows** for this:

#### Terminal 1 - Backend Server:
```bash
# Make sure you're in the project root directory
cd /path/to/artefact-context

# Activate the virtual environment (if not already active)
source venv/bin/activate  # macOS/Linux
# or
# venv\Scripts\activate   # Windows

# Start the Flask backend
python -m hc_services.runner.app

# You should see:
# * Running on http://127.0.0.1:8000
# Press CTRL+C to quit
```

#### Terminal 2 - Frontend Server:
```bash
# Open a new terminal
cd /path/to/artefact-context/viewer

# Start the React development server
npm run dev

# You should see:
# VITE v5.x.x  ready in xxx ms
# ➜  Local:   http://localhost:5173/
```

### Step 5: Test the Application

1. Open your web browser and go to **http://localhost:5173**
2. Click **"Select an image"** button
3. Choose any JPEG or PNG image file
4. Watch the upload and processing:
   - You'll see "Processing..." with a spinner
   - After 3-5 seconds, your image will appear with AI-generated labels
5. The labels currently show dummy data (leaf motif, gilded frame, Renaissance style)

### Verify Everything is Working

Check that files are being created:
```bash
# From the project root
ls artifacts/   # Should show uploaded images like: abc123def456.jpg
ls outputs/     # Should show JSON results like: abc123def456.json
```

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

---

## 🧪 Testing Different Scenarios

### Test Slow Processing
```bash
# In Terminal 1, stop the server (CTRL+C) and restart with:
SLEEP_SECS=10 python -m hc_services.runner.app
```
Upload an image - it will take ~10 seconds to process.

### Test Error Handling
```bash
# In Terminal 1, stop the server (CTRL+C) and restart with:
FORCE_ERROR=1 python -m hc_services.runner.app
```
Upload an image - you'll see an error message instead of results.

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

## ⚠️ Common Issues & Solutions

### Issue: `ModuleNotFoundError: No module named 'hc_services'`
**Solution**: Make sure you ran `pip install -e .` from the project root

### Issue: Frontend can't connect to backend
**Solution**: 
- Verify Flask is running on port 8000
- Check that `viewer/.env` contains `VITE_API_URL=http://localhost:8000`
- Make sure both servers are running

### Issue: `command not found: python3`
**Solution**: Install Python 3.9+ from https://www.python.org/downloads/

### Issue: `command not found: npm`
**Solution**: Install Node.js from https://nodejs.org/

---

## 📝 Notes

- All data is stored locally in `artifacts/` and `outputs/` directories
- The application uses in-memory storage for run metadata (resets on restart)
- No external services or cloud resources are required
- Perfect for development, testing, or offline use

---