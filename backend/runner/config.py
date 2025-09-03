"""
Unified configuration for Hugging Face datasets integration.
All runner modules should import from this module instead of defining their own paths.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

# Try to import required libraries
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("⚠️  datasets library not available - HF dataset loading disabled")
    DATASETS_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    print("⚠️  huggingface_hub library not available - HF file loading disabled")
    HF_HUB_AVAILABLE = False

# Environment variables for dataset names
ARTEFACT_JSON_DATASET = os.getenv('ARTEFACT_JSON_DATASET', 'samwaugh/artefact-json')
ARTEFACT_EMBEDDINGS_DATASET = os.getenv('ARTEFACT_EMBEDDINGS_DATASET', 'samwaugh/artefact-embeddings')
ARTEFACT_MARKDOWN_DATASET = os.getenv('ARTEFACT_MARKDOWN_DATASET', 'samwaugh/artefact-markdown')

# Legacy path variables for backward compatibility
JSON_INFO_DIR = "/data/hub/datasets--samwaugh--artefact-json/snapshots/latest"
EMBEDDINGS_DIR = "/data/hub/datasets--samwaugh--artefact-embeddings/snapshots/latest"
MARKDOWN_DIR = "/data/hub/datasets--samwaugh--artefact-markdown/snapshots/latest"

# Embedding file paths for backward compatibility
CLIP_EMBEDDINGS_ST = Path(EMBEDDINGS_DIR) / "clip_embeddings.safetensors"
PAINTINGCLIP_EMBEDDINGS_ST = Path(EMBEDDINGS_DIR) / "paintingclip_embeddings.safetensors"
CLIP_SENTENCE_IDS = Path(EMBEDDINGS_DIR) / "clip_embeddings_sentence_ids.json"
PAINTINGCLIP_SENTENCE_IDS = Path(EMBEDDINGS_DIR) / "paintingclip_embeddings_sentence_ids.json"
CLIP_EMBEDDINGS_DIR = EMBEDDINGS_DIR
PAINTINGCLIP_EMBEDDINGS_DIR = EMBEDDINGS_DIR

# READ root (repo data - read-only)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_READ_ROOT = PROJECT_ROOT / "data"

# WRITE root (Space volume - writable)
# HF Spaces uses /data for persistent storage
WRITE_ROOT = Path(os.getenv("HF_HOME", "/data"))

# Check if the directory exists and is writable
if not WRITE_ROOT.exists():
    print(f"⚠️  WRITE_ROOT {WRITE_ROOT} does not exist, trying to create it")
    try:
        WRITE_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created WRITE_ROOT: {WRITE_ROOT}")
    except Exception as e:
        print(f"❌ Failed to create {WRITE_ROOT}: {e}")
        raise RuntimeError(f"Cannot create writable directory: {e}")

# Check write permissions
if not os.access(WRITE_ROOT, os.W_OK):
    print(f"❌ WRITE_ROOT {WRITE_ROOT} is not writable")
    print(f"❌ Current permissions: {oct(WRITE_ROOT.stat().st_mode)[-3:]}")
    print(f"❌ Owner: {WRITE_ROOT.owner()}")
    raise RuntimeError(f"Directory {WRITE_ROOT} is not writable")

print(f"✅ Using WRITE_ROOT: {WRITE_ROOT}")
print(f"✅ Using READ_ROOT: {DATA_READ_ROOT}")

# Read-only directories (from repo)
MODELS_DIR = DATA_READ_ROOT / "models"
MARKER_DIR = DATA_READ_ROOT / "marker_output"

# Model directories
PAINTINGCLIP_MODEL_DIR = MODELS_DIR / "PaintingClip"  # Note the capital C

# Writable directories (outside repo)
OUTPUTS_DIR = WRITE_ROOT / "outputs"
ARTIFACTS_DIR = WRITE_ROOT / "artifacts"

# Ensure writable directories exist
for dir_path in [OUTPUTS_DIR, ARTIFACTS_DIR]:
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Ensured directory exists: {dir_path}")
    except Exception as e:
        print(f"⚠️  Could not create directory {dir_path}: {e}")

# Global data variables (will be populated from HF datasets)
sentences: Dict[str, Any] = {}
works: Dict[str, Any] = {}
creators: Dict[str, Any] = {}
topics: Dict[str, Any] = {}
topic_names: Dict[str, Any] = {}

def load_json_from_hf(repo_id: str, filename: str) -> Optional[Dict[str, Any]]:
    """Load a single JSON file from Hugging Face repository"""
    if not HF_HUB_AVAILABLE:
        print(f"⚠️  huggingface_hub not available - cannot load {filename}")
        return None
        
    try:
        print(f"🔍 Downloading {filename} from {repo_id}...")
        file_path = hf_hub_download(
            repo_id=repo_id, 
            filename=filename, 
            repo_type="dataset"
        )
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Successfully loaded {filename}: {len(data)} entries")
        return data
    except Exception as e:
        print(f"❌ Failed to load {filename} from {repo_id}: {e}")
        return None

def load_json_datasets() -> Optional[Dict[str, Any]]:
    """Load all JSON datasets from Hugging Face"""
    if not HF_HUB_AVAILABLE:
        print("⚠️  huggingface_hub library not available - skipping HF dataset loading")
        return None
        
    try:
        print(" Loading JSON files from Hugging Face repository...")
        
        # Load individual JSON files
        global sentences, works, creators, topics, topic_names
        
        creators = load_json_from_hf(ARTEFACT_JSON_DATASET, 'creators.json') or {}
        sentences = load_json_from_hf(ARTEFACT_JSON_DATASET, 'sentences.json') or {}
        works = load_json_from_hf(ARTEFACT_JSON_DATASET, 'works.json') or {}
        topics = load_json_from_hf(ARTEFACT_JSON_DATASET, 'topics.json') or {}
        topic_names = load_json_from_hf(ARTEFACT_JSON_DATASET, 'topic_names.json') or {}
        
        print(f"✅ Successfully loaded JSON files from HF:")
        print(f"   Sentences: {len(sentences)} entries")
        print(f"   Works: {len(works)} entries")
        print(f"   Creators: {len(creators)} entries")
        print(f"   Topics: {len(topics)} entries")
        print(f"   Topic Names: {len(topic_names)} entries")
        
        return {
            'creators': creators,
            'sentences': sentences,
            'works': works,
            'topics': topics,
            'topic_names': topic_names
        }
    except Exception as e:
        print(f"❌ Failed to load JSON datasets from HF: {e}")
        return None

def load_embeddings_datasets() -> Optional[Dict[str, Any]]:
    """Load embeddings datasets from Hugging Face using direct file download"""
    if not HF_HUB_AVAILABLE:
        print("⚠️  huggingface_hub library not available - skipping HF embeddings loading")
        return None
        
    try:
        print(f" Loading embeddings from {ARTEFACT_EMBEDDINGS_DATASET}...")
        
        # Return a flag indicating we should use direct file download
        # The actual loading will be done in inference.py
        return {
            'use_direct_download': True,
            'repo_id': ARTEFACT_EMBEDDINGS_DATASET
        }
    except Exception as e:
        print(f"❌ Failed to load embeddings datasets from HF: {e}")
        return None

# Initialize datasets
JSON_DATASETS = load_json_datasets()
EMBEDDINGS_DATASETS = load_embeddings_datasets()

# Initialize data loading
if JSON_DATASETS is None:
    print("⚠️  Some data failed to load from HF datasets")
else:
    print("✅ All data loaded successfully from HF datasets")

# Add this function for backward compatibility
def st_load_file(file_path: Path) -> Any:
    """Load a file using safetensors or other methods"""
    try:
        if file_path.suffix == '.safetensors':
            import safetensors
            return safetensors.safe_open(str(file_path), framework="pt")
        else:
            import torch
            return torch.load(str(file_path))
    except ImportError:
        print(f"⚠️  Required library not available for loading {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None