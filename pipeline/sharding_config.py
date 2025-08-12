# sharding_config.py
"""
Configuration and utilities for the sharding system.
"""

import os
from pathlib import Path

# Sharding configuration
DEFAULT_SHARDS = 32
DEFAULT_RUN_ROOT = Path.cwd()

# Environment variables
SHARDS = int(os.getenv("OA_SHARDS", str(DEFAULT_SHARDS)))
RUN_ROOT = Path(os.getenv("RUN_ROOT", str(DEFAULT_RUN_ROOT)))

# Derived paths
SHARDS_DIR = RUN_ROOT / "shards"
ARTISTS_JSON_DIR = RUN_ROOT / "Artist-JSONs"

# Validation
def validate_config():
    """Validate sharding configuration and create necessary directories."""
    if SHARDS <= 0:
        raise ValueError(f"Invalid number of shards: {SHARDS}")
    
    if not RUN_ROOT.exists():
        raise ValueError(f"Run root directory does not exist: {RUN_ROOT}")
    
    # Create shards directory structure
    SHARDS_DIR.mkdir(exist_ok=True)
    for i in range(SHARDS):
        shard_dir = SHARDS_DIR / f"shard_{i:02d}"
        shard_dir.mkdir(exist_ok=True)
        (shard_dir / "PDF_Bucket").mkdir(exist_ok=True)
    
    return True

# Convenience functions
def get_shard_info():
    """Return information about current sharding setup."""
    return {
        "shards": SHARDS,
        "run_root": str(RUN_ROOT),
        "shards_dir": str(SHARDS_DIR),
        "artists_dir": str(ARTISTS_JSON_DIR)
    }