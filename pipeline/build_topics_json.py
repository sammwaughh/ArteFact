#!/usr/bin/env python3
"""
build_topics_json.py
--------------------
Create topics.json – a reverse index of TopicID → [WorkIDs …] using the
data already stored in works.json from the Bede RUN_ROOT.

Usage
-----
$ python pipeline/build_topics_json.py
$ sbatch pipeline/run_build_topics.sbatch
"""

from __future__ import annotations

import json
import sys
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

# Get RUN_ROOT from environment variable, fallback to current directory
RUN_ROOT = Path(os.getenv("RUN_ROOT", ".")).resolve()
WORKS_FILE = RUN_ROOT / "works.json"
TOPICS_FILE = RUN_ROOT / "topics.json"

print(f"�� Working directory: {Path.cwd()}")
print(f"📁 RUN_ROOT: {RUN_ROOT}")
print(f"�� Works file: {WORKS_FILE}")
print(f"📄 Topics file: {TOPICS_FILE}")


def build_topics() -> Dict[str, List[str]]:
    if not WORKS_FILE.exists():
        sys.exit(f"❌ {WORKS_FILE} not found")
    
    print(f"📖 Loading {WORKS_FILE}...")
    works: Dict[str, Dict] = json.loads(WORKS_FILE.read_text())
    print(f"�� Found {len(works):,} works")
    
    topics: dict[str, list[str]] = defaultdict(list)
    topic_count = 0
    
    # single linear pass – O(total topic refs)
    for work_id, meta in works.items():
        for topic_id in meta.get("TopicIDs", []):
            topics[topic_id].append(work_id)
            topic_count += 1
    
    print(f"�� Found {topic_count:,} topic references across {len(topics):,} unique topics")
    
    # (optional) determinism: sort the lists
    for lst in topics.values():
        lst.sort()
    
    return dict(topics)


def main() -> None:
    print(f"🚀 Starting topic index building...")
    
    topics_dict = build_topics()
    
    print(f"�� Saving topics to {TOPICS_FILE}...")
    TOPICS_FILE.write_text(json.dumps(topics_dict, indent=2, ensure_ascii=False))
    
    print(f"✅ {len(topics_dict):,} topics written to {TOPICS_FILE.name}")
    print(f"�� File size: {TOPICS_FILE.stat().st_size / (1024**2):.1f} MB")


if __name__ == "__main__":
    main()