#!/usr/bin/env python3
"""
build_topics_json.py
--------------------
Create topics.json – a reverse index of TopicID → [WorkIDs …] using the
data already stored in works.json.

Usage
-----
$ python Pipeline/build_topics_json.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent
WORKS_FILE = ROOT / "works.json"
TOPICS_FILE = ROOT / "topics.json"


def build_topics() -> Dict[str, List[str]]:
    if not WORKS_FILE.exists():
        sys.exit(f"❌ {WORKS_FILE} not found")

    works: Dict[str, Dict] = json.loads(WORKS_FILE.read_text())
    topics: dict[str, list[str]] = defaultdict(list)

    # single linear pass – O(total topic refs)
    for work_id, meta in works.items():
        for topic_id in meta.get("TopicIDs", []):
            topics[topic_id].append(work_id)

    # (optional) determinism: sort the lists
    for lst in topics.values():
        lst.sort()

    return dict(topics)


def main() -> None:
    topics_dict = build_topics()
    TOPICS_FILE.write_text(json.dumps(topics_dict, indent=2, ensure_ascii=False))
    print(f"✅ {len(topics_dict):,} topics written to {TOPICS_FILE.name}")


if __name__ == "__main__":
    main()
