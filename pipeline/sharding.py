# sharding.py
from __future__ import annotations
import hashlib, json, os, threading, tempfile
from pathlib import Path

SHARDS = int(os.getenv("OA_SHARDS", "32"))
RUN_ROOT = Path(os.getenv("RUN_ROOT", Path.cwd()))
SHARDS_DIR = RUN_ROOT / "shards"

_locks = [threading.Lock() for _ in range(SHARDS)]

def shard_index(work_id: str, n: int = SHARDS) -> int:
    # stable, uniform index for a given work id
    h = hashlib.md5(work_id.encode("utf-8")).hexdigest()
    return int(h, 16) % n

def shard_dir(idx: int) -> Path:
    return SHARDS_DIR / f"shard_{idx:02d}"

def works_path(idx: int) -> Path:
    return shard_dir(idx) / f"works_shard_{idx:02d}.json"

def pdf_bucket(idx: int) -> Path:
    return shard_dir(idx) / "PDF_Bucket"

def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"{path}.tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)  # atomic on same filesystem

def locate_pdf_bucket(work_id: str) -> Path:
    idx = shard_index(work_id)
    b = pdf_bucket(idx)
    b.mkdir(parents=True, exist_ok=True)
    return b

def upsert_work_record(work_id: str, entry: dict) -> None:
    idx = shard_index(work_id)
    p = works_path(idx)
    with _locks[idx]:
        try:
            db = json.loads(p.read_text())
        except FileNotFoundError:
            db = {}
        db[work_id] = entry
        _atomic_write_json(p, db)
