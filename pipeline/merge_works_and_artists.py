#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

def merge_works(shards_dir: Path) -> dict:
    merged = {}
    for jf in sorted(shards_dir.glob("shard_*/works_shard_*.json")):
        try:
            merged.update(json.loads(jf.read_text()))
        except Exception:
            pass
    return merged

def build_artists(artist_json_dir: Path, final_work_ids: set[str]) -> dict:
    artists = {}
    for f in sorted(artist_json_dir.glob("*.json")):
        name = f.stem
        try:
            arr = json.loads(f.read_text())
        except Exception:
            continue
        hits = []
        for w in arr:
            wid = str(w.get("id","")).rsplit("/",1)[-1].strip()
            if wid in final_work_ids:
                hits.append(wid)
        if hits:
            artists[name] = hits
    return artists

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--artist-json-dir", required=True)
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    shards_dir = run_root / "shards"

    works = merge_works(shards_dir)
    (run_root / "works.json").write_text(json.dumps(works, indent=2))
    print("works.json:", len(works), "entries")

    artists = build_artists(Path(args.artist_json_dir), set(works.keys()))
    (run_root / "artists.json").write_text(json.dumps(artists, indent=2))
    print("artists.json:", len(artists), "artists")

if __name__ == "__main__":
    main()
