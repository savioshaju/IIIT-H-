#!/usr/bin/env python3
"""
generate_metadata.py
Scan a dataset split directory and produce metadata.csv with audio file info
Usage: python generate_metadata.py --base-dir IndicAccentDB_split --out metadata.csv
"""

import argparse
from pathlib import Path
import soundfile as sf
import pandas as pd
import hashlib
from tqdm import tqdm
import os

DEFAULT_BASE = "IndicAccentDB_split"  # change if your split lives elsewhere

# Default mapping: accent folder -> L1 language label
DEFAULT_LANG_MAP = {
    "andhra_pradesh": "telugu",
    "gujrat": "gujarati",
    "jharkhand": "hindi",
    "karnataka": "kannada",
    "kerala": "malayalam",
    "tamil": "tamil",
}

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}


def sha1_of_file(path: Path, block_size: int = 65536) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def scan(base_dir: Path, lang_map: dict):
    rows = []
    # If base_dir contains train/val/test use that; otherwise treat base_dir as single split 'raw'
    splits = []
    for name in ["train", "val", "test"]:
        if (base_dir / name).exists():
            splits.append(name)
    if not splits:
        # fallback: single folder with accents directly
        splits = [None]

    for split in splits:
        if split is None:
            root = base_dir
            split_label = "raw"
        else:
            root = base_dir / split
            split_label = split

        if not root.exists():
            print(f"Warning: {root} does not exist. Skipping.")
            continue

        for accent_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            accent = accent_dir.name
            lang_label = lang_map.get(accent, "unknown")
            # find audio files in this accent folder
            files = [p for p in accent_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS and p.is_file()]
            for f in files:
                try:
                    info = sf.info(str(f))
                    duration = info.duration
                    samplerate = info.samplerate
                    channels = info.channels
                except Exception:
                    # fallback: zero or try os.path.getsize
                    duration = None
                    samplerate = None
                    channels = None

                filesize = f.stat().st_size
                file_sha1 = sha1_of_file(f)

                rows.append({
                    "id": file_sha1,
                    "file_path": str(f.resolve()),
                    "split": split_label,
                    "accent": accent,
                    "language_label": lang_label,
                    "duration_s": duration,
                    "sample_rate": samplerate,
                    "channels": channels,
                    "bytes": filesize
                })

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE,
                        help="Base dataset directory containing train/val/test subfolders (default: %(default)s)")
    parser.add_argument("--out", type=str, default="metadata.csv", help="Output CSV file (default: metadata.csv)")
    parser.add_argument("--lang-map", type=str, default=None,
                        help="Optional path to a JSON file with accent->language mapping")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise SystemExit(f"Base dir not found: {base_dir}")

    # load custom mapping if provided
    lang_map = DEFAULT_LANG_MAP.copy()
    if args.lang_map:
        import json
        m = json.loads(Path(args.lang_map).read_text())
        lang_map.update(m)

    print(f"Scanning base dir: {base_dir}")
    df = scan(base_dir, lang_map)

    if df.empty:
        print("No audio files found. Exiting.")
        return

    df.to_csv(args.out, index=False)
    print(f"Saved metadata: {args.out}")
    # print quick stats
    print("\nPer-split counts:")
    print(df["split"].value_counts())
    print("\nPer-language counts:")
    print(df["language_label"].value_counts())

    # optionally save a small sample
    sample_csv = Path(args.out).stem + "_sample.csv"
    df.sample(min(20, len(df))).to_csv(sample_csv, index=False)
    print(f"Saved sample: {sample_csv}")


if __name__ == "__main__":
    main()
