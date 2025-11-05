import os
import re
import json
import soundfile as sf
import unicodedata
from tqdm import tqdm
from pathlib import Path

DATASET_DIR = "IndicAccentDB_16k"
TRANSCRIPT_DIR = "transcripts"
OUTPUT_DIR = "IndicAccentDB_16k_segmented"
ACCENTS = ["andhra_pradesh", "gujarat", "jharkhand", "karnataka", "kerala", "tamil_nadu"]


def safe_path(path):
    """Handle Windows long paths."""
    abs_path = os.path.abspath(path)
    if os.name == "nt" and not abs_path.startswith("\\\\?\\"):
        abs_path = "\\\\?\\" + abs_path
    return abs_path

def clean_filename(name: str) -> str:
    """Normalize and clean file names for safe saving."""
    name = unicodedata.normalize('NFKD', name)
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'[^\x00-\x7F]+', '', name)
    name = re.sub(r'\s+', '_', name)
    return name.strip('_')[:80] or "unknown"

def safe_write_wav(output_path, data, sr):
    """Write WAV safely with directory creation."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_path = safe_path(output_path)
        with sf.SoundFile(output_path, 'w', samplerate=sr, channels=len(data.shape) if data.ndim > 1 else 1) as f:
            f.write(data)
    except Exception as e:
        print(f"[WRITE ERROR] {output_path}: {e}")

for level in ["word", "sentence"]:
    for split in ["train", "val", "test"]:
        for accent in ACCENTS:
            path = Path(OUTPUT_DIR) / level / split / accent
            os.makedirs(path, exist_ok=True)

for split in ["train", "val", "test"]:
    transcript_path = Path(TRANSCRIPT_DIR) / f"{split}_transcripts.json"
    if not transcript_path.exists():
        print(f"[SKIP] No transcript file for {split}")
        continue

    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\nProcessing {split} split ...")
    for file_name, info in tqdm(data.items(), desc=f"{split}"):
        accent = info.get("accent", "unknown")
        audio_path = info.get("path")

        if not os.path.exists(audio_path):
            print(f"[MISSING] {audio_path}")
            continue

        try:
            audio, sr = sf.read(audio_path)
        except Exception as e:
            print(f"[READ ERROR] {audio_path}: {e}")
            continue

        clean_base = clean_filename(Path(file_name).stem)
        sentence_out = Path(OUTPUT_DIR) / "sentence" / split / accent / f"{clean_base}.wav"
        safe_write_wav(sentence_out, audio, sr)

        for i, w in enumerate(info.get("words", [])):
            start, end = w.get("start", 0), w.get("end", 0)
            if end <= start:
                continue

            word_audio = audio[int(start * sr):int(end * sr)]
            if len(word_audio) < 1000:  
                continue

            clean_word = clean_filename(w.get("word", "unknown"))
            word_name = f"{clean_base}_w{i}_{clean_word}.wav"
            word_out = Path(OUTPUT_DIR) / "word" / split / accent / word_name
            safe_write_wav(word_out, word_audio, sr)

print("\nWord-level and sentence-level datasets generated successfully!")
