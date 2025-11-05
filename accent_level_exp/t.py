import whisper
import os
import json
from tqdm import tqdm

DATASET_DIR = "IndicAccentDB_16k"
OUTPUT_DIR = "transcripts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = whisper.load_model("base")  # use "medium" if GPU allows

splits = ["train", "val", "test"]
results = {}

for split in splits:
    split_results = {}
    split_dir = os.path.join(DATASET_DIR, split)
    for accent in os.listdir(split_dir):
        accent_dir = os.path.join(split_dir, accent)
        if not os.path.isdir(accent_dir):
            continue

        for file in tqdm(os.listdir(accent_dir), desc=f"{split}/{accent}"):
            if not file.endswith(".wav"):
                continue
            path = os.path.join(accent_dir, file)
            result = model.transcribe(path, word_timestamps=True)
            split_results[file] = {
                "path": path,
                "accent": accent,
                "words": [
                    {"word": w["word"], "start": w["start"], "end": w["end"]}
                    for seg in result["segments"]
                    for w in seg.get("words", [])
                ]
            }

    with open(os.path.join(OUTPUT_DIR, f"{split}_transcripts.json"), "w") as f:
        json.dump(split_results, f, indent=2)
