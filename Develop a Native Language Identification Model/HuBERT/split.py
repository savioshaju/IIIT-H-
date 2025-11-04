import os, shutil, random, hashlib
from tqdm import tqdm

# CONFIG
SRC_DIR = "IndicAccentDB"  # root folder with all accents before split
OUT_DIR = "IndicAccentDB_clean_split"
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
ACCENTS = ["andhra_pradesh", "gujarat", "jharkhand", "karnataka", "kerala", "tamil"]

def hash_audio(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def safe_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)

# MAIN
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR)

global_hashes = set()

for accent in ACCENTS:
    accent_path = os.path.join(SRC_DIR, accent)
    files = [os.path.join(accent_path, f) for f in os.listdir(accent_path) if f.endswith(".wav")]

    # Remove true duplicates inside same accent
    unique_files = []
    for f in files:
        h = hash_audio(f)
        if h not in global_hashes:
            unique_files.append(f)
            global_hashes.add(h)

    random.shuffle(unique_files)
    n = len(unique_files)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    splits = {
        "train": unique_files[:n_train],
        "val": unique_files[n_train:n_train+n_val],
        "test": unique_files[n_train+n_val:]
    }

    for split_name, split_files in splits.items():
        for f in split_files:
            dst = os.path.join(OUT_DIR, split_name, accent, os.path.basename(f))
            safe_copy(f, dst)

print("✅ Clean dataset split completed at:", OUT_DIR)
