import os
import librosa
import numpy as np
from tqdm import tqdm
import joblib  # for saving features efficiently

DATASET_DIR = "IndicAccentDB_16k_segmented"
OUTPUT_DIR = "IndicAccentDB_16k_features_mfcc"   # <-- new output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_mfcc(path, sr=16000, n_mfcc=40):
    try:
        y, sr = librosa.load(path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)  # average over time
        return mfcc_mean
    except Exception as e:
        print(f"[ERROR] {path}: {e}")
        return None

for level in ["word", "sentence"]:
    for split in ["train", "val", "test"]:
        features, labels = [], []
        split_dir = os.path.join(DATASET_DIR, level, split)
        if not os.path.exists(split_dir):
            continue

        for accent in os.listdir(split_dir):
            accent_dir = os.path.join(split_dir, accent)
            if not os.path.isdir(accent_dir):
                continue

            for file in tqdm(os.listdir(accent_dir), desc=f"{level}-{split}-{accent}"):
                if not file.endswith(".wav"):
                    continue
                path = os.path.join(accent_dir, file)
                feat = extract_mfcc(path)
                if feat is not None:
                    features.append(feat)
                    labels.append(accent)

        save_dir = os.path.join(OUTPUT_DIR, level)
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(
            {"features": np.array(features), "labels": np.array(labels)},
            os.path.join(save_dir, f"{split}.pkl")
        )

print("\n MFCC features extracted and saved under:", OUTPUT_DIR)
