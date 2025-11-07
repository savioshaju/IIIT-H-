import os
import torch
import pandas as pd
import torchaudio
from tqdm import tqdm

# === Paths ===
BASE_DIR = "IndicAccentDB_clean_split"
META_FILE = "metadata.csv"
OUT_DIR = "mfcc_features"
os.makedirs(OUT_DIR, exist_ok=True)

# === Load metadata ===
meta = pd.read_csv(META_FILE)

# === Parameters ===
n_mfcc = 40
sr_target = 16000
max_len = 16000  # ~1 sec window; increase if clips are longer

# === Label encoding ===
label2idx = {label: idx for idx, label in enumerate(sorted(meta["language_label"].unique()))}
print("Label mapping:", label2idx)

# === MFCC extractor ===
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=sr_target,
    n_mfcc=n_mfcc,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
)

# === Feature extraction ===
def extract_features(df, split_name):
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {split_name}"):
        try:
            file_path = row["file_path"]
            if not os.path.exists(file_path):
                print(f"⚠️ Missing file: {file_path}")
                continue

            wav, sr = torchaudio.load(file_path)
            wav = wav.mean(dim=0, keepdim=True)  # convert to mono

            # Resample if needed
            if sr != sr_target:
                wav = torchaudio.functional.resample(wav, sr, sr_target)

            # Pad or trim
            if wav.shape[1] < max_len:
                wav = torch.nn.functional.pad(wav, (0, max_len - wav.shape[1]))
            else:
                wav = wav[:, :max_len]

            # Compute MFCCs
            mfcc = mfcc_transform(wav)
            X.append(mfcc.squeeze(0))
            y.append(label2idx[row["language_label"]])

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    # Stack & save tensors
    X = torch.stack(X)
    y = torch.tensor(y)
    torch.save(X, os.path.join(OUT_DIR, f"{split_name}_mfcc.pt"))
    torch.save(y, os.path.join(OUT_DIR, f"{split_name}_labels.pt"))
    print(f"✅ Saved {split_name}: {X.shape}, {y.shape}")


# === Run for all splits ===
for split in ["train", "val", "test"]:
    df_split = meta[meta["split"] == split]
    extract_features(df_split, split)
