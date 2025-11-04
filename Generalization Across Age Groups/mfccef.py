import os
import librosa
import torch
import numpy as np
from tqdm import tqdm

def extract_mfcc(root_dir, save_dir, n_mfcc=40, sr=16000):
    os.makedirs(save_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(root_dir, split)
        for accent in os.listdir(split_dir):
            accent_dir = os.path.join(split_dir, accent)
            out_dir = os.path.join(save_dir, split, accent)
            os.makedirs(out_dir, exist_ok=True)

            for file in tqdm(os.listdir(accent_dir), desc=f"MFCC {split}/{accent}"):
                if not file.endswith('.wav'):
                    continue
                path = os.path.join(accent_dir, file)
                y, _ = librosa.load(path, sr=sr)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                feat = torch.tensor(np.mean(mfcc, axis=1), dtype=torch.float32)
                torch.save(feat, os.path.join(out_dir, file.replace(".wav", ".pt")))

# Extract for adult and child sets
extract_mfcc("IndicAccentDB_age/adult", "features_mfcc/adult")
extract_mfcc("IndicAccentDB_age/child", "features_mfcc/child")
