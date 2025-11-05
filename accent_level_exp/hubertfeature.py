import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import joblib

DATASET_DIR = "IndicAccentDB_16k_segmented"
OUTPUT_DIR = "IndicAccentDB_16k_features_hubert"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(DEVICE)
model.eval()

def extract_hubert_features(path, sr=16000):
    try:
        y, _ = librosa.load(path, sr=sr)
        inputs = extractor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(inputs.input_values.to(DEVICE))
        hidden_states = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        feat_mean = np.mean(hidden_states, axis=0)
        return feat_mean
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
                feat = extract_hubert_features(path)
                if feat is not None:
                    features.append(feat)
                    labels.append(accent)

        save_dir = os.path.join(OUTPUT_DIR, level)
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(
            {"features": np.array(features), "labels": np.array(labels)},
            os.path.join(save_dir, f"{split}.pkl")
        )

print("\n✅ HuBERT features extracted and saved under:", OUTPUT_DIR)
