import os
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from tqdm import tqdm

DATA_DIR = "IndicAccentDB_16k"   
OUT_DIR = "hubert_features"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

LABELS = {
    "andhra_pradesh": 0,
    "gujarat": 1,
    "jharkhand": 2,
    "karnataka": 3,
    "kerala": 4,
    "tamil": 5
}

extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(DEVICE)
model.eval()

for split in ["train", "val", "test"]:
    split_dir = os.path.join(DATA_DIR, split)
    out_split = os.path.join(OUT_DIR, split)
    os.makedirs(out_split, exist_ok=True)

    all_feats, all_labels = [], []

    for accent in LABELS.keys():
        folder = os.path.join(split_dir, accent)
        if not os.path.isdir(folder):
            continue

        files = [f for f in os.listdir(folder) if f.endswith(".wav")]
        for fname in tqdm(files, desc=f"{split}-{accent}"):
            path = os.path.join(folder, fname)
            wav, sr = torchaudio.load(path)
            inputs = extractor(wav.squeeze(0), sampling_rate=sr, return_tensors="pt", padding=True)
            with torch.no_grad():
                feats = model(**inputs.to(DEVICE)).last_hidden_state.mean(dim=1).cpu()
            all_feats.append(feats)
            all_labels.append(torch.tensor(LABELS[accent]))

    all_feats = torch.cat(all_feats)
    all_labels = torch.stack(all_labels)
    torch.save(all_feats, os.path.join(out_split, f"{split}_hubert.pt"))
    torch.save(all_labels, os.path.join(out_split, f"{split}_labels.pt"))

print("✅ HuBERT feature extraction complete. Files saved to:", OUT_DIR)
