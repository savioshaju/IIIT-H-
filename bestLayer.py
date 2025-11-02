import os
from glob import glob
import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from sklearn.metrics import accuracy_score
import random
import csv

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths and constants
BASE_PATH = "./IndicAccentDS"
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 10  # Tune this according to your compute resources
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-2
NUM_WORKERS = 0
AUG_PROB = 0.5
EARLY_STOPPING_PATIENCE = 5

# Load pretrained HuBERT model and feature extractor
HUBERT_MODEL_NAME = "facebook/hubert-base-ls960"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_MODEL_NAME)
hubert_model = HubertModel.from_pretrained(HUBERT_MODEL_NAME).to(DEVICE)
hubert_model.eval()
for param in hubert_model.parameters():
    param.requires_grad = False

# Reproducibility seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

# Dataset for accent classification, caches features per layer
class AccentDataset(Dataset):
    def __init__(self, files, labels, split, layer_idx, augment=False):
        self.files = files
        self.labels = labels
        self.split = split
        self.layer_idx = layer_idx
        self.augment = augment
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav = self.files[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        cache_path = os.path.join(CACHE_DIR, f"{self.split}_layer{self.layer_idx}_{os.path.basename(wav).replace('.wav','.pt')}")

        if os.path.exists(cache_path):
            feats = torch.load(cache_path, map_location="cpu")
        else:
            waveform, sr = torchaudio.load(wav)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            if self.augment and torch.rand(1).item() < AUG_PROB:
                waveform = self.freq_mask(waveform)
                waveform = self.time_mask(waveform)
            input_values = feature_extractor(waveform.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt")["input_values"].to(DEVICE)
            with torch.no_grad():
                out = hubert_model(input_values, output_hidden_states=True)
                feats = out.hidden_states[self.layer_idx].squeeze(0).cpu()
            torch.save(feats, cache_path)
        return feats, label

# Collate function to pad variable-length sequences
def collate_fn(batch):
    feats, labels = zip(*batch)
    max_len = max(f.size(0) for f in feats)
    input_dim = feats[0].size(1)
    padded = torch.zeros(len(feats), max_len, input_dim)
    for i, f in enumerate(feats):
        padded[i, : f.size(0)] = f
    return padded, torch.stack(labels)

# Load dataset files and labels, creating label map dynamically
def load_data(base_path, split):
    groups = ["adult", "child"]
    all_files, all_labels = [], []
    label_map = {}
    idx = 0
    for group in groups:
        grp_path = os.path.join(base_path, group, split)
        if not os.path.exists(grp_path):
            continue
        accents = sorted([d for d in os.listdir(grp_path) if os.path.isdir(os.path.join(grp_path, d))])
        for acc in accents:
            if acc not in label_map:
                label_map[acc] = idx
                idx += 1
            files = glob(os.path.join(grp_path, acc, "*.wav"))
            all_files.extend(files)
            all_labels.extend([label_map[acc]] * len(files))
    print(f"[{split}] Loaded {len(all_files)} files across {len(label_map)} accents.")
    return all_files, all_labels, len(label_map), label_map

# Create weighted sampler to handle class imbalance
def create_weighted_sampler(labels):
    labels_np = np.array(labels)
    class_counts = np.bincount(labels_np)
    class_counts[class_counts == 0] = 1
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels_np]
    sample_weights = torch.from_numpy(sample_weights).float()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler

# BiLSTM classifier with LayerNorm and Dropout
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=256, layers=2, dropout=0.5):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        x = self.norm(x)
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        return self.fc(self.drop(out))

# Evaluation on validation set
def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            preds.append(out.argmax(dim=1).cpu())
            labels.append(y.cpu())
    if len(labels) == 0:
        return 0.0
    return accuracy_score(torch.cat(labels), torch.cat(preds))

# Training loop with early stopping and scheduler
def train_model(model, optimizer, train_loader, val_loader, epochs, early_stopping_patience=EARLY_STOPPING_PATIENCE):
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    best_acc = 0
    epochs_no_improve = 0

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        scheduler.step(val_acc)

        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered.")
                break
    return best_acc

def main():
    train_files, train_labels, num_classes, label_map = load_data(BASE_PATH, "train")
    val_files, val_labels, _, _ = load_data(BASE_PATH, "val")

    input_dim = 768  # HuBERT base hidden size

    best_layer = None
    best_accuracy = 0.0
    results = []

    for layer_idx in range(13):  # Layers 0 to 12
        print(f"\n--- Evaluating HuBERT Layer {layer_idx} ---")

        train_dataset = AccentDataset(train_files, train_labels, "train", layer_idx, augment=True)
        val_dataset = AccentDataset(val_files, val_labels, "val", layer_idx, augment=False)

        train_sampler = create_weighted_sampler(train_labels)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                                  collate_fn=collate_fn, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=collate_fn, num_workers=NUM_WORKERS)

        model = BiLSTMClassifier(input_dim, num_classes).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        val_acc = train_model(model, optimizer, train_loader, val_loader, EPOCHS)
        print(f"Layer {layer_idx} Validation Accuracy: {val_acc:.4f}")

        results.append((layer_idx, val_acc))

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_layer = layer_idx

    print(f"\nBest layer: {best_layer} with Validation Accuracy: {best_accuracy:.4f}")

    # Save results to CSV
    with open("hubert_layer_results.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Layer", "Validation Accuracy"])
        writer.writerows(results)

if __name__ == "__main__":
    main()
