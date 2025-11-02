import os
from glob import glob
import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from sklearn.metrics import accuracy_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
BASE_PATH = "./IndicAccentDS"
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-2
NUM_WORKERS = 0
AUG_PROB = 0.5

HUBERT_MODEL_NAME = "facebook/hubert-base-ls960"

# Load feature extractor and HuBERT model with safetensors enabled
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_MODEL_NAME)
hubert_model = HubertModel.from_pretrained(HUBERT_MODEL_NAME, use_safetensors=True).to(DEVICE)
hubert_model.eval()
for param in hubert_model.parameters():
    param.requires_grad = False  # Freeze HuBERT weights

# Dataset class
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

        feats = None
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

def collate_fn(batch):
    feats, labels = zip(*batch)
    max_len = max(f.size(0) for f in feats)
    input_dim = feats[0].size(1)
    padded = torch.zeros(len(feats), max_len, input_dim)
    for i, f in enumerate(feats):
        padded[i, : f.size(0)] = f
    return padded, torch.stack(labels)

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

def create_weighted_sampler(labels):
    labels_np = np.array(labels)
    class_counts = np.bincount(labels_np)
    class_counts[class_counts == 0] = 1  # avoid div by zero
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels_np]
    sample_weights = torch.from_numpy(sample_weights).float()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler

# Classifiers
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=256, layers=2, dropout=0.3):
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

class CNNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_filters=128, kernel_size=5, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, num_filters, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return self.fc(x)

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.3):
        super().__init__()
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead,
                                                    dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.transpose(0, 1)  # seq_len, batch, feature
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # mean over sequence
        return self.fc(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # shape (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: seq_len, batch_size, d_model
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

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

def l1_regularization(model):
    l1_norm = 0.0
    for name, param in model.named_parameters():
        # Apply L1 only on weights of linear and conv layers, skip biases and norm layers
        if param.requires_grad and ("weight" in name) and (("lstm" in name) or ("fc" in name) or ("conv" in name)):
            l1_norm += param.abs().sum()
    return l1_norm

def train_model(model, optimizer, train_loader, val_loader, epochs, checkpoint_path, l1_lambda=0.0):
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    best_acc = 0

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)

            if l1_lambda > 0:
                l1_norm = l1_regularization(model)
                loss = loss + l1_lambda * l1_norm

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
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model at epoch {epoch} with accuracy {best_acc:.4f}")

    return best_acc

def main():
    # Load data paths and labels
    train_files, train_labels, num_classes, label_map = load_data(BASE_PATH, "train")
    val_files, val_labels, _, _ = load_data(BASE_PATH, "val")

    # Choose which hidden layer to extract features from (e.g., 12 is good for HuBERT base)
    LAYER_IDX = 12

    # Prepare datasets and loaders
    train_dataset = AccentDataset(train_files, train_labels, "train", LAYER_IDX, augment=True)
    val_dataset = AccentDataset(val_files, val_labels, "val", LAYER_IDX, augment=False)

    train_sampler = create_weighted_sampler(train_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=collate_fn)

    # Choose your model architecture here
    input_dim = 768  # HuBERT base hidden size
    model = BiLSTMClassifier(input_dim, num_classes).to(DEVICE)
    # model = CNNClassifier(input_dim, num_classes).to(DEVICE)
    # model = TransformerClassifier(input_dim, num_classes).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Train with selective L1 regularization lambda (0 disables it)
    best_acc = train_model(model, optimizer, train_loader, val_loader, EPOCHS, "best_model.pth", l1_lambda=1e-5)

    print(f"Training finished. Best validation accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
