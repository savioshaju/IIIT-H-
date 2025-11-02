import torch
import torchaudio
from torch import nn
from transformers import Wav2Vec2FeatureExtractor, HubertModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BiLSTM Classifier (must match your trained model)
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

# Load pretrained HuBERT and feature extractor (freeze weights)
HUBERT_MODEL_NAME = "facebook/hubert-base-ls960"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_MODEL_NAME)
hubert_model = HubertModel.from_pretrained(HUBERT_MODEL_NAME).to(DEVICE)  # <-- Removed use_safetensors=True
hubert_model.eval()
for param in hubert_model.parameters():
    param.requires_grad = False

# Number of classes in your classification task
num_classes = 6  # Update this according to your training

# Initialize and load your trained BiLSTM model
input_dim = 768  # HuBERT base output feature dimension
model = BiLSTMClassifier(input_dim, num_classes).to(DEVICE)
model.load_state_dict(torch.load("Model/HuBERT/best_model.pth", map_location=DEVICE))
model.eval()

def extract_features(audio_path, layer_idx=12):
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    input_values = feature_extractor(
        waveform.squeeze().cpu().numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    )["input_values"].to(DEVICE)

    with torch.no_grad():
        outputs = hubert_model(input_values, output_hidden_states=True)
        feats = outputs.hidden_states[layer_idx].squeeze(0)  # (seq_len, feature_dim)
    return feats

def predict(audio_path):
    feats = extract_features(audio_path).unsqueeze(0)  # Add batch dimension: (1, seq_len, feature_dim)
    feats = feats.to(DEVICE)
    with torch.no_grad():
        logits = model(feats)
        predicted_class = logits.argmax(dim=1).item()
    return predicted_class

# Example usage
if __name__ == "__main__":
    audio_file = "IndicAccentDS/child/test/kerala/Kerala_speaker_03_malayalam_s01_005.wav"  # Change to your file path
    pred_class = predict(audio_file)
    print(f"Predicted class index: {pred_class}")







# import torch
# import numpy as np
# import sounddevice as sd
# import keyboard  # For keypress detection
# from torch import nn
# from transformers import Wav2Vec2FeatureExtractor, HubertModel
# import torch.nn.functional as F
# import threading
# import time

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class BiLSTMClassifier(nn.Module):
#     def __init__(self, input_dim, num_classes, hidden=256, layers=2, dropout=0.3):
#         super().__init__()
#         self.norm = nn.LayerNorm(input_dim)
#         self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True, bidirectional=True, dropout=dropout)
#         self.drop = nn.Dropout(dropout)
#         self.fc = nn.Linear(hidden * 2, num_classes)

#     def forward(self, x):
#         x = self.norm(x)
#         out, _ = self.lstm(x)
#         out = out.mean(dim=1)
#         return self.fc(self.drop(out))

# # Load models as before
# HUBERT_MODEL_NAME = "facebook/hubert-base-ls960"
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_MODEL_NAME)
# hubert_model = HubertModel.from_pretrained(HUBERT_MODEL_NAME).to(DEVICE)
# hubert_model.eval()
# for param in hubert_model.parameters():
#     param.requires_grad = False

# num_classes = 6
# input_dim = 768
# model = BiLSTMClassifier(input_dim, num_classes).to(DEVICE)
# model.load_state_dict(torch.load("Model/HuBERT/best_model.pth", map_location=DEVICE))
# model.eval()

# SAMPLE_RATE = 16000
# WINDOW_SECONDS = 5
# STEP_SECONDS = WINDOW_SECONDS // 2
# WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS
# STEP_SIZE = SAMPLE_RATE * STEP_SECONDS

# audio_buffer = np.zeros(0, dtype=np.float32)
# CLASS_LABELS = ["class_0", "class_1", "class_2", "class_3", "class_4", "class_5"]

# is_listening = False  # Global flag controlled by keyboard events
# predictions = []      # Store predictions during listening

# def normalize_audio(audio):
#     max_val = np.max(np.abs(audio))
#     if max_val > 0:
#         audio = audio / max_val
#     return audio

# def predict_from_audio(audio_np):
#     audio_np = normalize_audio(audio_np)
#     input_values = feature_extractor(audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt")["input_values"].to(DEVICE)
#     with torch.no_grad():
#         outputs = hubert_model(input_values, output_hidden_states=True)
#         feats = outputs.hidden_states[12].squeeze(0).unsqueeze(0)
#         logits = model(feats)
#         probs = F.softmax(logits, dim=1)
#         conf, pred_idx = torch.max(probs, dim=1)
#         return pred_idx.item(), conf.item()

# def audio_callback(indata, frames, time, status):
#     global audio_buffer, is_listening, predictions
#     if status:
#         print(f"Audio Status: {status}")
#     if not is_listening:
#         return  # Ignore audio if not listening

#     try:
#         audio_chunk = indata[:, 0].copy()  # mono channel
#         audio_buffer = np.concatenate((audio_buffer, audio_chunk))

#         while len(audio_buffer) >= WINDOW_SIZE:
#             window_audio = audio_buffer[:WINDOW_SIZE]
#             pred_idx, conf = predict_from_audio(window_audio)
#             label = CLASS_LABELS[pred_idx]
#             print(f"Predicted class: {label} with confidence: {conf:.2f}")
#             predictions.append((label, conf))
#             audio_buffer = audio_buffer[STEP_SIZE:]
#     except Exception as e:
#         print(f"Error in audio callback: {e}")

# def keyboard_listener():
#     global is_listening
#     print("Press 's' to START listening and 'q' to STOP and show results.")
#     while True:
#         if keyboard.is_pressed('s') and not is_listening:
#             print("\n*** Listening started ***")
#             is_listening = True
#             time.sleep(0.5)  # debounce
#         elif keyboard.is_pressed('q') and is_listening:
#             print("\n*** Listening stopped ***")
#             is_listening = False
#             break
#         time.sleep(0.1)

# def main():
#     listener_thread = threading.Thread(target=keyboard_listener, daemon=True)
#     listener_thread.start()

#     print("Ready for input...")
#     with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=STEP_SIZE, callback=audio_callback):
#         while listener_thread.is_alive():
#             sd.sleep(100)

#     # After stopping, summarize predictions
#     if predictions:
#         from collections import Counter
#         counts = Counter([p[0] for p in predictions])
#         most_common = counts.most_common(1)[0]
#         print(f"\nFinal prediction summary: '{most_common[0]}' occurred {most_common[1]} times out of {len(predictions)} windows.")
#     else:
#         print("No predictions were made.")

# if __name__ == "__main__":
#     main()
