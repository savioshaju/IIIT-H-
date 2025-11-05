import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import os
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
model.eval()

def extract_hubert(root_dir, save_dir, sr=16000):
    os.makedirs(save_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(root_dir, split)
        for accent in os.listdir(split_dir):
            accent_dir = os.path.join(split_dir, accent)
            out_dir = os.path.join(save_dir, split, accent)
            os.makedirs(out_dir, exist_ok=True)

            for file in tqdm(os.listdir(accent_dir), desc=f"HuBERT {split}/{accent}"):
                if not file.endswith('.wav'):
                    continue
                path = os.path.join(accent_dir, file)
                wav, sr_ = torchaudio.load(path)
                wav = torchaudio.functional.resample(wav, sr_, sr).mean(0)

                inputs = processor(wav.numpy(), sampling_rate=sr, return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs = model(inputs.input_values.to(device))
                hidden_states = outputs.last_hidden_state.squeeze(0)
                feat = torch.mean(hidden_states, dim=0).cpu()
                torch.save(feat, os.path.join(out_dir, file.replace(".wav", ".pt")))

extract_hubert("IndicAccentDB_age/adult", "feature_hubert/adult")
extract_hubert("IndicAccentDB_age/child", "feature_hubert/child")
