import os
import tempfile
import platform
import subprocess
import torch
import torchaudio
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, send_from_directory
from model_def import DANNModel

app = Flask(__name__, static_folder="static", template_folder="templates")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_STATE_PATH = "model/accent_dann_model.pt"
N_CLASSES = 6
ACCENT_LABELS = ["andhra_pradesh", "gujarat", "jharkhand", "karnataka", "kerala", "tamil"]
model = None
HUBERT = None

def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_STATE_PATH):
            raise FileNotFoundError(f"Model checkpoint not found: {MODEL_STATE_PATH}")
        model = DANNModel(input_dim=768, hidden_dim=512, num_classes=N_CLASSES, n_domains=4)
        state = torch.load(MODEL_STATE_PATH, map_location=DEVICE, weights_only=True)
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)
        model = model.to(DEVICE).float()
        model.eval()
    return model

def get_hubert():
    global HUBERT
    if HUBERT is None:
        HUBERT = torchaudio.pipelines.HUBERT_BASE.get_model().to(DEVICE).float()
        HUBERT.eval()
    return HUBERT

def preprocess_and_extract_features(wav_path):
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform.float().to(DEVICE)
    waveform = waveform / waveform.abs().max().clamp(min=1e-9)
    waveform = waveform - waveform.mean()
    if waveform.size(1) < 16000:
        raise ValueError("Audio too short for reliable prediction (<1s)")
    hubert = get_hubert()
    with torch.no_grad():
        features, _ = hubert.extract_features(waveform)
        accent_stack = torch.stack(features[2:5] + features[11:13])
        avg_all = accent_stack.mean(dim=0)
        pooled = avg_all.mean(dim=1)
    return pooled

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "no audio file provided"}), 400
    f = request.files["audio"]
    if f.filename == "":
        return jsonify({"error": "empty filename"}), 400
    _, ext = os.path.splitext(f.filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".webm") as tmp:
        f.save(tmp.name)
        input_path = tmp.name
    wav_path = input_path + "_conv.wav"
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ac", "1", "-ar", "16000", "-f", "wav", wav_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True, shell=(platform.system() == "Windows"))
        feats = preprocess_and_extract_features(wav_path)
        feats = feats.to(DEVICE).float()
        mdl = load_model()
        with torch.no_grad():
            logits, _ = mdl(feats, grl_lambda=0.0)
            temperature = 1.25
            probs = F.softmax(logits / temperature, dim=1)
            top_idx = int(probs.argmax(dim=1).cpu().item())
            conf = float(probs[0, top_idx].cpu().item())
            
        return jsonify({
            "accent": top_idx,
            "accent_index": top_idx,
            "label": ACCENT_LABELS[top_idx],
            "confidence": round(conf, 4)
        }), 200

    except subprocess.CalledProcessError:
        return jsonify({"error": "Audio conversion failed — check FFmpeg installation or PATH"}), 500
    except Exception as e:
        print("[ERROR]", e)
        return jsonify({"error": f"Server error: {str(e)}"}), 500
    finally:
        for p in [input_path, wav_path]:
            try:
                os.remove(p)
            except Exception:
                pass

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
