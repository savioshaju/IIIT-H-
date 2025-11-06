import os
import tempfile
import platform
import subprocess
import torch
import torchaudio
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, send_from_directory
from model_def import DANNModel

# --------------------------------------------------------
# Flask App Setup
# --------------------------------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_STATE_PATH = "model/accent_dann_model.pt"
N_CLASSES = 6
ACCENT_LABELS = ["andhra_pradesh", "gujarat", "jharkhand", "karnataka", "kerala", "tamil"]

# Global variables (lazy-loaded)
model = None
HUBERT = None


# --------------------------------------------------------
# Lazy Loading
# --------------------------------------------------------
def load_model():
    """Load the accent classifier model only once (on demand)."""
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

        model = model.to(DEVICE)
        model.eval()
        print("[INFO] Accent model loaded successfully.")
    return model


def get_hubert():
    """Load the HuBERT model only once (on demand)."""
    global HUBERT
    if HUBERT is None:
        HUBERT = torchaudio.pipelines.HUBERT_BASE.get_model().to(DEVICE)
        HUBERT.eval()
        print("[INFO] HuBERT model loaded successfully.")
    return HUBERT


# --------------------------------------------------------
# Feature Extraction
# --------------------------------------------------------
def preprocess_and_extract_features(wav_path):
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    waveform = waveform.to(DEVICE)
    hubert = get_hubert()

    with torch.no_grad():
        features = hubert.extract_features(waveform)
        feats = features[0] if isinstance(features, (tuple, list)) else features
        last = feats[-1] if isinstance(feats, (list, tuple)) else feats
        pooled = last.mean(dim=1) if last.dim() == 3 else last.unsqueeze(0)

    return pooled


# --------------------------------------------------------
# Routes
# --------------------------------------------------------
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
        # Convert uploaded audio to mono 16kHz WAV
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
            probs = F.softmax(logits, dim=1)
            top_idx = int(probs.argmax(dim=1).cpu().item())
            conf = float(probs[0, top_idx].cpu().item())

        return jsonify({
            "accent": top_idx,
            "confidence": round(conf, 4)
        })

    except subprocess.CalledProcessError:
        return jsonify({"error": "Audio conversion failed — check FFmpeg installation or PATH"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        for p in [input_path, wav_path]:
            try:
                os.remove(p)
            except Exception:
                pass


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


# --------------------------------------------------------
# Entrypoint
# --------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
