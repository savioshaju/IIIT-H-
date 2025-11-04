import os
import tempfile
import torch
import torchaudio
import torch.nn.functional as F
import gradio as gr
import soundfile as sf
from model_def import DANNModel

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "accent_dann_model.pt"
ACCENTS = ["andhra_pradesh", "gujarat", "jharkhand", "karnataka", "kerala", "tamil"]

# ---------------- FOOD MAP ----------------
FOOD_MAP = {
    "kerala": ["Puttu and Kadala Curry", "Sadya", "Porotta and Beef Fry", "Dosa"],
    "tamil": ["Chettinad Chicken Curry", "Pongal", "Kothu Porotta"],
    "karnataka": ["Bisi Bele Bath", "Mysore Pak", "Raggi Mudde"],
    "andhra_pradesh": ["Pesarattu", "Ulava Charu", "Punugulu"],
    "jharkhand": ["Litti Chokha", "Thekua", "Dhuska"],
    "gujarat": ["Dhokla", "Thepla", "Undhiyu", "Khandvi"]
}

# ---------------- LOAD MODEL ----------------
model = DANNModel(input_dim=768, hidden_dim=512, num_classes=len(ACCENTS), n_domains=4)
state = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(state, dict) and "model_state" in state:
    model.load_state_dict(state["model_state"])
else:
    model.load_state_dict(state)
model = model.to(DEVICE).eval()

HUBERT = torchaudio.pipelines.HUBERT_BASE.get_model().to(DEVICE).eval()

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform.to(DEVICE)
    with torch.no_grad():
        features, _ = HUBERT.extract_features(waveform)
        pooled = features[-1].mean(dim=1)
    return pooled

# ---------------- PREDICT FUNCTION ----------------
def predict_accent(audio):
    if audio is None:
        return "⚠️ Please record or upload an audio.", None, None

    sr, data = audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, data, sr)

    feats = extract_features(tmp_path)
    os.remove(tmp_path)

    with torch.no_grad():
        logits, _ = model(feats, grl_lambda=0.0)
        probs = F.softmax(logits, dim=1)
        idx = probs.argmax(dim=1).item()
        conf = probs[0, idx].item()

    accent = ACCENTS[idx]
    foods = FOOD_MAP.get(accent, ["No recommendations available."])

    accent_name = accent.replace("_", " ").title()
    conf_text = f"{conf * 100:.2f}%"
    food_list = "\n".join(f"🍽️ {f}" for f in foods)

    return f"🌍 {accent_name}", f"📊 Confidence: {conf_text}", food_list

# ---------------- MODERN GRADIO UI ----------------
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")) as demo:
    gr.Markdown(
        """
        <div style="text-align:center; margin-bottom:20px;">
            <h1 style="font-size:2.3em; font-weight:700;">🎙️ Taste Tales</h1>
            <p style="font-size:1.1em;">Detect your regional accent & get authentic food recommendations from your culture.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",
                label="🎧 Speak or Upload Your Voice",
                
            )
            submit_btn = gr.Button("🔍 Analyze Accent", variant="primary")

        with gr.Column(scale=1):
            accent_output = gr.Textbox(label="Detected Accent", interactive=False)
            confidence_output = gr.Textbox(label="Model Confidence", interactive=False)
            food_output = gr.Textbox(label="🍴 Suggested Traditional Dishes", lines=6, interactive=False)

    submit_btn.click(
        predict_accent,
        inputs=audio_input,
        outputs=[accent_output, confidence_output, food_output]
    )

    gr.Markdown(
        """
        <div style="text-align:center; font-size:0.9em; margin-top:30px; color:gray;">
            Built with <b>PyTorch</b> + <b>HuBERT</b> + <b>Gradio</b>
        </div>
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, debug=True)
