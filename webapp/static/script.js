const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const resultDiv = document.getElementById("result");
const accentText = document.getElementById("accentText");
const confidenceText = document.getElementById("confidenceText");

const recordBtn = document.getElementById("recordBtn");
const predictRecordBtn = document.getElementById("predictRecordBtn");
const recStatus = document.getElementById("recStatus");

let mediaRecorder = null;
let recordedChunks = [];

async function postAudioBlob(blob) {
  const fd = new FormData();
  fd.append("audio", blob, "clip.wav");
  const res = await fetch("/predict", { method: "POST", body: fd });
  const data = await res.json();
  if (res.ok) {
    resultDiv.classList.remove("hidden");
    accentText.textContent = `Accent: ${data.accent}`;
    confidenceText.textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
  } else {
    alert("Error: " + (data.error || JSON.stringify(data)));
  }
}

uploadBtn.addEventListener("click", async () => {
  if (!fileInput.files.length) return alert("Select an audio file first");
  const file = fileInput.files[0];
  await postAudioBlob(file);
});

recordBtn.addEventListener("click", async () => {
  if (!mediaRecorder || mediaRecorder.state === "inactive") {
    try {
      recordedChunks = [];
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) recordedChunks.push(e.data);
      };
      mediaRecorder.onstop = () => {
        recStatus.textContent = `Recorded ${recordedChunks.length} chunk(s)`;
        predictRecordBtn.disabled = false;
      };
      mediaRecorder.start();
      recordBtn.textContent = "⏹ Stop Recording";
      recStatus.textContent = "Recording...";
      predictRecordBtn.disabled = true;
    } catch (err) {
      console.error("Mic access error:", err);
      recStatus.textContent = "Microphone access failed: " + err.message;
      alert("Microphone access error — check browser permissions or HTTPS.");
    }
  } else {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
    recordBtn.textContent = "🎙️ Start Recording";
    recStatus.textContent = "Processing recording...";
  }
});

predictRecordBtn.addEventListener("click", async () => {
  if (!recordedChunks.length) return alert("No recording available");
  const blob = new Blob(recordedChunks, { type: "audio/wav" });
  await postAudioBlob(blob);
});
