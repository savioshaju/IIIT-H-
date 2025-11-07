const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const resultDiv = document.getElementById("result");
const accentText = document.getElementById("accentText");
const confidenceText = document.getElementById("confidenceText");

const recordBtn = document.getElementById("recordBtn");
const predictRecordBtn = document.getElementById("predictRecordBtn");
const recStatus = document.getElementById("recStatus");
const clearBtn = document.getElementById("clearBtn");

let mediaRecorder = null;
let recordedChunks = [];

async function postAudioBlob(blob) {
  try {
    recStatus.textContent = "Processing...";
    resultDiv.classList.add("d-none");

    const fd = new FormData();
    fd.append("audio", blob, "clip.wav");

    const res = await fetch("/predict", { method: "POST", body: fd });

    let data;
    try {
      const text = await res.text();
      data = text ? JSON.parse(text) : {};
    } catch (e) {
      throw new Error("Invalid or empty JSON response from server.");
    }

    if (!res.ok) throw new Error(data.error || "Prediction failed");

    recStatus.textContent = "Ready";
    resultDiv.classList.remove("d-none");
    accentText.textContent = `Accent: ${data.label}`;
    confidenceText.textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;

    handleAccentResult(data.accent_index ?? data.accent);
  } catch (err) {
    recStatus.textContent = "Error";
    alert(err.message || "Unknown error occurred.");
  }
}


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
        predictRecordBtn.disabled = false;
        recStatus.textContent = "Recording completed — ready to predict.";
      };

      mediaRecorder.start();
      recordBtn.textContent = "⏹ Stop Recording";
      recStatus.textContent = "Recording...";
      predictRecordBtn.disabled = true;
    } catch (err) {
      console.error("Mic access error:", err);
      alert("Microphone access denied or unavailable.");
    }
  } else {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach((t) => t.stop());
    recordBtn.textContent = "🎙️ Start Recording";
  }
});

predictRecordBtn.addEventListener("click", async () => {
  if (!recordedChunks.length) return alert("No recording found.");
  const blob = new Blob(recordedChunks, { type: "audio/wav" });
  await postAudioBlob(blob);
});

uploadBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", async () => {
  if (!fileInput.files.length) return;
  const file = fileInput.files[0];

  recStatus.textContent = "Processing uploaded audio...";
  recordBtn.disabled = true;
  uploadBtn.disabled = true;

  await postAudioBlob(file);

  recStatus.textContent = "Ready ";
  recordBtn.disabled = false;
  uploadBtn.disabled = false;
});

clearBtn.addEventListener("click", () => {
  resultDiv.classList.add("d-none");
  accentText.textContent = "";
  confidenceText.textContent = "";
  recStatus.textContent = "";
  fileInput.value = "";
  recordedChunks = [];
  predictRecordBtn.disabled = true;

  const allStates = ["kerala", "tamil", "karnataka", "andra", "jharkhand", "gujarat"];
  allStates.forEach((state) => {
    document.querySelectorAll(`.${state}`).forEach((card) => {
      card.style.display = "none";
    });
  });
  document.querySelector(".state").style.display = "none";
});

document.querySelector(".demo-btn").addEventListener("click", function () {
  const inputValue = document.querySelector(".demo-i").value.trim();
  const allStates = ["kerala", "tamil", "karnataka", "andra", "jharkhand", "gujarat"];
  allStates.forEach((state) => {
    document.querySelectorAll(`.${state}`).forEach((card) => {
      card.style.display = "none";
    });
  });

  let stateClass = "";
  switch (inputValue) {
    case "0": stateClass = "andra"; break;
    case "1": stateClass = "gujarat"; break;
    case "2": stateClass = "jharkhand"; break;
    case "3": stateClass = "karnataka"; break;
    case "4": stateClass = "kerala"; break;
    case "5": stateClass = "tamil"; break;
    default: alert("Enter a valid number (0–5)"); return;
  }

  document.querySelectorAll(`.${stateClass}`).forEach((card) => {
    card.style.display = "block";
  });

  const stateNameMap = {
    kerala: "Kerala",
    tamil: "Tamil Nadu",
    karnataka: "Karnataka",
    andra: "Andhra Pradesh",
    jharkhand: "Jharkhand",
    gujarat: "Gujarat",
  };
  document.querySelector(".state").textContent = stateNameMap[stateClass];
  document.querySelector(".state").style.display = "inline";
});

function handleAccentResult(accentNum) {
  const orderedStates = ["andra", "gujarat", "jharkhand", "karnataka", "kerala", "tamil"];
  const stateClass = typeof accentNum === "number" ? orderedStates[accentNum] : accentNum;
  if (!stateClass) return;

  const allStates = ["kerala", "tamil", "karnataka", "andra", "jharkhand", "gujarat"];
  allStates.forEach((state) => {
    document.querySelectorAll(`.${state}`).forEach((card) => {
      card.style.display = "none";
    });
  });

  document.querySelectorAll(`.${stateClass}`).forEach((card) => {
    card.style.display = "block";
  });

  const stateNameMap = {
    kerala: "Kerala",
    tamil: "Tamil Nadu",
    karnataka: "Karnataka",
    andra: "Andhra Pradesh",
    jharkhand: "Jharkhand",
    gujarat: "Gujarat",
  };
  document.querySelector(".state").textContent = stateNameMap[stateClass];
  document.querySelector(".state").style.display = "inline";
}
