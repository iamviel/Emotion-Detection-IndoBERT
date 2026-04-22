import os

# =========================
# 🔥 RESOURCE CONTROL (WAJIB)
# =========================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import pandas as pd
import threading

# limit thread torch
torch.set_num_threads(1)

# lock untuk multi-user (anti crash)
lock = threading.Lock()

# =========================
# 1. MODEL PATH (HUGGINGFACE)
# =========================
MODEL_PATH = "iamviel/indobert-emotion"

# =========================
# 2. LOAD MODEL (CACHED)
# =========================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()
    model.to("cpu")  # pastikan CPU only

    return tokenizer, model

tokenizer, model = load_model()

# =========================
# 3. LABEL MAPPING
# =========================
labels = ["Anger", "Fear", "Joy", "Love", "Neutral", "Sad"]

# =========================
# 4. PREPROCESSING
# =========================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =========================
# 5. STREAMLIT UI
# =========================
st.set_page_config(page_title="Emotion Detection", layout="centered")

st.title("🧠 Emotion Detection IndoBERT")
st.write("Masukkan teks tweet untuk diprediksi emosinya")

text_input = st.text_area("Input Text")

# =========================
# 6. PREDICT
# =========================
if st.button("Predict"):

    if text_input.strip() == "":
        st.warning("Masukkan teks dulu")

    else:
        clean_text = preprocess_text(text_input)

        inputs = tokenizer(
            clean_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=48  # lebih ringan dari 64
        )

        # 🔥 LOCK untuk handle multi-device
        with lock:
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                pred_id = torch.argmax(probs, dim=1).item()

        predicted_label = labels[pred_id]
        confidence = probs[0][pred_id].item()

        # =========================
        # 7. OUTPUT
        # =========================
        st.subheader(f"Prediksi: {predicted_label}")
        st.write(f"Confidence: {confidence:.4f}")

        # =========================
        # 8. DETAIL PROBABILITAS
        # =========================
        st.write("### Probabilitas per kelas:")

        for i, label in enumerate(labels):
            st.write(f"{label}: {probs[0][i].item():.4f}")

        # =========================
        # 9. VISUAL BAR CHART
        # =========================
        prob_df = pd.DataFrame({
            "Emotion": labels,
            "Probability": probs[0].tolist()
        })

        st.bar_chart(prob_df.set_index("Emotion"))