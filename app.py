git add requirements.txt
git commit -m "Added requirements.txt"
git push

import streamlit as st
import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from speechbrain.pretrained import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Voice CAPTCHA (Embeddings)", layout="wide")

# ------------------ Load Pretrained Model ------------------
@st.cache_resource
def load_model():
    return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

classifier = load_model()

# ------------------ Helper Functions ------------------
def extract_embedding(file):
    signal, sr = librosa.load(file, sr=16000)  # ECAPA expects 16kHz
    signal = torch.tensor(signal).unsqueeze(0)
    embedding = classifier.encode_batch(signal).detach().cpu().numpy()
    return embedding.squeeze()

def log_result(exp_no, user_id, clip_name, score, result):
    log_file = "authentication_log.csv"
    data = {
        "Timestamp": datetime.now(),
        "Experiment": exp_no,
        "User": user_id,
        "Clip": clip_name,
        "Score": score,
        "Result": result
    }
    df = pd.DataFrame([data])
    if os.path.exists(log_file):
        df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df.to_csv(log_file, index=False)

# ------------------ App Session State ------------------
if "embeddings" not in st.session_state:
    st.session_state.embeddings = {}  # user_id -> mean embedding
if "exp_no" not in st.session_state:
    st.session_state.exp_no = 1

# ------------------ UI ------------------
st.title("🔐 Text-Independent CAPTCHA Voice Authentication (Embeddings)")

tabs = st.tabs(["🎤 Enroll User", "🔎 Validate User", "📊 View Log"])

# ------------------ Enroll User Tab ------------------
with tabs[0]:
    st.subheader("🎤 Enroll a New User")
    user_id = st.text_input("Enter User ID")
    uploaded_clips = st.file_uploader("Upload 3 Audio Clips (.wav)", type=["wav"], accept_multiple_files=True)

    if user_id and uploaded_clips and len(uploaded_clips) == 3:
        embeddings = []
        for clip in uploaded_clips:
            emb = extract_embedding(clip)
            embeddings.append(emb)
        mean_emb = np.mean(embeddings, axis=0)
        st.session_state.embeddings[user_id] = mean_emb
        st.success(f"✅ User '{user_id}' enrolled successfully with embeddings.")
    elif uploaded_clips and len(uploaded_clips) != 3:
        st.warning("⚠️ Please upload exactly 3 clips.")

# ------------------ Validate User Tab ------------------
with tabs[1]:
    st.subheader("🔎 Validate a User with Test Audio")
    test_user_id = st.text_input("Enter User ID for Validation")
    test_clip = st.file_uploader("Upload Test Audio Clip (.wav)", type=["wav"])

    if test_user_id and test_clip and st.session_state.embeddings:
        test_emb = extract_embedding(test_clip).reshape(1, -1)

        scores = {}
        for uid, emb in st.session_state.embeddings.items():
            sim = cosine_similarity(test_emb, emb.reshape(1, -1))[0][0]
            scores[uid] = sim

        # Sort scores
        score_df = pd.DataFrame({
            "User": list(scores.keys()),
            "Cosine Similarity": list(scores.values())
        }).sort_values("Cosine Similarity", ascending=False)

        st.write("### 🔢 Cosine Similarity Scores")
        st.dataframe(score_df)

        # Plot
        plt.figure(figsize=(6,4))
        plt.bar(score_df["User"], score_df["Cosine Similarity"])
        plt.title("Cosine Similarity per User")
        st.pyplot(plt)

        # Decision
        best_match = score_df.iloc[0]["User"]
        best_score = score_df.iloc[0]["Cosine Similarity"]

        if best_match == test_user_id:
            st.success(f"✅ Authenticated as '{test_user_id}' (Similarity: {best_score:.3f})")
            result = "Authenticated"
        else:
            st.error(f"❌ Access Denied! Closest match: '{best_match}' (Similarity: {best_score:.3f})")
            result = "Denied"

        log_result(st.session_state.exp_no, test_user_id, test_clip.name, best_score, result)
        st.session_state.exp_no += 1

# ------------------ Log Tab ------------------
with tabs[2]:
    st.subheader("📊 Authentication Log")
    if os.path.exists("authentication_log.csv"):
        df_log = pd.read_csv("authentication_log.csv")
        st.dataframe(df_log)
        st.download_button("📥 Download Log", df_log.to_csv(index=False), file_name="authentication_log.csv")
    else:
        st.info("No log data available yet.")
