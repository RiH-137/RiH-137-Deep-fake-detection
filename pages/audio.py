import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import os
from moviepy.editor import VideoFileClip
import tempfile

# Load the trained model
MODEL_PATH = "deepfake_voice_detection_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Function to extract features
def extract_features(file_path, max_pad_len=862):
    try:
        audio, sample_rate = librosa.load(file_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print("Feature extraction error:", e)
        return None

# Prediction function
def predict(file_path):
    features = extract_features(file_path)
    if features is None:
        return None, 0.0
    features = features[np.newaxis, ..., np.newaxis]  # Shape: (1, 40, 862, 1)
    prediction = model.predict(features)[0][0]
    label = "Fake Voice (Deepfake/Overdubbed)" if prediction > 0.5 else "Real Voice (Original)"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence

# Streamlit UI
st.set_page_config(page_title="Fake Voice Detection", layout="centered")

st.title("🎙️ Fake Voice Detection App")
st.markdown("Detect whether an audio or video contains a **real** or **fake** voice using AI.")

option = st.radio("Choose an input type:", ("🎧 Audio File", "🎥 Video File"))

uploaded_file = None

if option == "🎧 Audio File":
    uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])
elif option == "🎥 Video File":
    uploaded_file = st.file_uploader("Upload a video file (e.g., MP4)", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file, format="audio/wav" if option == "🎧 Audio File" else "audio/mp4")

    if option == "🎥 Video File":
        st.info("Extracting audio from video...")
        audio_path = tmp_path.replace(".mp4", ".wav")
        try:
            video = VideoFileClip(tmp_path)
            video.audio.write_audiofile(audio_path, codec='pcm_s16le')
            file_to_predict = audio_path
        except Exception as e:
            st.error(f"Error extracting audio: {e}")
            file_to_predict = None
    else:
        file_to_predict = tmp_path

    if file_to_predict:
        st.info("Running prediction...")
        label, confidence = predict(file_to_predict)
        if label:
            st.success(f"🧠 Prediction: {label}")
            st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
        else:
            st.error("Failed to process the file.")
