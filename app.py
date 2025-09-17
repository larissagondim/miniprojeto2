import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
import os
import soundfile as sf

# --------------------------------- PARTE 1: EXTRAIR FEATURES --------------------------------- #

# Carregar o modelo e o scaler
import joblib

# Definir caminhos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_recognition_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Verificar e carregar modelo e scaler
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Modelo ou scaler não encontrados. Por favor, certifique-se de que os arquivos estão na pasta 'models'.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    st.error(f"Erro ao carregar modelo ou scaler: {str(e)}")
    st.stop()

# Lista de emoções
EMOTIONS = ["angry", "calm", "disgust", "fear",
            "happy", "neutral", "sad", "surprise"]
EMOTIONS_DISPLAY = ["Angry!? 😠", "Calm 😌", "Disgust... 🤢", "Fear 😨 ...", "Happy!! 😊", "Neutral 😐", "Sad...😢", "Surprise!! 😲"]
# Função para extrair features
def extract_features(audio_path):
    data, sr = librosa.load(audio_path, sr=16000, mono=True)
    features = []

    # Zero Crossing Rate
    # Extract the zcr here
    zcr = librosa.feature.zero_crossing_rate(data).mean()
    features.extend([zcr]) # zcr é escalar, logo, pro extend funcionar, precisa estar em uma lista
    

    # Chroma STFT
    # Extract the chroma stft here
    chromastft = librosa.feature.chroma_stft(y=data, sr=sr).mean(axis=1)
    features.extend(chromastft)

    # MFCCs
    # Extract the mfccs here
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).mean(axis=1)
    features.extend(mfccs)

    # RMS
    # Extract the rms here
    rms = librosa.feature.rms(y=data).mean()
    features.extend([rms]) # mesma coisa que o zcr

    # Mel Spectrogram
    # Extract the mel here
    mel = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=106).mean(axis=1)  
    features.extend(mel)

    # Garantir que tenha exatamente 162 features (ou truncar/zerar)
    target_length = 155
    if len(features) < target_length:
        features.extend([0] * (target_length - len(features)))
    elif len(features) > target_length:
        features = features[:target_length]

    return np.array(features).reshape(1, -1)


# --------------------------------- PARTE 2: STREAMLIT --------------------------------- #

# Configuração do app Streamlit (Título e descrição)
# Code here
st.title("Emotion detector from audio 🎧")
st.write("Try uploading a random audio file to find out which emotion it brings to the scene! 😄")
# Upload de arquivo de áudio (wav, mp3, ogg)
uploaded_file = st.file_uploader(
    "Choose your type of audio file (wav, mp3 or ogg)", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Salvar temporariamente o áudio
    Path = "temp_audio.wav"
    with open(Path, "wb") as f:
        f.write(uploaded_file.read())
    # Reproduzir o áudio enviado
    st.audio(Path) 
    # Code here

    # Extrair features
    try:
        # extrair features
        features = extract_features(Path)
        # normalizar
        features_norm = scaler.transform(features)
        # ajustar formato pro modelo
        features_adj = features_norm.reshape(1, 155, 1)
        x = np.array(features_adj)  
        # printar dimensões
        st.write(f"Shape: {x.shape}")
        # predição
        y = model.predict(features_adj)
        # resultado
        st.write("Result of the emotion prediction: ")
        st.write(f"**{EMOTIONS_DISPLAY[np.argmax(y)]}** with a confidence of {np.max(y)*100:.2f}%")
        # probabilidades
        st.write("Prediction probabilities for each emotion:")
        for i, emotion in enumerate(EMOTIONS_DISPLAY):
            st.write(f"{emotion}: {y[0][i]*100:.2f}%")
        st.bar_chart(y.T, y_label="Probability (%)", x_label=EMOTIONS_DISPLAY)
            
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        if os.path.exists(Path):
            os.remove(Path)