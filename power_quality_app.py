import datetime
import io
import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft

try:
    from src.dsp import compute_rms, compute_peak, compute_crest_factor, compute_thd, compute_fft_spectrum
    from src.rule_engine import RuleEngine
    from src.arbitration_engine import ArbitrationEngine
    from src.model_utils import load_pq_model, predict_waveform
except ImportError:
    RuleEngine = None
    ArbitrationEngine = None

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="IntelliPQ - Power Quality Monitoring System",
    page_icon="⚡",
    layout="wide",
)

# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_ai_model():
    model_paths = ["pq_model.h5", "models/pq_model.h5"]
    for path in model_paths:
        if os.path.exists(path):
            try:
                from tensorflow.keras.models import load_model
                return load_model(path)
            except Exception as e:
                st.warning(f"⚠️ Error loading model from {path}: {e}")
                return None
    return None

model = load_ai_model()

# ==============================
# HEADER
# ==============================

st.title("⚡ IntelliPQ: Power Quality Monitoring & Fault Detection System")
st.markdown("**Real-Time Intelligent Hybrid Diagnosis: Deterministic IEEE Standards + Deep Learning (CNN-LSTM)**")
st.markdown("---")

# ==============================
# SIDEBAR - DATA SOURCE SELECTION
# ==============================

st.sidebar.header("🎛️ Input Mode")
input_mode = st.sidebar.radio(
    "Select Input Source",
    ["Interactive Waveform Synthesizer", "Upload Waveform CSV", "Preloaded Benchmark Samples"],
)

timesteps = 1000
duration = 1.0
t = np.linspace(0, duration, timesteps, endpoint=False)
signal = np.zeros(timesteps)
amplitude = 1.0
frequency = 50.0
harmonic = 0.0

if input_mode == "Interactive Waveform Synthesizer":
    st.sidebar.subheader("Presets")
    preset_col1, preset_col2 = st.sidebar.columns(2)
    with preset_col1:
        if st.button("Normal Grid"):
            st.session_state["amp"] = 1.0
            st.session_state["freq"] = 50
            st.session_state["harm"] = 0.0
        if st.button("Voltage Swell"):
            st.session_state["amp"] = 1.30
            st.session_state["freq"] = 50
            st.session_state["harm"] = 0.0
    with preset_col2:
        if st.button("Voltage Sag"):
            st.session_state["amp"] = 0.65
            st.session_state["freq"] = 50
            st.session_state["harm"] = 0.0
        if st.button("Harmonics"):
            st.session_state["amp"] = 1.0
            st.session_state["freq"] = 50
            st.session_state["harm"] = 0.35

    def_amp = st.session_state.get("amp", 1.0)
    def_freq = st.session_state.get("freq", 50)
    def_harm = st.session_state.get("harm", 0.0)

    st.sidebar.subheader("Fine-Tune Parameters")
    amplitude = st.sidebar.slider("⚡ Voltage Amplitude (pu)", 0.3, 1.6, float(def_amp), 0.01)
    frequency = st.sidebar.slider("📊 Frequency (Hz)", 45, 55, int(def_freq))
    harmonic = st.sidebar.slider("🌊 3rd Harmonic Level (pu)", 0.0, 0.6, float(def_harm), 0.01)
    noise_level = st.sidebar.slider("🔊 Background Noise Level (pu)", 0.0, 0.08, 0.0, 0.005)

    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    if harmonic > 0:
        signal += harmonic * np.sin(2 * np.pi * 3 * frequency * t)
    if noise_level > 0:
        signal += np.random.normal(0, noise_level, timesteps)

elif input_mode == "Upload Waveform CSV":
    st.sidebar.subheader("Upload Signal CSV")
    uploaded_file = st.sidebar.file_uploader("Upload CSV (columns: time_seconds, voltage_pu)", type=["csv"])
    if uploaded_file is not None:
        try:
            df_up = pd.read_csv(uploaded_file)
            v_col = [col for col in df_up.columns if "volt" in col.lower() or "signal" in col.lower()]
            if v_col:
                raw_sig = df_up[v_col[0]].to_numpy()
            else:
                raw_sig = df_up.iloc[:, -1].to_numpy()

            if len(raw_sig) != timesteps:
                indices = np.linspace(0, len(raw_sig) - 1, timesteps)
                signal = np.interp(indices, np.arange(len(raw_sig)), raw_sig)
            else:
                signal = raw_sig

            amplitude = float(np.max(np.abs(signal)))
            st.sidebar.success(f"✓ Uploaded {len(raw_sig)} samples successfully!")
        except Exception as e:
            st.sidebar.error(f"Error parsing CSV: {e}")
            signal = np.sin(2 * np.pi * 50 * t)
    else:
        st.sidebar.info("Upload a CSV file or switch to 'Preloaded Benchmark Samples'.")
        signal = np.sin(2 * np.pi * 50 * t)

else:  # Preloaded Benchmark Samples
    st.sidebar.subheader("Select Benchmark Disturbance")
    sample_files = {
        "Normal Waveform": "data/sample_waveforms/normal_sample.csv",
        "Voltage Sag Event": "data/sample_waveforms/voltage_sag_sample.csv",
        "Voltage Swell Event": "data/sample_waveforms/voltage_swell_sample.csv",
        "Harmonic Distortion Event": "data/sample_waveforms/harmonic_distortion_sample.csv",
    }
    selected_sample = st.sidebar.selectbox("Choose Sample Event", list(sample_files.keys()))
    sample_path = sample_files[selected_sample]

    if os.path.exists(sample_path):
        df_sample = pd.read_csv(sample_path)
        signal = df_sample["voltage_pu"].to_numpy()
        amplitude = float(np.max(np.abs(signal)))
        st.sidebar.success(f"Loaded: {selected_sample}")
    else:
        st.sidebar.warning(f"Sample file not found at {sample_path}. Using synthetic normal waveform.")
        signal = np.sin(2 * np.pi * 50 * t)

st.sidebar.markdown("---")
st.sidebar.info("""
**IEEE Operational Bounds:**  
• **Normal:** Amplitude 0.9-1.1 pu, THD < 5%  
• **Voltage Sag:** < 0.9 pu  
• **Voltage Swell:** > 1.1 pu  
• **Harmonic Distortion:** THD ≥ 5%
""")
