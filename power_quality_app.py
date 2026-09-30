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
