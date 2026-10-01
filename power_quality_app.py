import datetime
import io
import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft

# Optional modular imports with standalone fallbacks
try:
    from src.dsp import compute_rms, compute_peak, compute_crest_factor, compute_thd, compute_fft_spectrum
    from src.rule_engine import RuleEngine
    from src.arbitration_engine import ArbitrationEngine
    from src.model_utils import load_pq_model, predict_waveform
except ImportError:
    # Standalone fallbacks if run outside package structure
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
                from tensorflow.keras.models import load_model # type: ignore
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

# ==============================
# SIGNAL PROCESSING (DSP)
# ==============================

rms = float(np.sqrt(np.mean(signal**2)))
peak = float(np.max(np.abs(signal)))
mean = float(np.mean(signal))
std = float(np.std(signal))
crest_factor = (peak / rms) if rms > 1e-6 else 0.0

# THD Calculation using FFT
fft_vals = np.abs(fft(signal))
half_n = len(fft_vals) // 2
fft_vals = fft_vals[:half_n]

fundamental_idx = int(frequency * timesteps / (timesteps / t[-1])) if input_mode == "Interactive Waveform Synthesizer" else 50
if 0 < fundamental_idx < len(fft_vals):
    fundamental = fft_vals[fundamental_idx]
else:
    fundamental = fft_vals[1]

harmonics_squared = sum(fft_vals[i]**2 for i in range(2, min(20, len(fft_vals))) if i != fundamental_idx)
harmonics = np.sqrt(harmonics_squared)
thd = float((harmonics / fundamental * 100)) if fundamental > 1e-10 else 0.0

# ==============================
# DISPLAY WAVEFORM & FFT SPECTRUM
# ==============================

tab_wave, tab_fft = st.tabs(["📈 Voltage Waveform", "📊 Frequency Spectrum (FFT)"])

with tab_wave:
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(t, signal, linewidth=2, color='#1f77b4', label="Measured Voltage (pu)")
    ax.axhline(y=1.1, color='orange', linestyle='--', linewidth=1, alpha=0.7, label="Swell Limit (1.10 pu)")
    ax.axhline(y=0.9, color='red', linestyle='--', linewidth=1, alpha=0.7, label="Sag Limit (0.90 pu)")
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.6, alpha=0.4)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Voltage (pu)")
    ax.set_title("Real-Time Monitored Voltage Signal")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-2.0, 2.0])
    ax.legend(loc="upper right", framealpha=0.8)
    st.pyplot(fig)

with tab_fft:
    fig_fft, ax_fft = plt.subplots(figsize=(12, 3.8))
    freq_axis = np.linspace(0, 500, half_n)
    norm_fft = (fft_vals * 2.0) / len(signal)
    ax_fft.bar(freq_axis[:150], norm_fft[:150], width=2.5, color='#2ca02c', alpha=0.85)
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Amplitude (pu)")
    ax_fft.set_title("Harmonic Spectrum Analysis (0 - 300 Hz)")
    ax_fft.grid(True, alpha=0.3)
    st.pyplot(fig_fft)

# ==============================
# SIGNAL PARAMETERS METRICS
# ==============================

st.subheader("📊 Signal Extraction Parameters")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Amplitude (Peak)", f"{peak:.3f} pu")
col2.metric("RMS Voltage", f"{rms:.3f} pu")
col3.metric("Total Harmonic Distortion", f"{thd:.2f}%", delta="Normal" if thd < 5.0 else "High THD", delta_color="inverse")
col4.metric("Crest Factor", f"{crest_factor:.3f}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Fundamental Freq", f"{frequency:.1f} Hz")
col6.metric("Signal Mean", f"{mean:.4f}")
col7.metric("Standard Deviation", f"{std:.4f}")
col8.metric("Harmonic Input", f"{harmonic:.2f} pu")

st.markdown("---")

# ==============================
# RULE-BASED DETECTION
# ==============================

st.subheader("🔍 Rule-Based Detection Engine (IEEE 1159 & IEEE 519)")

SAG_LIMIT = 0.90
SWELL_LIMIT = 1.10
THD_LIMIT = 5.0

# Effective amplitude measure
amp_measure = amplitude if input_mode == "Interactive Waveform Synthesizer" else peak

if thd >= THD_LIMIT:
    rule_result = "Harmonic Distortion"
    trigger_reason = f"THD ({thd:.2f}%) exceeds IEEE 519 threshold ({THD_LIMIT}%)"
    status_icon = "🔴"
elif amp_measure < SAG_LIMIT:
    rule_result = "Voltage Sag"
    trigger_reason = f"Voltage ({amp_measure:.3f} pu) fell below IEEE 1159 sag threshold ({SAG_LIMIT} pu)"
    status_icon = "🟡"
elif amp_measure > SWELL_LIMIT:
    rule_result = "Voltage Swell"
    trigger_reason = f"Voltage ({amp_measure:.3f} pu) exceeded IEEE 1159 swell threshold ({SWELL_LIMIT} pu)"
    status_icon = "🟡"
else:
    rule_result = "Normal"
    trigger_reason = "All electrical parameters within nominal IEEE operational limits"
    status_icon = "🟢"

if rule_result == "Normal":
    st.success(f"{status_icon} **Rule-Based Result: {rule_result}** — *{trigger_reason}*")
elif rule_result == "Harmonic Distortion":
    st.error(f"{status_icon} **Rule-Based Result: {rule_result}** — *{trigger_reason}*")
else:
    st.warning(f"{status_icon} **Rule-Based Result: {rule_result}** — *{trigger_reason}*")

with st.expander("📊 View IEEE Detection Rules & Verification Matrix"):
    st.markdown("#### IEEE Compliance Matrix")
    st.table({
        "Parameter": ["Voltage Amplitude", "THD (IEEE 519)", "Nominal Frequency"],
        "Current Measured": [f"{amp_measure:.3f} pu", f"{thd:.2f}%", f"{frequency:.1f} Hz"],
        "Standard Bound": [f"{SAG_LIMIT} - {SWELL_LIMIT} pu", f"< {THD_LIMIT}%", "49.5 - 50.5 Hz"],
        "Compliance Status": [
            "✅ Compliant" if SAG_LIMIT <= amp_measure <= SWELL_LIMIT else "❌ Non-Compliant",
            "✅ Compliant" if thd < THD_LIMIT else "❌ Non-Compliant",
            "✅ Compliant" if 49.5 <= frequency <= 50.5 else "⚠️ Slight Deviation"
        ]
    })
    st.caption(f"**Trigger Reason:** {trigger_reason}")

# ==============================
# AI DETECTION (CNN + LSTM)
# ==============================

st.subheader("🤖 AI-Based Classification (1D-CNN + Bidirectional LSTM)")

ai_result = "N/A"
ai_conf = 0.0
prediction = np.zeros((1, 4))
fault_dict = {0: "Normal", 1: "Voltage Sag", 2: "Voltage Swell", 3: "Harmonic Distortion"}

if model is not None:
    signal_input = signal.reshape(1, timesteps, 1)
    prediction = model.predict(signal_input, verbose=0)
    predicted_class = int(np.argmax(prediction[0]))
    ai_result = fault_dict.get(predicted_class, "Unknown")
    ai_conf = float(prediction[0][predicted_class] * 100.0)

    if ai_result == "Normal":
        st.success(f"🟢 **AI Classification: {ai_result}** (Confidence: {ai_conf:.1f}%)")
    elif ai_result == "Harmonic Distortion":
        st.error(f"🔴 **AI Classification: {ai_result}** (Confidence: {ai_conf:.1f}%)")
    else:
        st.warning(f"🟡 **AI Classification: {ai_result}** (Confidence: {ai_conf:.1f}%)")

    with st.expander("🎯 View AI Neural Network Confidence Breakdown"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Normal", f"{prediction[0][0]*100:.1f}%")
        c2.metric("Voltage Sag", f"{prediction[0][1]*100:.1f}%")
        c3.metric("Voltage Swell", f"{prediction[0][2]*100:.1f}%")
        c4.metric("Harmonic Dist.", f"{prediction[0][3]*100:.1f}%")

        fig2, ax2 = plt.subplots(figsize=(10, 2.8))
        classes = list(fault_dict.values())
        probabilities = prediction[0] * 100
        colors = ['#2ecc71' if i == predicted_class else '#bdc3c7' for i in range(4)]
        ax2.barh(classes, probabilities, color=colors)
        ax2.set_xlabel('Model Confidence (%)')
        ax2.set_xlim([0, 100])
        for i, v in enumerate(probabilities):
            ax2.text(v + 1.5, i, f'{v:.1f}%', va='center', fontweight='bold' if i == predicted_class else 'normal')
        st.pyplot(fig2)
else:
    st.info("ℹ️ AI Model (`pq_model.h5`) not found in workspace.")
    st.caption("💡 **Tip:** Run `python train_model.py` in your terminal to train and export the 1D-CNN + Bi-LSTM neural network. The rule-based engine and arbitration fallback are currently active.")

# ==============================
# INTELLIGENT FAULT RESOLUTION
# ==============================

st.subheader("🧠 Intelligent Neuro-Symbolic Arbitration System")

if model is not None and ai_result != "N/A":
    if rule_result == ai_result:
        final_decision = rule_result
        decision_method = "Consensus (Both Engines Agree)"
        confidence_level = max(ai_conf, 92.0)
    else:
        if ai_conf > 90.0:
            final_decision = ai_result
            decision_method = "AI High Confidence (>90%)"
            confidence_level = ai_conf
        elif thd >= THD_LIMIT * 1.5:
            final_decision = "Harmonic Distortion"
            decision_method = "Critical Safety Override (Severe THD)"
            confidence_level = 95.0
        elif amp_measure < SAG_LIMIT * 0.85:
            final_decision = "Voltage Sag"
            decision_method = "Critical Safety Override (Severe Voltage Sag)"
            confidence_level = 95.0
        elif amp_measure > SWELL_LIMIT * 1.15:
            final_decision = "Voltage Swell"
            decision_method = "Critical Safety Override (Severe Voltage Swell)"
            confidence_level = 95.0
        elif ai_conf > 70.0:
            verification_passed = False
            if ai_result == "Harmonic Distortion" and (thd >= THD_LIMIT * 0.8 or harmonic > 0.15):
                verification_passed = True
            elif ai_result == "Voltage Sag" and (amp_measure < SAG_LIMIT * 1.05 or rms < 0.9):
                verification_passed = True
            elif ai_result == "Voltage Swell" and (amp_measure > SWELL_LIMIT * 0.95 or rms > 1.1):
                verification_passed = True
            elif ai_result == "Normal" and (SAG_LIMIT * 0.95 <= amp_measure <= SWELL_LIMIT * 1.05 and thd < THD_LIMIT * 0.8):
                verification_passed = True

            if verification_passed:
                final_decision = ai_result
                decision_method = "AI Verified by Physical Parameters"
                confidence_level = ai_conf
            else:
                final_decision = rule_result
                decision_method = "Rule-Based (AI Not Verified by Physics)"
                confidence_level = 75.0
        else:
            final_decision = rule_result
            decision_method = "Rule-Based (Low AI Confidence <70%)"
            confidence_level = 70.0

    st.markdown("### 🎯 FINAL DIAGNOSIS")
    if final_decision == "Normal":
        st.success(f"## ✅ {final_decision.upper()}")
    elif final_decision == "Harmonic Distortion":
        st.error(f"## 🚨 {final_decision.upper()}")
    else:
        st.warning(f"## ⚠️ {final_decision.upper()}")

    st.markdown(f"**Confidence: {confidence_level:.1f}%** • *Decision Path: {decision_method}*")
    st.progress(float(confidence_level) / 100.0)

    if rule_result != ai_result:
        st.markdown("---")
        st.markdown("#### 🔬 Conflict Arbitration & Evidence Analysis")
        col_r, col_a, col_m = st.columns(3)
        with col_r:
            st.markdown("##### 📏 Rule-Based Result")
            st.info(f"**{rule_result}**")
            st.caption(f"Trigger: {trigger_reason}")
        with col_a:
            st.markdown("##### 🤖 AI Model Result")
            st.info(f"**{ai_result}** (Conf: {ai_conf:.1f}%)")
        with col_m:
            st.markdown("##### ⚖️ Arbitration Decision")
            st.success(f"**Selected: {final_decision}**")
            st.caption(f"Rationale: {decision_method}")

else:
    final_decision = rule_result
    confidence_level = 80.0
    decision_method = "Deterministic Rule-Based Baseline"

    st.markdown("### 🎯 FINAL DIAGNOSIS")
    if final_decision == "Normal":
        st.success(f"## ✅ {final_decision.upper()}")
    elif final_decision == "Harmonic Distortion":
        st.error(f"## 🚨 {final_decision.upper()}")
    else:
        st.warning(f"## ⚠️ {final_decision.upper()}")

    st.markdown(f"**Confidence: {confidence_level:.1f}%** • *{decision_method}*")
    st.caption(f"Trigger: {trigger_reason}")

# ==============================
# PRESCRIPTIVE ACTIONS & REMEDIATION
# ==============================

st.markdown("---")
st.subheader("🛠️ Recommended Engineering Actions")

if final_decision == "Normal":
    remedy_text = "System operates within standard IEEE tolerances. Maintain routine periodic monitoring."
    st.info(f"**Action:** {remedy_text}")
elif final_decision == "Harmonic Distortion":
    remedy_text = (
        f"Critical THD ({thd:.2f}%) exceeds the IEEE 519 5% limit. Locate non-linear loads (VFDs, switched-mode power supplies). "
        "Inspect passive tuned filters and active power filters. Check shunt capacitor banks for potential harmonic resonance."
    )
    st.error(f"**Action:** {remedy_text}")
elif final_decision == "Voltage Sag":
    drop_pct = ((1.0 - amp_measure) * 100) if amp_measure < 1.0 else 0.0
    remedy_text = (
        f"Voltage dropped {drop_pct:.1f}% below nominal. Check for large inductive motor starting surges, short-circuit faults, "
        "or upstream grid switching events. Inspect Automatic Voltage Regulators (AVR) and ensure uninterruptible power supply (UPS) availability."
    )
    st.warning(f"**Action:** {remedy_text}")
else:  # Swell
    rise_pct = ((amp_measure - 1.0) * 100) if amp_measure > 1.0 else 0.0
    remedy_text = (
        f"Voltage surged {rise_pct:.1f}% above nominal. Investigate sudden large load shedding or single line-to-ground faults on ungrounded systems. "
        "De-energize oversized capacitor banks immediately to prevent insulation breakdown on sensitive electronics."
    )
    st.warning(f"**Action:** {remedy_text}")

# ==============================
# EXPORT REPORTS & WAVEFORM DATA
# ==============================

st.markdown("---")
st.subheader("📥 Export & Reporting Tools")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    report_content = f"""================================================================================
INTELLIPQ POWER QUALITY INSPECTION CERTIFICATE
Timestamp: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
================================================================================

[EXECUTIVE SUMMARY]
Final Diagnosis:     {final_decision}
Diagnostic Method:   {decision_method}
Overall Confidence:  {confidence_level:.1f}%
System Status:       {"PASS - Normal Operations" if final_decision == "Normal" else "ALERT - Power Quality Anomaly Detected"}

[MEASURED ELECTRICAL PARAMETERS]
• Voltage Peak:      {peak:.4f} pu
• Voltage RMS:       {rms:.4f} pu
• Total Harmonic Distortion (THD): {thd:.2f}% (IEEE 519 Limit: 5.0%)
• Crest Factor:      {crest_factor:.4f}
• Frequency:         {frequency:.2f} Hz
• Mean Voltage:      {mean:.4f} pu
• Std Deviation:     {std:.4f}

[DUAL-DIAGNOSTIC TRACEABILITY]
• Deterministic Rule Engine Result: {rule_result}
  - Trigger Reason: {trigger_reason}
• AI Deep Learning Model Result:   {ai_result}
  - Model Confidence: {ai_conf:.1f}%

[ENGINEERING RECOMMENDATION & REMEDIATION]
{remedy_text}

================================================================================
Generated by IntelliPQ AI-Based Monitoring System
https://github.com/niranjan-crypt/IntelliPQ-AI-Based-Power-Quality-Monitoring-Fault-Detection
================================================================================
"""
    st.download_button(
        label="📄 Download Diagnostic Inspection Report (.txt)",
        data=report_content,
        file_name=f"intellipq_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )

with col_exp2:
    csv_buffer = io.StringIO()
    df_current = pd.DataFrame({"time_seconds": t, "voltage_pu": signal})
    df_current.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📊 Download Current Waveform Dataset (.csv)",
        data=csv_buffer.getvalue(),
        file_name=f"waveform_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'>⚡ <b>IntelliPQ</b> • AI-Driven Smart Grid Power Quality Intelligence • Built with Streamlit & TensorFlow</div>", unsafe_allow_html=True)
