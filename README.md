# ⚡ IntelliPQ: AI-Based Power Quality Monitoring & Fault Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-brightgreen.svg)](https://www.python.org/)
[![Framework: Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Deep Learning: TensorFlow](https://img.shields.io/badge/AI-TensorFlow%20%2F%20Keras-orange.svg)](https://tensorflow.org/)
[![Standards: IEEE 1159 & 519](https://img.shields.io/badge/Compliance-IEEE%201159%20%7C%20519-yellow.svg)](https://standards.ieee.org/)

**IntelliPQ** is a smart grid diagnostic intelligence platform that combines **Digital Signal Processing (DSP)**, **IEEE Deterministic Standards**, and **Deep Learning (1D-CNN + Bidirectional LSTM)** to monitor, classify, and resolve power quality disturbances in real time.

Unlike traditional power monitoring systems that rely solely on static threshold rules or purely black-box AI models, IntelliPQ features a **Neuro-Symbolic Arbitration Engine** that arbitrates between physics-based rules and deep neural networks to prevent false alarms, safeguard critical equipment, and provide explainable engineering decisions.

---

## 📌 Table of Contents
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Power Quality Disturbances Classified](#-power-quality-disturbances-classified)
- [Neuro-Symbolic Arbitration Engine](#-neuro-symbolic-arbitration-engine)
- [Mathematical & Engineering Formulations](#-mathematical--engineering-formulations)
- [Repository Structure](#-repository-structure)
- [Quick Start Guide](#-quick-start-guide)
- [Deep Learning Model Training](#-deep-learning-model-training)
- [Running via Docker](#-running-via-docker)
- [Automated Testing](#-automated-testing)
- [CSV Ingestion Format](#-csv-ingestion-format)
- [License & Author](#-license--author)

---

## 🚀 Key Features

* **Multi-Modal Data Ingestion**:
  * *Interactive Waveform Synthesizer*: Fine-tune voltage amplitude, grid frequency, harmonic injection, and Gaussian sensor noise via intuitive sliders.
  * *CSV File Uploader*: Drag-and-drop field-recorded waveforms or digital fault recorder (DFR) logs.
  * *Preloaded Benchmark Samples*: One-click access to standard IEEE disturbance event benchmarks.
* **Full DSP Analytics Suite**:
  * Fast Fourier Transform (FFT) amplitude spectrum isolation.
  * Real-time Total Harmonic Distortion (THD), RMS voltage, Peak amplitude, and Crest Factor estimation.
* **Dual-Diagnosis Processing**:
  * *Deterministic Rule Engine*: Strictly evaluated against IEEE 1159 and IEEE 519 limits.
  * *1D-CNN + Bidirectional LSTM Model*: Classifies raw waveform temporal features and harmonic patterns across AC cycles.
* **Intelligent Arbitration Layer**:
  * Resolves disagreements between AI and Rule engines using consensus, critical threshold overrides, and multi-parameter evidence cross-validation.
* **Prescriptive Engineering Action Engine**:
  * Translates raw diagnostic flags into actionable electrical utility procedures (e.g., VFD harmonic filtration, AVR adjustment, capacitor bank de-energization).
* **Automated Audit Certificates**:
  * One-click download of official diagnostic inspection certificates (`.txt`) and exported waveform data (`.csv`).

---

## 🏛️ System Architecture

```
                  ┌────────────────────────────────────────┐
                  │          Input Voltage Signal          │
                  │ (Synthesizer / CSV Upload / Benchmarks)│
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │    Digital Signal Processing (DSP)     │
                  │  • RMS, Peak, Mean, Std, Crest Factor  │
                  │  • FFT Spectrum & THD Calculation      │
                  └─────────────┬──────────────────────────┘
                                │
               ┌────────────────┴────────────────┐
               ▼                                 ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│     Rule-Based Detection      │ │      AI Neural Network        │
│   (IEEE 1159 / 519 Limits)    │ │   (1D-CNN + Bi-LSTM Model)    │
└──────────────┬────────────────┘ └──────────────┬────────────────┘
               │                                 │
               └────────────────┬────────────────┘
                                │
                                ▼
         ┌───────────────────────────────────────────────┐
         │     Neuro-Symbolic Arbitration Engine         │
         │  1. Check Consensus                           │
         │  2. Safety Critical Threshold Override        │
         │  3. AI High-Confidence (>90%) Validation      │
         │  4. Parameter Evidence Cross-Validation       │
         └──────────────────────┬────────────────────────┘
                                │
                                ▼
         ┌───────────────────────────────────────────────┐
         │              Final Diagnosis & XAI            │
         │  • Diagnostic Verdict & Confidence Score      │
         │  • Parameter Evidence Breakdown               │
         │  • Prescriptive Engineering Remediation       │
         │  • Exportable Inspection Certificate          │
         └───────────────────────────────────────────────┘
```

---

## ⚡ Power Quality Disturbances Classified

| Disturbance Class | IEEE Standard | Threshold Criteria | Primary Causes |
| :--- | :--- | :--- | :--- |
| **Normal** | IEEE 1159 | $0.90 \le V \le 1.10\text{ pu}$, $\text{THD} < 5\%$ | Stable, balanced grid operations |
| **Voltage Sag** | IEEE 1159 | $V < 0.90\text{ pu}$ (typically $0.1 - 0.9\text{ pu}$) | Large induction motor starts, line-to-ground faults |
| **Voltage Swell** | IEEE 1159 | $V > 1.10\text{ pu}$ (typically $1.1 - 1.8\text{ pu}$) | Abrupt large load disconnection, capacitor energization |
| **Harmonic Distortion** | IEEE 519 | $\text{THD} \ge 5.0\%$ | Non-linear power electronics (VFDs, rectifiers, inverters) |

---

## 🧠 Neuro-Symbolic Arbitration Engine

When deep learning models encounter noisy waveforms or edge-case transients, they may output conflicting predictions with deterministic physics rules. IntelliPQ resolves conflicts using an expert arbitration hierarchy:

1. **Consensus Mode**: If Rule-Based and AI agree $\rightarrow$ Accepted immediately with high confidence.
2. **Critical Safety Override**: If physical parameters severely breach safety margins:
   * $\text{THD} \ge 7.5\%$ ($1.5\times$ limit) $\rightarrow$ Overrides to **Harmonic Distortion**.
   * $V < 0.765\text{ pu}$ ($0.85\times$ sag limit) $\rightarrow$ Overrides to **Voltage Sag**.
   * $V > 1.265\text{ pu}$ ($1.15\times$ swell limit) $\rightarrow$ Overrides to **Voltage Swell**.
3. **High AI Confidence**: If AI confidence exceeds $90\%$ and no critical safety rules are breached $\rightarrow$ AI pattern recognition takes precedence.
4. **Parameter Cross-Validation**: If AI confidence is between $70\%$ and $90\%$, the system audits 5 physical parameters (Amplitude, THD, RMS, Harmonic input, Frequency). If cross-validation fails, the system safely falls back to the deterministic rule.

---

## 📐 Mathematical & Engineering Formulations

### 1. Root Mean Square (RMS) Voltage
$$V_{\text{rms}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} v[i]^2}$$

### 2. Crest Factor (CF)
$$CF = \frac{V_{\text{peak}}}{V_{\text{rms}}} \quad (\text{Ideal sine wave } CF = \sqrt{2} \approx 1.4142)$$

### 3. Total Harmonic Distortion (THD)
According to IEEE 519:
$$\text{THD} = \frac{\sqrt{\sum_{h=2}^{H} V_h^2}}{V_1} \times 100\%$$
Where:
* $V_1$ is the RMS magnitude of the fundamental frequency component ($50\text{ Hz}$ or $60\text{ Hz}$).
* $V_h$ is the RMS magnitude of harmonic order $h$.

---

## 📂 Repository Structure

```
IntelliPQ/
├── .github/
│   └── workflows/
│       └── ci.yml                     # Automated multi-version CI test pipeline
├── data/
│   └── sample_waveforms/              # Preloaded benchmark CSVs for rapid testing
│       ├── normal_sample.csv
│       ├── voltage_sag_sample.csv
│       ├── voltage_swell_sample.csv
│       └── harmonic_distortion_sample.csv
├── scripts/
│   └── generate_sample_data.py        # Generates benchmark CSV waveforms
├── src/
│   ├── __init__.py                    # Core package init
│   ├── signal_generator.py            # IEEE 1159 synthetic disturbance generator
│   ├── dsp.py                         # FFT, THD, RMS, Crest Factor algorithms
│   ├── rule_engine.py                 # Deterministic IEEE threshold evaluator
│   ├── arbitration_engine.py          # Neuro-symbolic arbitration & XAI engine
│   └── model_utils.py                 # CNN + Bi-LSTM architecture & inference
├── tests/
│   ├── __init__.py
│   ├── test_dsp.py                    # Unit tests for FFT & THD calculations
│   ├── test_rules.py                  # Unit tests for rule trigger limits
│   ├── test_signal_generator.py       # Unit tests for synthetic signal generation
│   └── test_arbitration.py            # Unit tests for consensus & override rules
├── power_quality_app.py               # Streamlit interactive application
├── train_model.py                     # Synthetic dataset generation & CNN-LSTM training
├── Dockerfile                         # Container build definition
├── docker-compose.yml                 # Streamlit orchestration
├── requirements.txt                   # Production dependencies
├── setup.py                           # Python package installation
├── .gitignore                         # Git exclusion rules
├── LICENSE                            # MIT License
└── README.md                          # Comprehensive documentation
```

---

## ⚡ Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/niranjan-crypt/IntelliPQ-AI-Based-Power-Quality-Monitoring-Fault-Detection.git
cd IntelliPQ-AI-Based-Power-Quality-Monitoring-Fault-Detection
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Generate Benchmark Samples
```bash
python scripts/generate_sample_data.py
```

### 5. Launch the Dashboard
```bash
streamlit run power_quality_app.py
```
Open your browser at `http://localhost:8501`.

---

## 🤖 Deep Learning Model Training

The repository includes an automated pipeline that generates synthetic IEEE-compliant waveforms, trains a **1D-CNN + Bidirectional LSTM** network, and serializes the model to `pq_model.h5`:

```bash
python train_model.py --samples-per-class 600 --epochs 12 --batch-size 32 --output pq_model.h5
```

### Model Architecture:
1. **1D-CNN Block**: 
   * `Conv1D(32, kernel=7)` $\rightarrow$ `BatchNorm` $\rightarrow$ `MaxPool1D(2)`
   * `Conv1D(64, kernel=5)` $\rightarrow$ `BatchNorm` $\rightarrow$ `MaxPool1D(2)`
2. **Temporal Modeling**:
   * `Bidirectional(LSTM(48))` captures cyclical waveform dependencies.
3. **Classification Head**:
   * `Dense(64)` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Dense(4, activation='softmax')`

Once trained, `power_quality_app.py` automatically detects and activates `pq_model.h5`.

---

## 🐳 Running via Docker

To run the entire system in an isolated container without configuring local Python environments:

```bash
docker compose up --build
```
Access the application at `http://localhost:8501`.

---

## 🧪 Automated Testing

Run the full pytest suite covering DSP accuracy, THD numerical integrity, rule thresholds, and arbitration logic:

```bash
pytest tests/ -v
```

---

## 📊 CSV Ingestion Format

When uploading custom CSV waveform files, format your data with the following headers:

```csv
time_seconds,voltage_pu
0.000,0.0000
0.001,0.3090
0.002,0.5878
...
```
* Sample rate: $1000\text{ Hz}$ recommended ($1000$ points for $1.0\text{ s}$ duration). Signals of differing lengths are automatically interpolated by the ingestion engine.

---

## 📄 License & Author

Distributed under the **MIT License**. See [LICENSE](LICENSE) for more information.

Developed by **[Niranjan Krishnakumar](https://github.com/niranjan-crypt)**.
