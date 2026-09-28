# IntelliPQ Benchmark Datasets & Waveform Schemas

This directory contains preloaded IEEE benchmark voltage disturbance samples for testing the IntelliPQ diagnostic dashboard and machine learning inference pipeline.

## CSV Waveform Format Specification
All waveform CSV files follow standard Digital Fault Recorder (DFR) conventions:

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `time_seconds` | Float | Timestamp in seconds (from 0.0000 to 1.0000s at 1000 Hz) |
| `voltage_pu` | Float | Instantaneous voltage normalized in per-unit (pu) |

## Benchmark Scenarios
* `normal_sample.csv`: Nominal 50 Hz sinusoidal waveform ($V = 1.0\text{ pu}$, $\text{THD} < 5\%$) with mild Gaussian sensor noise.
* `voltage_sag_sample.csv`: Voltage drop ($V = 0.65\text{ pu}$) occurring between $0.2\text{s}$ and $0.8\text{s}$.
* `voltage_swell_sample.csv`: Voltage surge ($V = 1.35\text{ pu}$) occurring between $0.2\text{s}$ and $0.8\text{s}$.
* `harmonic_distortion_sample.csv`: Waveform injected with 3rd ($150\text{ Hz}$) and 5th ($250\text{ Hz}$) harmonic orders producing $\text{THD} > 5\%$.
