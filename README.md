# Machine Learning-Based Detection of GPS Spoofing in Autonomous Vehicles

## Author
**Brian Stauffer**  
Cybersecurity & Networking Researcher  

---

## Project Overview
This project explores how **machine learning** can detect **GPS spoofing attacks** in autonomous vehicles (AVs), a growing threat to navigation safety.  

I evaluated **LSTM**, **CNN**, and **Transformer** deep learning models, and found that **hybrid approaches** provide the best defense against spoofing attempts.  

---

## Key Highlights
- **LSTM** → Excellent at spotting sequential GPS anomalies (98.5% accuracy).  
- **CNN** → Strong at spatial feature extraction (up to 99% accuracy).  
- **Transformers** → Handle complex, long-range interactions (>95% accuracy).  
- **Hybrid Models** → Combining LSTM + CNN + Transformer improves resilience.  

---

## Applications
- Self-driving taxis and logistics fleets  
- Public transit and smart cities  
- Military UAVs & defense navigation  
- Industrial automation, aerospace, and maritime GPS systems  

---

## What You’ll Find
- `Stauffer_GPS_Spoofing.pdf` – Full manuscript with methodology, analysis, and findings.
- `Publication Version.pdf` – Publication version of GPS Spoofing manuscript.
- Model comparison and future research recommendations.  
- Demo Code
  
---

## Demo Code
Hybrid GPS Spoofing Detection Demo

This repository includes a working hybrid deep-learning demo implementing a CNN → LSTM → Transformer architecture for GPS spoofing detection in autonomous systems.

The demo illustrates the practical application of the concepts described in the manuscript, showing how hybrid temporal–spatial–attention modeling improves robustness against coordinated GNSS spoofing attacks.

📁 Folder: /hybrid_gps_demo

Contents:

data_gen.py — synthetic GNSS data generator (pseudorange, SNR, Doppler)

model_hybrid.py — hybrid deep-learning architecture (Conv1D, LSTM, MultiHeadAttention)

train_hybrid_demo.py — trains the model and saves results to /artifacts

predict_demo.py — runs predictions on a CSV or new synthetic sequence

README.md — step-by-step setup and usage

requirements.txt — dependencies for local setup

---

**Outputs**

After training, the demo produces an artifacts/ directory containing:

- model.keras — trained Keras model

- roc_plot.png — ROC curve image

- history.json, metrics.json — training metrics

- summary.txt — textual report with classification results

- example_sequence.csv — sample input for prediction testing

---

**Model Summary**

The model fuses:

- CNN layers to capture local spectral features

- LSTM layers to model temporal dependencies

- Transformer attention to focus on anomalous temporal patterns across satellites

This hybrid approach increases detection reliability under diverse spoofing strategies, achieving target metrics of Pd ≥ 95% and Pfa ≤ 1% in controlled synthetic trials.
