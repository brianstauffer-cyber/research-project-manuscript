# Hybrid CNN→LSTM→Transformer GPS Spoofing Demo

This project trains a **hybrid deep-learning model** (1D CNN + LSTM + Transformer attention)
to detect GPS spoofing from synthetic GNSS-like time-series (pseudorange residuals, SNR, Doppler).

## Files
- `data_gen.py` — generates synthetic sequences with optional spoof anomalies.
- `model_hybrid.py` — defines the Keras model: Conv1D → LSTM → MultiHeadAttention → Dense.
- `train_hybrid_demo.py` — trains and evaluates the model, saves artifacts to `artifacts/`.
- `predict_demo.py` — loads a saved model to predict on a provided CSV or a new synthetic sample.
- `requirements.txt` — Python dependencies.

## Quick Start

```bash
# 1) Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Train the demo (adjust epochs, samples in arguments)
python train_hybrid_demo.py --samples 1200 --seq-len 256 --epochs 8 --batch-size 64

# 4) Inspect outputs
ls artifacts/
#  - model.keras
#  - history.json
#  - roc_plot.png
#  - metrics.json
#  - example_sequence.csv

# 5) Run a prediction on a CSV (3 columns: pseudorange,snr,doppler)
python predict_demo.py --csv artifacts/example_sequence.csv
# Or generate a new synthetic sample
python predict_demo.py --synthetic --seq-len 256
```

## CSV Format
Provide a **single sequence** CSV with columns:
```
pseudorange,snr,doppler
0.12,29.8,0.01
...
```

## Notes
- The **Transformer** block uses `tf.keras.layers.MultiHeadAttention` with a residual + layer norm.
- This is a compact, demonstration-scale model — increase model width/depth and training data for stronger results.
- You can switch to `tf.data` pipelines and mixed precision for speed on GPU.

## License
MIT — use freely with attribution.
