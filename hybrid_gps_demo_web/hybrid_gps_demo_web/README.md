# Hybrid GPS Spoofing Detector — Flask Web App

A minimal Flask UI that loads your **hybrid CNN→LSTM→Transformer** model and performs inference
on uploaded CSV sequences (`pseudorange,snr,doppler`). Provides synthetic demo buttons too.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train (if you haven't yet)
The app expects a trained model at: `../hybrid_gps_demo/artifacts/model.keras`

If you don’t have one yet, train it first:
```bash
cd ../hybrid_gps_demo
pip install -r requirements.txt
python train_hybrid_demo.py --samples 1200 --seq-len 256 --epochs 8 --batch-size 64
```

## Run the app
```bash
cd hybrid_gps_demo_web
python app.py
# Visit http://127.0.0.1:5000
```

## CSV format
```
pseudorange,snr,doppler
0.12,29.8,0.01
...
```

## Notes
- If a saved model isn't found or TensorFlow can't load it, the app will **fallback**
  to a tiny synthetic-trained model (2 epochs) so you can still demo the UI.
- Optionally place a ROC image at either:
  - `../hybrid_gps_demo/artifacts/roc_plot.png`, or
  - `hybrid_gps_demo_web/static/roc_plot.png`
