
import os
import io
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Attempt TF import only when running locally
try:
    import tensorflow as tf
except Exception as e:
    tf = None

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "hybrid_gps_demo"))
MODEL_PATH = os.path.join(PROJECT_DIR, "artifacts", "model.keras")  # expected trained model
ROC_PATHS = [
    os.path.join(PROJECT_DIR, "artifacts", "roc_plot.png"),
    os.path.join(BASE_DIR, "static", "roc_plot.png")  # optional drop-in
]

EXPECTED_FEATURES = 3   # pseudorange, snr, doppler
DEFAULT_SEQ_LEN = 256

app = Flask(__name__)
app.secret_key = "dev-secret"  # replace for production

def generate_synthetic(seq_len=256, spoof=False, seed=123):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, seq_len)
    pseudo = 0.5 * np.sin(2 * np.pi * 3 * t) + 0.05 * rng.standard_normal(seq_len)
    snr = 30 + 2 * np.sin(2 * np.pi * 1.5 * t) + 0.5 * rng.standard_normal(seq_len)
    doppler = 0.1 * np.sin(2 * np.pi * 5 * t) + 0.02 * rng.standard_normal(seq_len)
    if spoof:
        start = rng.integers(seq_len // 8, seq_len // 2)
        end = min(seq_len, start + rng.integers(seq_len // 8, seq_len // 2))
        pseudo[start:end] += rng.uniform(2.0, 6.0)
        snr[start:end] -= rng.uniform(5.0, 12.0)
        if rng.random() > 0.5:
            doppler += np.linspace(0, rng.uniform(0.5, 2.0), seq_len)
    return np.stack([pseudo, snr, doppler], axis=-1)  # (T, 3)

def load_model_or_fallback():
    # Load trained model; if unavailable, try to build a tiny model and warn user.
    banner = None
    model = None
    input_len = DEFAULT_SEQ_LEN

    if tf is None:
        banner = "TensorFlow not available. Install dependencies and restart app."
        return model, input_len, banner

    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            # infer expected timesteps
            ishape = model.input_shape
            if isinstance(ishape, list):
                ishape = ishape[0]
            input_len = ishape[1] or DEFAULT_SEQ_LEN
            banner = f"Loaded model: {MODEL_PATH}"
            return model, input_len, banner
        except Exception as e:
            banner = f"Failed to load model: {e}"

    # Fallback: build a tiny model for demo (fast train) so UI still works
    try:
        from tensorflow.keras import layers, Model

        def transformer_block(x, num_heads=2, key_dim=16, ff_dim=64, dropout=0.1):
            attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
            x = layers.Add()([x, attn]); x = layers.LayerNormalization(epsilon=1e-6)(x)
            ff = layers.Dense(ff_dim, activation="relu")(x)
            ff = layers.Dropout(dropout)(ff)
            ff = layers.Dense(x.shape[-1])(ff)
            x = layers.Add()([x, ff]); x = layers.LayerNormalization(epsilon=1e-6)(x)
            return x

        inp = layers.Input(shape=(DEFAULT_SEQ_LEN, EXPECTED_FEATURES))
        x = layers.Conv1D(32, 5, padding="same", activation="relu")(inp)
        x = layers.MaxPooling1D(2)(x)
        x = layers.LSTM(32, return_sequences=True)(x)
        x = transformer_block(x, num_heads=2, key_dim=16, ff_dim=64, dropout=0.1)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(32, activation="relu")(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])

        # Quick warm start on synthetic data
        X = np.stack([generate_synthetic(DEFAULT_SEQ_LEN, spoof=False, seed=i) for i in range(64)] +
                     [generate_synthetic(DEFAULT_SEQ_LEN, spoof=True, seed=1000+i) for i in range(64)], axis=0)
        y = np.array([0]*64 + [1]*64, dtype=np.int32)
        model.fit(X, y, epochs=2, batch_size=32, verbose=0)

        banner = "No saved model found. Started with a quick fallback model (synthetic data, 2 epochs)."
        input_len = DEFAULT_SEQ_LEN
        return model, input_len, banner
    except Exception as e:
        banner = f"Could not build fallback model: {e}"
        return None, input_len, banner

MODEL, INPUT_LEN, LOAD_BANNER = load_model_or_fallback()

def process_csv(file_stream):
    df = pd.read_csv(file_stream)
    arr = df.values.astype("float32")
    if arr.ndim != 2 or arr.shape[1] != EXPECTED_FEATURES:
        raise ValueError(f"CSV must have exactly {EXPECTED_FEATURES} columns: pseudorange,snr,doppler.")
    # pad or truncate to INPUT_LEN
    if arr.shape[0] < INPUT_LEN:
        pad = np.zeros((INPUT_LEN - arr.shape[0], EXPECTED_FEATURES), dtype="float32")
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.shape[0] > INPUT_LEN:
        arr = arr[:INPUT_LEN]
    return arr

def predict_prob(seq_2d):
    if MODEL is None or tf is None:
        raise RuntimeError("Model not available. Install TensorFlow and/or train the hybrid model first.")
    seq = seq_2d[None, ...]  # (1, T, 3)
    prob = float(MODEL.predict(seq, verbose=0).ravel()[0])
    return prob

def find_roc_path():
    for p in ROC_PATHS:
        if os.path.exists(p):
            return p
    return None

from flask import send_from_directory

@app.route("/", methods=["GET"])
def index():
    roc_available = False
    roc_path = find_roc_path()
    if roc_path and os.path.exists(roc_path):
        # If the ROC is outside static/, copy once for display
        static_target = os.path.join(BASE_DIR, "static", "roc_plot.png")
        if roc_path != static_target:
            try:
                import shutil
                shutil.copy(roc_path, static_target)
            except Exception:
                pass
        roc_available = os.path.exists(static_target)
    return render_template("index.html",
                           load_banner=LOAD_BANNER,
                           input_len=INPUT_LEN,
                           roc_available=roc_available)

@app.route("/predict", methods=["POST"])
def predict():
    action = request.form.get("action", "upload")
    try:
        if action == "synthetic-benign":
            arr = generate_synthetic(seq_len=INPUT_LEN, spoof=False, seed=999)
        elif action == "synthetic-spoof":
            arr = generate_synthetic(seq_len=INPUT_LEN, spoof=True, seed=999)
        else:
            f = request.files.get("file")
            if not f or f.filename == "":
                flash("Please choose a CSV file.", "warning")
                return redirect(url_for("index"))
            filename = secure_filename(f.filename)
            arr = process_csv(f.stream)

        prob = predict_prob(arr)
        pred = int(prob >= 0.5)
        result = {
            "prob_spoof": round(prob, 4),
            "pred_label": int(pred),
            "threshold": 0.5
        }
        return render_template("index.html",
                               load_banner=LOAD_BANNER,
                               input_len=INPUT_LEN,
                               result=json.dumps(result, indent=2),
                               roc_available=os.path.exists(os.path.join(BASE_DIR, "static", "roc_plot.png")))
    except Exception as e:
        flash(str(e), "danger")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
