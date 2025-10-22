
import argparse, numpy as np, pandas as pd, json
import tensorflow as tf
from model_hybrid import build_hybrid_model

def load_sequence_from_csv(path):
    df = pd.read_csv(path)
    arr = df.values.astype("float32")
    assert arr.shape[1] == 3, "CSV must have 3 columns: pseudorange,snr,doppler"
    return arr

def generate_synthetic(seq_len=256, spoof=False):
    from data_gen import generate_sample
    return generate_sample(seq_len, spoof=spoof)

def main(args):
    # Load model
    model = tf.keras.models.load_model(args.model_path)
    # Load or create sequence
    if args.csv:
        seq = load_sequence_from_csv(args.csv)
    else:
        seq = generate_synthetic(seq_len=args.seq_len, spoof=args.spoof)
    seq = seq.astype("float32")[None, ...]  # (1, T, 3)
    prob = float(model.predict(seq, verbose=0).ravel()[0])
    pred = int(prob > args.threshold)
    out = {"prob_spoof": prob, "pred_label": pred, "threshold": args.threshold}
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default="artifacts/model.keras")
    p.add_argument("--csv", type=str, default=None, help="Path to CSV with pseudorange,snr,doppler")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic generation instead of CSV")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--spoof", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()
    if args.synthetic and args.csv:
        raise SystemExit("Use either --csv or --synthetic, not both.")
    if not args.synthetic and not args.csv:
        args.synthetic = True
    main(args)
