
import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from data_gen import build_dataset
from model_hybrid import build_hybrid_model

def main(args):
    os.makedirs("artifacts", exist_ok=True)
    # Data
    X, y = build_dataset(n_samples=args.samples, seq_len=args.seq_len, spoof_frac=args.spoof_frac, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    # Model
    model = build_hybrid_model(seq_len=args.seq_len, n_features=X.shape[-1],
                               conv_filters=args.conv_filters,
                               conv_kernel=args.conv_kernel,
                               lstm_units=args.lstm_units,
                               num_heads=args.num_heads,
                               key_dim=args.key_dim,
                               ff_dim=args.ff_dim,
                               dropout=args.dropout)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2
    )

    # Evaluate
    y_proba = model.predict(X_test, batch_size=args.batch_size).ravel()
    y_pred = (y_proba > 0.5).astype(int)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, digits=3)

    # Save model and metrics
    model.save("artifacts/model.keras")
    with open("artifacts/history.json","w") as f:
        json.dump({k:[float(x) for x in v] for k,v in history.history.items()}, f, indent=2)
    with open("artifacts/metrics.json","w") as f:
        json.dump({"auc":float(auc)}, f, indent=2)

    # ROC plot
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1],"--", linewidth=0.8)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC - Hybrid Demo"); plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("artifacts/roc_plot.png")
    plt.close()

    # Save example sequence
    np.savetxt("artifacts/example_sequence.csv", X_test[0], delimiter=",", header="pseudorange,snr,doppler", comments="")

    # Save report text
    with open("artifacts/summary.txt","w") as f:
        f.write(f"Hybrid CNN→LSTM→Transformer Demo\\nSamples: {args.samples}, Seq_len: {args.seq_len}\\n"
                f"AUC: {auc:.4f}\\n\\nClassification Report:\\n{report}\\n")

    print("Training complete. Artifacts saved in ./artifacts")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--spoof-frac", type=float, default=0.35)
    p.add_argument("--conv-filters", type=int, default=64)
    p.add_argument("--conv-kernel", type=int, default=5)
    p.add_argument("--lstm-units", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--key-dim", type=int, default=32)
    p.add_argument("--ff-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    args = p.parse_args()
    main(args)
