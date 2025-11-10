#!/usr/bin/env python3
# scripts/attack_fgsm.py
"""
Generate FGSM / PGD adversarial examples using a PyTorch surrogate and ART,
then evaluate against a saved XGBoost baseline.

Outputs:
 - results/attacks_{attack}_seed{seed}.csv   (summary metrics)
 - results/predictions_{attack}_seed{seed}.csv (per-sample before/after)
 - figures/fgsm_roc_seed{seed}.png           (ROC before vs after)
"""

import argparse, os, time, joblib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve)
from sklearn.model_selection import train_test_split

# PyTorch surrogate
import torch
import torch.nn as nn
import torch.optim as optim

# ART
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

warnings.filterwarnings("ignore")

# --------------------
# Simple MLP surrogate
# --------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 2)  # binary -> 2 logits
        )

    def forward(self, x):
        return self.net(x)

# --------------------
# Utility functions
# --------------------
def sanitize_df(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df

def compute_metrics(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }

# --------------------
# Main
# --------------------
def run(seed, model_path="models/xgb_seed42.joblib", data_path="data/processed/cicids2017_clean.parquet",
        subsample_frac=0.05, attack_type="fgsm", eps=0.1, epochs=10, batch_size=512):
    np.random.seed(seed)
    torch.manual_seed(seed)

    OUT = Path("results"); OUT.mkdir(exist_ok=True)
    FIGS = Path("figures"); FIGS.mkdir(exist_ok=True)
    MODELS = Path("models")

    # Load baseline model (XGBoost sklearn API)
    print("Loading XGBoost model:", model_path)
    xgb_model = joblib.load(model_path)

    # Load data
    print("Loading dataset:", data_path)
    df = pd.read_parquet(data_path)
    df = sanitize_df(df)

    # Subsample for speed if requested
    if subsample_frac is not None and 0 < subsample_frac < 1:
        df = df.sample(frac=subsample_frac, random_state=seed).reset_index(drop=True)
        print(f"Subsampled to {len(df)} rows (frac={subsample_frac})")

    label_col = "Label"
    X = df.drop(columns=[label_col])
    y = df[label_col].astype(int).values

    # Drop constant columns if any
    nunique = X.nunique()
    const_cols = list(nunique[nunique <= 1].index)
    if const_cols:
        print("Dropping constant columns:", const_cols)
        X = X.drop(columns=const_cols)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    # Scale features
    scaler = StandardScaler().fit(X_train.values)
    X_train_s = scaler.transform(X_train.values).astype(np.float32)
    X_test_s = scaler.transform(X_test.values).astype(np.float32)

    # ---------------
    # Train surrogate (PyTorch MLP)
    # ---------------
    input_dim = X_train_s.shape[1]
    surrogate = SimpleMLP(input_dim, hidden=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surrogate.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(surrogate.parameters(), lr=1e-3)

    # Wrap surrogate with ART PyTorchClassifier
    surrogate_clf = PyTorchClassifier(
        model=surrogate,
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=(input_dim,),
        nb_classes=2,
        device_type="gpu" if torch.cuda.is_available() else "cpu"
    )

    # Train surrogate for a few epochs
    print("Training surrogate MLP for", epochs, "epochs on device", device)
    X_train_tensor = torch.from_numpy(X_train_s)
    y_train_tensor = torch.from_numpy(y_train.astype(np.int64))

    surrogate.train()
    n = X_train_s.shape[0]
    for ep in range(epochs):
        idx = np.random.permutation(n)
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i+batch_size]
            xb = X_train_tensor[batch_idx].to(device)
            yb = y_train_tensor[batch_idx].to(device)
            optimizer.zero_grad()
            out = surrogate(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
        if (ep+1) % max(1, epochs//5) == 0:
            print(f" Surrogate epoch {ep+1}/{epochs} loss {float(loss):.4f}")

    surrogate.eval()

    # ---------------
    # Evaluate baseline on clean test
    # ---------------
    y_score_before = xgb_model.predict_proba(X_test)[:, 1]
    y_pred_before = (y_score_before >= 0.5).astype(int)
    metrics_before = compute_metrics(y_test, y_pred_before, y_score_before)
    print("Baseline (clean) metrics:", metrics_before)

    # ---------------
    # Create ART attack (FGSM or PGD) on surrogate
    # ---------------
    attack_type = attack_type.lower()
    if attack_type == "fgsm":
        attack = FastGradientMethod(estimator=surrogate_clf, eps=eps)
    elif attack_type == "pgd":
        attack = ProjectedGradientDescent(estimator=surrogate_clf, norm=np.inf, eps=eps, max_iter=40, eps_step=eps/10)
    else:
        raise ValueError("Unsupported attack_type (choose 'fgsm' or 'pgd')")

    # Generate adversarial examples from test set (use numpy arrays)
    print(f"Generating adversarial examples with {attack_type} eps={eps} ...")
    X_test_adv_s = attack.generate(x=X_test_s, y=y_test.astype(np.int64))
    # convert back to unscaled space for XGBoost evaluation
    X_test_adv = scaler.inverse_transform(X_test_adv_s)

    # Sanity clamp: keep numeric ranges sane (optional)
    # X_test_adv = np.clip(X_test_adv, X.min().values, X.max().values)

    # ---------------
    # Evaluate baseline on adversarial test
    # ---------------
    y_score_after = xgb_model.predict_proba(X_test_adv)[:, 1]
    y_pred_after = (y_score_after >= 0.5).astype(int)
    metrics_after = compute_metrics(y_test, y_pred_after, y_score_after)
    print("Metrics after attack:", metrics_after)

    # ---------------
    # Compute Attack Success Rate (ASR)
    # define ASR = fraction of malicious samples (y==1) that were correctly detected before,
    # but misclassified as benign after the attack (i.e., attacker succeeds to evade detection).
    # ---------------
    mask_mal = (y_test == 1)
    if mask_mal.sum() > 0:
        correct_before_mask = (y_pred_before == 1) & mask_mal
        denom = correct_before_mask.sum()
        if denom > 0:
            evaded = ((y_pred_after == 0) & correct_before_mask).sum()
            asr = float(evaded) / int(denom)
        else:
            asr = float("nan")
    else:
        asr = float("nan")

    # ---------------
    # Save results
    # ---------------
    stamp = int(time.time())
    summary = {
        "seed": int(seed),
        "attack": attack_type,
        "eps": float(eps),
        "n_test": int(len(y_test)),
        "n_malicious_test": int(mask_mal.sum()),
        "roc_auc_before": metrics_before["roc_auc"],
        "roc_auc_after": metrics_after["roc_auc"],
        "accuracy_before": metrics_before["accuracy"],
        "accuracy_after": metrics_after["accuracy"],
        "asr": asr
    }

    out_csv = OUT / f"attacks_{attack_type}_seed{seed}.csv"
    pd.DataFrame([summary]).to_csv(out_csv, index=False)
    print("Wrote summary ->", out_csv)

    # per-sample predictions
    preds_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred_before": y_pred_before, "y_score_before": y_score_before,
        "y_pred_after": y_pred_after, "y_score_after": y_score_after
    })
    preds_csv = OUT / f"predictions_{attack_type}_seed{seed}.csv"
    preds_df.to_csv(preds_csv, index=False)
    print("Wrote per-sample preds ->", preds_csv)

    # ROC plot
    import matplotlib.pyplot as plt
    fpr_b, tpr_b, _ = roc_curve(y_test, y_score_before)
    fpr_a, tpr_a, _ = roc_curve(y_test, y_score_after)
    plt.figure(figsize=(6,4))
    plt.plot(fpr_b, tpr_b, label=f'Before ROC AUC={metrics_before["roc_auc"]:.4f}')
    plt.plot(fpr_a, tpr_a, label=f'After ROC AUC={metrics_after["roc_auc"]:.4f}')
    plt.plot([0,1],[0,1],'--', linewidth=0.7)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC Before vs After ({attack_type}, eps={eps})")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout()
    fig_path = FIGS / f"{attack_type}_roc_seed{seed}.png"
    plt.savefig(fig_path, dpi=200); plt.close()
    print("Saved ROC figure ->", fig_path)

    print("Done seed", seed, "attack", attack_type, "eps", eps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", type=str, default="models/xgb_seed42.joblib")
    parser.add_argument("--data", type=str, default="data/processed/cicids2017_clean.parquet")
    parser.add_argument("--subsample", type=float, default=0.05)
    parser.add_argument("--attack", type=str, default="fgsm", choices=["fgsm","pgd"])
    parser.add_argument("--eps", type=float, default=0.1, help="L-inf perturbation bound (on scaled features)")
    parser.add_argument("--epochs", type=int, default=10, help="surrogate training epochs")
    args = parser.parse_args()

    run(args.seed, model_path=args.model, data_path=args.data,
        subsample_frac=args.subsample, attack_type=args.attack, eps=args.eps, epochs=args.epochs)
