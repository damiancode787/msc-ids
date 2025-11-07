#!/usr/bin/env python3
# scripts/train_xgb.py
import argparse, time, random, joblib, os
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, average_precision_score)
import xgboost as xgb

def run(seed, data_path="data/processed/cicids2017_clean.parquet", subsample_frac=None):
    np.random.seed(seed)
    random.seed(seed)
    MODELS_DIR = Path("models"); MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)
    FIGS_DIR = Path("figures"); FIGS_DIR.mkdir(exist_ok=True)

    # Load data
    df = pd.read_parquet(data_path)

    # ---- SANITISE ----
    df = df.replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
    df = df.fillna(0)  # Replace remaining NaN with 0 (safe for XGBoost)
    print("Cleaned infinities and NaN values")

    if subsample_frac:
        df = df.sample(frac=subsample_frac, random_state=seed)

    label_col = "Label"
    X = df.drop(columns=[label_col])
    y = df[label_col].astype(int)

    # drop constant cols
    nunique = X.nunique()
    const_cols = list(nunique[nunique <= 1].index)
    if const_cols:
        X = X.drop(columns=const_cols)

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    # model
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=seed,
        n_jobs=-1
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:,1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics = {
        "seed": seed,
        "model": f"xgb_seed{seed}",
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "pr_auc": float(average_precision_score(y_test, y_score)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "train_time_s": float(train_time)
    }

    res_file = RESULTS_DIR / "baseline_results.csv"
    pd.DataFrame([metrics]).to_csv(res_file, mode='a', header=not res_file.exists(), index=False)

    joblib.dump(model, MODELS_DIR / f"xgb_seed{seed}.joblib")
    pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred, "y_score": y_score}).to_csv(RESULTS_DIR / f"predictions_seed{seed}.csv", index=False)

    # Save ROC plot
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt
    fpr, tpr, _ = roc_curve(y_test, y_score)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'ROC AUC={metrics["roc_auc"]:.4f}')
    plt.plot([0,1],[0,1],'--', linewidth=0.7)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC Curve (seed={seed})")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(FIGS_DIR / f"roc_seed{seed}.png", dpi=200); plt.close()

    # Feature importance (gain)
    fi = model.get_booster().get_score(importance_type="gain")
    fi_items = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:30]
    if fi_items:
        names, gains = zip(*fi_items)
        plt.figure(figsize=(6,8))
        plt.barh(range(len(names)), list(reversed(gains)))
        plt.yticks(range(len(names)), list(reversed(names)))
        plt.title(f"Top feature gain (seed={seed})")
        plt.tight_layout()
        plt.savefig(FIGS_DIR / f"fi_seed{seed}.png", dpi=200); plt.close()

    print("Done seed", seed, metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data", type=str, default="data/processed/cicids2017_clean.parquet")
    parser.add_argument("--subsample", type=float, default=None, help="fraction to subsample (0-1)")
    args = parser.parse_args()
    run(args.seed, args.data, subsample_frac=args.subsample)
