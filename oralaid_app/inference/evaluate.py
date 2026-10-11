"""
Evaluate the production pipeline on specialist-labelled cases.

    python -m inference.evaluate --patients 300

Uses data/original/patient_<No.>/ images and the "Specialist diagnosis"
column of data/metadata.csv. Prints AUC and operating-point metrics so the
numbers in registry.py can be refreshed after retraining.
"""
import argparse
import csv
import json
import os
import random

import numpy as np
from sklearn.metrics import roc_auc_score

from .engine import HIGH_RISK_THRESHOLD, SUSPICION_THRESHOLD, InferenceEngine, load_image
from .registry import REPO_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=300)
    ap.add_argument("--images-per-patient", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    data = REPO_ROOT / "data"
    labels = {
        r["No."]: int(r["Specialist diagnosis"] == "Suspicious")
        for r in csv.DictReader(open(data / "metadata.csv"))
        if r["Specialist diagnosis"] in ("Suspicious", "Non suspicious")
    }
    patients = [p for p in os.listdir(data / "original") if p.split("_")[1] in labels]
    random.Random(args.seed).shuffle(patients)

    engine = InferenceEngine()
    probs, ys = [], []
    for p in patients[: args.patients]:
        for f in sorted(os.listdir(data / "original" / p))[: args.images_per_patient]:
            prob, _ = engine.predict_proba(load_image(data / "original" / p / f))
            probs.append(prob)
            ys.append(labels[p.split("_")[1]])
    probs, ys = np.array(probs), np.array(ys)

    pred = probs >= SUSPICION_THRESHOLD
    tp, tn = int((pred & (ys == 1)).sum()), int((~pred & (ys == 0)).sum())
    fp, fn = int((pred & (ys == 0)).sum()), int((~pred & (ys == 1)).sum())
    report = {
        "n": int(len(ys)),
        "positives": int(ys.sum()),
        "auc": round(float(roc_auc_score(ys, probs)), 3),
        "threshold": SUSPICION_THRESHOLD,
        "sensitivity": round(tp / max(tp + fn, 1), 3),
        "specificity": round(tn / max(tn + fp, 1), 3),
        "accuracy": round((tp + tn) / len(ys), 3),
        "ppv": round(tp / max(tp + fp, 1), 3),
        "npv": round(tn / max(tn + fn, 1), 3),
        "high_risk_ppv": round(float(ys[probs >= HIGH_RISK_THRESHOLD].mean()) if (probs >= HIGH_RISK_THRESHOLD).any() else 0, 3),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
