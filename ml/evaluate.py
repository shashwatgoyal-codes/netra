"""Evaluate the exported ONNX model exactly as the app will run it.

Every image goes raw PNG -> crop_retina -> bilinear resize -> normalize, the same
steps web/lib/preprocess.ts performs, so these numbers describe the shipped model.

1. Pick the referral threshold on the validation split: the highest threshold
   at which the app's referral rule (`refer_decision`) reaches
   TARGET_SENSITIVITY (missing disease is the costly error in screening).
2. Report grading and referral metrics on the held-out test split.
3. Write metrics, model-meta and example heatmaps to artifacts/<model>/<variant>/.

Usage: uv run python evaluate.py --model dinov3 --variant f16w
"""

import argparse
import json
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm

from common import (
    ARTIFACTS,
    DARK_THRESHOLD,
    GRADES,
    INPUT_SIZE,
    MEAN,
    DATA,
    REFERABLE_FROM,
    SPLITS,
    STD,
    crop_retina,
    to_tensor_array,
)

TARGET_SENSITIVITY = 0.90


def run(sess: ort.InferenceSession, frame: pd.DataFrame) -> tuple[np.ndarray, list[np.ndarray]]:
    """Infer on the original files listed in splits.csv (raw, relative to DATA)."""
    return run_paths(sess, [DATA / raw for raw in frame.raw])


def run_paths(sess: ort.InferenceSession, paths: list) -> tuple[np.ndarray, list[np.ndarray]]:
    probs, cams = [], []
    for path in tqdm(paths, desc="infer"):
        img = crop_retina(Image.open(path))
        p, c = sess.run(None, {"image": to_tensor_array(img)[None]})
        probs.append(p[0])
        cams.append(c[0])
    return np.stack(probs), cams


def refer_decision(probs: np.ndarray, threshold: float) -> np.ndarray:
    """The app's rule (web/lib/model.ts): refer when the referable probability
    reaches the threshold, or when the most likely grade is itself referable."""
    return (probs[:, REFERABLE_FROM:].sum(1) >= threshold) | (probs.argmax(1) >= REFERABLE_FROM)


def pick_threshold(probs: np.ndarray, is_referable: np.ndarray) -> float:
    """Highest threshold at which the full rule reaches TARGET_SENSITIVITY."""
    for t in np.sort(np.unique(probs[:, REFERABLE_FROM:].sum(1)))[::-1]:
        if refer_decision(probs, t)[is_referable].mean() >= TARGET_SENSITIVITY:
            return float(t)
    return 0.0


def referral_stats(pred: np.ndarray, is_referable: np.ndarray) -> dict:
    tp = int((pred & is_referable).sum())
    tn = int((~pred & ~is_referable).sum())
    fp = int((pred & ~is_referable).sum())
    fn = int((~pred & is_referable).sum())
    return {
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "ppv": tp / (tp + fp) if tp + fp else 0.0,
        "npv": tn / (tn + fn) if tn + fn else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def heatmap_examples(paths, labels, probs, cams, path) -> None:
    picks = [i for g in range(len(GRADES)) for i in np.nonzero(labels == g)[0][:1]]
    fig, axes = plt.subplots(2, len(picks), figsize=(3.2 * len(picks), 6.6))
    for col, i in enumerate(picks):
        img = crop_retina(Image.open(paths[i])).resize((INPUT_SIZE, INPUT_SIZE))
        pred = int(probs[i].argmax())
        cam = np.maximum(cams[i][pred], 0)
        cam = cam / cam.max() if cam.max() > 0 else cam
        cam_img = Image.fromarray(np.uint8(cam * 255)).resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"true: {GRADES[labels[i]]}", fontsize=9)
        axes[1, col].imshow(img)
        axes[1, col].imshow(np.asarray(cam_img), cmap="inferno", alpha=0.45)
        axes[1, col].set_title(f"pred: {GRADES[pred]} ({probs[i][pred]:.0%})", fontsize=9)
        for ax in axes[:, col]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", choices=["fp32", "f16w"], default="f16w")
    args = ap.parse_args()
    onnx_file = ARTIFACTS / args.model / f"netra.{args.variant}.onnx".replace(".fp32", "")
    out_dir = ARTIFACTS / args.model / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = pd.read_csv(SPLITS)
    sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])

    val = splits[splits.split == "val"]
    val_probs, _ = run(sess, val)
    threshold = pick_threshold(val_probs, val.diagnosis.to_numpy() >= REFERABLE_FROM)

    test = splits[splits.split == "test"]
    labels = test.diagnosis.to_numpy()
    probs, cams = run(sess, test)
    ref_prob = probs[:, REFERABLE_FROM:].sum(1)
    is_ref = labels >= REFERABLE_FROM
    fpr, tpr, _ = roc_curve(is_ref, ref_prob)
    keep = np.unique(np.linspace(0, len(fpr) - 1, 60).astype(int))

    metrics = {
        "model": args.model,
        "variant": args.variant,
        "modelMB": round(onnx_file.stat().st_size / 1e6, 1),
        "dataset": "APTOS 2019 (Aravind Eye Hospital), held-out test split",
        "train_sources": sorted(splits[splits.split == "train"].get("source", pd.Series(["aptos"])).unique().tolist()),
        "n_test": int(len(test)),
        "n_referable": int(is_ref.sum()),
        "threshold": threshold,
        "target_sensitivity": TARGET_SENSITIVITY,
        "referral": {
            "auc": float(roc_auc_score(is_ref, ref_prob)),
            **referral_stats(refer_decision(probs, threshold), is_ref),
        },
        "grading": {
            "qwk": float(cohen_kappa_score(labels, probs.argmax(1), weights="quadratic")),
            "accuracy": float((probs.argmax(1) == labels).mean()),
            "confusion": confusion_matrix(labels, probs.argmax(1), labels=range(len(GRADES))).tolist(),
        },
        "roc": {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))

    meta = {
        "inputSize": INPUT_SIZE,
        "mean": MEAN,
        "std": STD,
        "darkThreshold": DARK_THRESHOLD,
        "grades": GRADES,
        "referableFrom": REFERABLE_FROM,
        "threshold": threshold,
        "camSize": list(cams[0].shape[1:]),
        "version": f"{args.model}-{args.variant}-{date.today():%Y%m%d}",
    }
    (out_dir / "model-meta.json").write_text(json.dumps(meta, indent=1))
    heatmap_examples([DATA / r for r in test.raw], labels, probs, cams, out_dir / "heatmaps.png")

    r, g = metrics["referral"], metrics["grading"]
    print(f"threshold {threshold:.3f} (chosen on val for sensitivity >= {TARGET_SENSITIVITY:.0%})")
    print(f"test referral: AUC {r['auc']:.3f}  sens {r['sensitivity']:.3f}  spec {r['specificity']:.3f}  "
          f"PPV {r['ppv']:.3f}  NPV {r['npv']:.3f}")
    print(f"test grading: QWK {g['qwk']:.3f}  acc {g['accuracy']:.3f}")


if __name__ == "__main__":
    main()
