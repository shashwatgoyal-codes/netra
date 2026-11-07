"""Validate the shipped model on datasets outside its training data.

  ddr        DDR test split (147 hospitals, China). External for an APTOS-only
             model; once DDR's train split is in training, it measures held-out
             photos from those same hospitals instead.
  idrid      IDRiD (Nanded, India): all 516 graded photos. Never trained on.
  messidor2  Messidor-2 (France) with Google's adjudicated grades. Never trained on.

Nothing is tuned here: the referral threshold stays the one set on the
training sources' validation split, because the question is what a new
clinic would get on day one. Ungradable photos are excluded.

Writes an "external" list into artifacts/<model>/<variant>/metrics.json,
which ship.py copies into the app's model page.

Usage: uv run python external.py --model dinov3 [--variant f16w] [--datasets ddr idrid messidor2]
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import onnxruntime as ort
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from common import ARTIFACTS, DATA, REFERABLE_FROM
from evaluate import refer_decision, referral_stats, run_paths


def ddr() -> pd.DataFrame:
    root = DATA / "ddr" / "DDR-dataset" / "DR_grading"
    df = pd.read_csv(root / "test.txt", sep=" ", header=None, names=["file", "grade"])
    df = df[df.grade <= 4]
    return pd.DataFrame({"path": [root / "test" / f for f in df.file], "grade": df.grade.to_numpy()})


def idrid() -> pd.DataFrame:
    root = next((DATA / "idrid").rglob("B. Disease Grading"))
    frames = []
    for csv in sorted((root / "2. Groundtruths").glob("*.csv")):
        split = "a. Training Set" if "Training" in csv.name else "b. Testing Set"
        df = pd.read_csv(csv).iloc[:, :2].dropna()
        df.columns = ["name", "grade"]
        frames.append(
            pd.DataFrame(
                {"path": [root / "1. Original Images" / split / f"{n}.jpg" for n in df.name], "grade": df.grade.astype(int).to_numpy()}
            )
        )
    return pd.concat(frames, ignore_index=True)


def messidor2() -> pd.DataFrame:
    root = DATA / "messidor2"
    labels = pd.read_csv(root / "messidor_data.csv")
    labels = labels[labels.adjudicated_gradable == 1].dropna(subset=["adjudicated_dr_grade"])
    files = {p.stem.lower(): p for p in root.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif")}
    paths = [files.get(Path(i).stem.lower()) for i in labels.image_id]
    keep = [p is not None for p in paths]
    return pd.DataFrame({"path": [p for p in paths if p is not None], "grade": labels.adjudicated_dr_grade.astype(int).to_numpy()[keep]})


DATASETS = {
    "ddr": (ddr, "DDR test split (147 hospitals, China)"),
    "idrid": (idrid, "IDRiD (Nanded, India)"),
    "messidor2": (messidor2, "Messidor-2 (France), Google adjudicated grades"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", choices=["fp32", "f16w"], default="f16w")
    ap.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = ap.parse_args()
    report = ARTIFACTS / args.model / args.variant
    metrics = json.loads((report / "metrics.json").read_text())
    onnx_file = ARTIFACTS / args.model / f"netra.{args.variant}.onnx".replace(".fp32", "")
    sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])

    results = {r["key"]: r for r in metrics.get("external", []) if isinstance(metrics.get("external"), list)}
    for key in args.datasets:
        load, name = DATASETS[key]
        try:
            frame = load()
        except (FileNotFoundError, StopIteration):
            print(f"{key}: data not found, skipped")
            continue
        probs, _ = run_paths(sess, frame.path.tolist())
        grades = frame.grade.to_numpy()
        is_ref = grades >= REFERABLE_FROM
        results[key] = {
            "key": key,
            "dataset": name,
            "n": int(len(frame)),
            "n_referable": int(is_ref.sum()),
            "referral": {
                "auc": float(roc_auc_score(is_ref, probs[:, REFERABLE_FROM:].sum(1))),
                **referral_stats(refer_decision(probs, metrics["threshold"]), is_ref),
            },
            "grading": {
                "qwk": float(cohen_kappa_score(grades, probs.argmax(1), weights="quadratic")),
                "accuracy": float((probs.argmax(1) == grades).mean()),
            },
        }
        r = results[key]["referral"]
        print(f"{key} external ({len(frame)} photos, {int(is_ref.sum())} referable): AUC {r['auc']:.3f}  "
              f"sens {r['sensitivity']:.3f}  spec {r['specificity']:.3f}  QWK {results[key]['grading']['qwk']:.3f}")

    metrics["external"] = [results[k] for k in DATASETS if k in results]
    (report / "metrics.json").write_text(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
