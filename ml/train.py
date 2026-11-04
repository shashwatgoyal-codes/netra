"""Fine-tune a pretrained ConvNeXt backbone to grade diabetic retinopathy (0-4).

Candidates (both end in global-pool -> LayerNorm -> linear, which export.py
relies on for exact heatmaps):
  dinov3   convnext_tiny.dinov3_lvd1689m  Meta DINOv3 (2025), distilled from a 7B
           self-supervised model trained on 1.7B images; DINOv3 license
  cnv2     convnextv2_nano.fcmae_ft_in22k_in1k_384  ConvNeXt V2 (2023); CC-BY-NC

Usage: uv run python train.py --model dinov3 [--epochs 12] [--batch 16] [--lr 1e-4]
Writes artifacts/<model>/best.pt (best validation score) and history.json.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

from common import ARTIFACTS, GRADES, INPUT_SIZE, MEAN, PROCESSED, REFERABLE_FROM, SPLITS, STD

MODELS = {
    "dinov3": "convnext_tiny.dinov3_lvd1689m",
    "cnv2": "convnextv2_nano.fcmae_ft_in22k_in1k_384",
}


class FundusDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.ids = frame.id_code.tolist()
        self.labels = frame.diagnosis.tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(PROCESSED / f"{self.ids[i]}.jpg").convert("RGB")
        return self.transform(img), self.labels[i]


def transforms(train: bool):
    tail = [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)]
    if not train:
        return T.Compose([T.Resize((INPUT_SIZE, INPUT_SIZE)), *tail])
    return T.Compose(
        [
            T.RandomResizedCrop(INPUT_SIZE, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(180),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            *tail,
        ]
    )


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(key: str, pretrained: bool = True) -> nn.Module:
    return timm.create_model(MODELS[key], pretrained=pretrained, num_classes=len(GRADES), drop_path_rate=0.1)


def autocast(dev: torch.device):
    # bf16 autocast is ~4x faster than fp32 for ConvNeXt on Apple GPUs.
    return torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type in ("mps", "cuda"))


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, dev: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        with autocast(dev):
            logits = model(x.to(dev, memory_format=torch.channels_last))
        probs.append(logits.float().softmax(1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def score(probs: np.ndarray, labels: np.ndarray) -> dict:
    referable_prob = probs[:, REFERABLE_FROM:].sum(1)
    return {
        "qwk": float(cohen_kappa_score(labels, probs.argmax(1), weights="quadratic")),
        "referable_auc": float(roc_auc_score(labels >= REFERABLE_FROM, referable_prob)),
        "accuracy": float((probs.argmax(1) == labels).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    out_dir = ARTIFACTS / args.model

    torch.manual_seed(0)
    dev = device()
    splits = pd.read_csv(SPLITS)
    train_df, val_df = splits[splits.split == "train"], splits[splits.split == "val"]

    loader = lambda df, train: DataLoader(  # noqa: E731
        FundusDataset(df, transforms(train)),
        batch_size=args.batch,
        shuffle=train,
        num_workers=args.workers,
        persistent_workers=True,
        drop_last=train,
    )
    train_dl, val_dl = loader(train_df, True), loader(val_df, False)

    # Inverse-sqrt class weights: lift rare grades (severe, proliferative) without
    # letting them dominate the loss.
    counts = np.bincount(train_df.diagnosis, minlength=len(GRADES))
    weights = torch.tensor((counts.max() / counts) ** 0.5, dtype=torch.float32, device=dev)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    model = build_model(args.model).to(dev, memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    out_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(dev, memory_format=torch.channels_last), y.to(dev)
            with autocast(dev):
                logits = model(x)
            loss = criterion(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()

        metrics = score(*predict(model, val_dl, dev))
        # QWK tracks grading quality, AUC tracks the refer/don't-refer decision
        # the app actually makes; weight them equally.
        combined = (metrics["qwk"] + metrics["referable_auc"]) / 2
        row = {"epoch": epoch, "loss": running / len(train_dl), **metrics, "seconds": time.time() - start}
        history.append(row)
        flag = ""
        if combined > best:
            best = combined
            torch.save({"model": model.state_dict(), "key": args.model, "epoch": epoch, **metrics}, out_dir / "best.pt")
            flag = "  *saved"
        print(
            f"epoch {epoch:2d}  loss {row['loss']:.3f}  qwk {metrics['qwk']:.3f}  "
            f"auc {metrics['referable_auc']:.3f}  acc {metrics['accuracy']:.3f}  {row['seconds']:.0f}s{flag}",
            flush=True,
        )
        (out_dir / "history.json").write_text(json.dumps(history, indent=1))


if __name__ == "__main__":
    main()
