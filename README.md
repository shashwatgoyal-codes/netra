# Netra — offline diabetic retinopathy screening

Smart India Hackathon 2026, problem statement **SIH26038** (MathWorks): *Explainable AI for Diabetic Retinopathy Screening in Rural India.*

India has over 77 million adults with diabetes and roughly one ophthalmologist per 100,000 rural people. Early detection prevents most vision loss from diabetic retinopathy (DR), but screening needs specialists who aren't there. Netra lets a health worker photograph the retina, get a referral decision with an explanation in seconds, and track the referral. Everything runs on a low-cost phone, with no internet connection.

## What it does

- **Grades DR on the standard 5-level scale** (none, mild, moderate, severe, proliferative) and turns that into a **refer / don't-refer decision**. "Referable" means moderate or worse, the threshold used by international guidance.
- **Shows why.** A heatmap highlights the retina regions that drove the grade, so the health worker and the specialist can check the model's reasoning.
- **Runs fully offline.** The model (ONNX Runtime Web, WASM) runs inside the installed web app. Photos and results stay on the device (IndexedDB).
- **Checks photo quality** (blur, framing, exposure) before analysis.
- **Tracks referrals:** a queue sorted by urgency, a status workflow (needs referral → referred → seen), notes, and CSV export for programme reporting.
- **Model card in the app**: sensitivity, specificity, ROC curve and confusion matrix for the exact model file shipped.

## Repository

```
ml/    training, evaluation and export (Python, PyTorch, timm, ONNX)
web/   the clinic app (Next.js static export, installable PWA)
```

### ML pipeline

| Step | Script | Notes |
|---|---|---|
| Download | `download.py` | APTOS 2019, 3,662 graded fundus photos from Aravind Eye Hospital |
| Prepare | `prepare.py` | crop black border, square-pad, cache; stratified 80/10/10 split |
| Train | `train.py --model dinov3` | fine-tune with bf16 autocast on Apple GPU (MPS) |
| Export | `export.py --model dinov3` | ONNX with probabilities plus exact class activation maps; fp16 weight storage |
| Evaluate | `evaluate.py --model dinov3` | evaluates the exported file through the app's own preprocessing; sets the referral threshold on validation for ≥90% sensitivity |
| Ship | `ship.py --model dinov3` | copies the model into `web/public/model` and bumps the offline cache |

**Backbones compared.** Both are recent self-supervised ConvNeXt models small enough for a phone:
- **DINOv3 ConvNeXt-Tiny** (Meta, 2025): distilled from a 7B self-supervised model trained on 1.7 billion images.
- **ConvNeXt V2-Nano @384** (2023): FCMAE pretraining.

The better one on the held-out test split ships.

**Explainability without gradients.** Both backbones end in global-average-pool → LayerNorm → linear. A logit can be rewritten exactly as the spatial mean of `Σₖ W[c,k]·γₖ·(Aₖ(h,w) − μ)/σ`, so that map is an exact class activation map. The phone gets it from the same single forward pass. `export.py` asserts that the decomposition reproduces the logits.

**Why fp16 weights, not int8.** Static and dynamic int8 quantization both cost ConvNeXt a lot of accuracy. Storing weights in fp16 and casting to fp32 when the model loads halves the download and keeps predictions identical (max probability difference under 0.001).

### Run it

```bash
# ML (needs ~/.kaggle/kaggle.json and the APTOS 2019 rules accepted on Kaggle)
cd ml
uv sync
uv run python download.py && uv run python prepare.py
uv run python train.py --model dinov3 && uv run python export.py --model dinov3
uv run python evaluate.py --model dinov3 && uv run python ship.py --model dinov3

# App
cd ../web
pnpm install          # also copies the ONNX Runtime files into public/ort
pnpm dev              # or: pnpm build → static site in out/, host anywhere over HTTPS
```

On an Android phone, open the site in Chrome and choose **Install app**. After the first load it works offline.

## Limits

Netra is a screening aid, not a diagnosis, and it has not been clinically validated. It was trained on photos from one hospital network. Other fundus cameras, smartphone adapters and populations need local validation before use. It does not detect other eye diseases and does not grade macular edema separately. A trained grader or ophthalmologist must confirm every referral.
