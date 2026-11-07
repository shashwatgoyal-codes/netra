# Netra — offline diabetic retinopathy screening

**Try it:** https://netra-three-gamma.vercel.app (open on an Android phone in Chrome and choose *Install app*; after the first visit it works offline) · **Demo video:** [`demo/netra-demo.mp4`](demo/netra-demo.mp4)

Built on a real problem from Smart India Hackathon 2026: statement **SIH26038** (MathWorks), *Explainable AI for Diabetic Retinopathy Screening in Rural India.*

India has over 77 million adults with diabetes and roughly one ophthalmologist per 100,000 rural people. Early detection prevents most vision loss from diabetic retinopathy (DR), but screening needs specialists who aren't there. Netra lets a health worker photograph the retina, get a referral decision with an explanation in seconds, and track the referral. Everything runs on a low-cost phone, with no internet connection.

## What it does

- **Grades DR on the standard 5-level scale** (none, mild, moderate, severe, proliferative) and turns that into a **refer / don't-refer decision**. "Referable" means moderate or worse, the threshold used by international guidance.
- **Shows why.** A heatmap highlights the retina regions that drove the grade, so the health worker and the specialist can check the model's reasoning. It marks regions, not individual lesions (a 12×12 map stretched over the photo).
- **Runs fully offline.** The model (ONNX Runtime Web, WASM) runs inside the installed web app. Photos and results stay on the device (IndexedDB).
- **Checks photo quality** (blur, framing, exposure) before analysis.
- **Tracks referrals:** a queue sorted by urgency, a status workflow (needs referral → referred → seen), notes, and CSV export for programme reporting.
- **Model card in the app**: sensitivity, specificity, ROC curve, confusion matrix and external results for the exact model file shipped.

## Results

The shipped model is DINOv3 ConvNeXt-Tiny fine-tuned on **APTOS 2019** (Aravind Eye Hospital, India) and **DDR** (147 hospitals, China): 9,189 training photos. Every number below comes from the 56 MB file the app runs, through the app's own preprocessing and referral rule. The referral threshold was set once, on validation, to catch at least 90% of referable disease, and was never retuned for any test set.

| Test set | Photos (referable) | Sensitivity | Specificity | AUC | Grading κ |
|---|---|---|---|---|---|
| APTOS held-out test (India) | 367 (149) | **91.3%** | 91.7% | 0.977 | 0.900 |
| DDR test (China)¹ | 3,759 (1,690) | **73.8%** | 95.1% | 0.941 | 0.778 |
| **IDRiD (India), never seen** | 516 (323) | **84.8%** | 82.9% | 0.935 | 0.721 |
| **Messidor-2 (France), never seen**² | 1,744 (457) | **89.7%** | 69.1% | 0.894 | 0.541 |

¹ DDR's train split is in training, so this row is held-out photos from hospitals the model has seen, not a new site.
² Graded by Google's adjudication panel.

**Why two sources.** The first model trained on APTOS alone scored 95.3% sensitivity on its own test set, then caught **37%** of referable eyes at DDR's hospitals and **34%** on Messidor-2. Almost all the misses were moderate disease on unfamiliar cameras. Adding DDR traded a few points on APTOS for reliability everywhere else:

| Sensitivity | APTOS only | APTOS + DDR |
|---|---|---|
| APTOS test | 95.3% | 91.3% |
| DDR test | 37.0% | 73.8% |
| IDRiD | 80.8% | 84.8% |
| Messidor-2 | 33.9% | 89.7% |

The cost is more false referrals at some sites (69% specificity on Messidor-2). For a screening tool that is the safer way to be wrong.

## Repository

```
ml/    training, evaluation and export (Python, PyTorch, timm, ONNX)
web/   the clinic app (Next.js static export, installable PWA)
```

### ML pipeline

| Step | Script | Notes |
|---|---|---|
| Download | `download.py` | APTOS 2019 via the Kaggle API; DDR, IDRiD and Messidor-2 from their Kaggle mirrors |
| Prepare | `prepare.py` | crop black border, square-pad, cache; APTOS split 80/10/10, DDR keeps its own train/valid |
| Train | `train.py --model dinov3 [--resume]` | bf16 autocast on Apple GPU (MPS); full checkpoint every epoch |
| Export | `export.py --model dinov3` | ONNX with probabilities plus exact class activation maps; fp16 weight storage |
| Evaluate | `evaluate.py --model dinov3` | APTOS test through the app's preprocessing and referral rule; sets the threshold on validation |
| External | `external.py --model dinov3` | DDR test, IDRiD, Messidor-2, with no retuning |
| Ship | `ship.py --model dinov3` | copies the model into `web/public/model` and bumps the offline cache |

`run_all.sh [epochs] [models]` chains prepare → train → export → evaluate → external. `pause.sh` freezes a run in place and `pause.sh resume` continues it.

**Backbone.** DINOv3 ConvNeXt-Tiny (Meta, 2025), distilled from a 7B self-supervised model trained on 1.7 billion images. ConvNeXt V2-Nano was also set up for comparison, but was dropped once it was clear the problem was training data, not architecture. Its weights are also licensed non-commercial only.

**Explainability without gradients.** The backbone ends in global-average-pool → LayerNorm → linear. A logit can be rewritten exactly as the spatial mean of `Σₖ W[c,k]·γₖ·(Aₖ(h,w) − μ)/σ`, so that map is an exact class activation map. The phone gets it from the same single forward pass. `export.py` asserts that the decomposition reproduces the logits.

**Why fp16 weights, not int8.** Static and dynamic int8 quantization both cost ConvNeXt a lot of accuracy. Storing weights in fp16 and casting to fp32 when the model loads halves the download and changes probabilities by under 0.0001.

**Browser and Python agree exactly.** The app reimplements PIL's bilinear downscale instead of relying on canvas smoothing, which drifted predictions by up to 10 points. On real test photos, the app and `evaluate.py` give the same grade and the same referral decision.

### Run it

```bash
# ML (needs ~/.kaggle/kaggle.json and the APTOS 2019 rules accepted on Kaggle)
cd ml
uv sync
uv run python download.py
./run_all.sh 8 dinov3
uv run python ship.py --model dinov3

# App
cd ../web
pnpm install          # also copies the ONNX Runtime files into public/ort
pnpm dev              # or: pnpm build → static site in out/, host anywhere over HTTPS
```

On an Android phone, open the site in Chrome and choose **Install app**. After the first load it works offline.

## Limits

Netra is a screening aid, not a diagnosis, and it has not been clinically validated. Performance varies by site: from 74% to 91% sensitivity across the four test sets above, with specificity as low as 69%. Any clinic using it should validate it on its own camera and population first. It does not detect other eye diseases and does not grade macular edema separately. A trained grader or ophthalmologist must confirm every referral.

Data: APTOS 2019 (Kaggle competition data), DDR (Li et al., 2019), IDRiD (CC BY 4.0), Messidor-2 with Google's adjudicated grades (CC0).
