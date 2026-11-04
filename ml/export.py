"""Export a trained model to ONNX for in-browser inference.

The exported graph returns two tensors:
  probs  [B, 5]        softmax over DR grades
  cams   [B, 5, h, w]  class activation maps, one per grade

Both candidate backbones end in  global-avg-pool -> LayerNorm -> linear.  With
p = mean_hw(A), mu/sigma the LayerNorm statistics of p, and g the LayerNorm
scale, every logit decomposes exactly as

  logit_c = mean_hw[ sum_k W[c,k] * g_k * (A_k(h,w) - mu) / sigma ] + const_c

so the bracketed map is an exact class activation map (Zhou et al., 2016): its
spatial mean is the logit, up to a per-class constant. The phone draws the
heatmap from a single forward pass, with no gradients.

Shipping format: weights stored as fp16 and cast back to fp32 when the session
loads (netra.f16w.onnx). That halves the download while every op still runs in
fp32, so it behaves the same on any phone's CPU (WASM) or GPU (WebGPU). Static
and dynamic int8 were tried first; ConvNeXt's LayerNorm-heavy blocks lost too
much accuracy under both.

Usage: uv run python export.py --model dinov3
Reads artifacts/<model>/best.pt, writes netra.onnx (fp32) and netra.f16w.onnx.
"""

import argparse

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from onnx import TensorProto, helper, numpy_helper
from PIL import Image
from torch import nn

from common import ARTIFACTS, INPUT_SIZE, PROCESSED, SPLITS, to_tensor_array
from train import MODELS, build_model

MIN_FP16_ELEMENTS = 1024  # small tensors (biases, norms) stay fp32


class ScreeningHead(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        head = self.net.head
        features = self.net.forward_features(x)  # [B, C, h, w]
        pooled = head.global_pool(features)  # [B, C, 1, 1]
        probs = head.fc(head.flatten(head.norm(pooled))).softmax(1)

        mu = pooled.mean(1, keepdim=True)
        sigma = (pooled.var(1, unbiased=False, keepdim=True) + head.norm.eps).sqrt()
        scaled = head.fc.weight * head.norm.weight  # [5, C]
        cams = torch.einsum("kc,bchw->bkhw", scaled, (features - mu) / sigma)
        return probs, cams


def store_weights_fp16(src, dst) -> None:
    model = onnx.load(str(src))
    casts = []
    for init in model.graph.initializer:
        if init.data_type != TensorProto.FLOAT or np.prod(init.dims) < MIN_FP16_ELEMENTS:
            continue
        name = init.name
        init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), f"{name}__f16"))
        casts.append(helper.make_node("Cast", [f"{name}__f16"], [name], to=TensorProto.FLOAT, name=f"cast_{name}"))
    for node in reversed(casts):
        model.graph.node.insert(0, node)
    onnx.save(model, str(dst))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    args = ap.parse_args()
    out_dir = ARTIFACTS / args.model

    ckpt = torch.load(out_dir / "best.pt", map_location="cpu")
    net = build_model(args.model, pretrained=False)
    net.load_state_dict(ckpt["model"])
    model = ScreeningHead(net).eval()

    fp32 = out_dir / "netra.onnx"
    torch.onnx.export(
        model,
        (torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE),),
        fp32,
        input_names=["image"],
        output_names=["probs", "cams"],
        dynamic_axes={"image": {0: "batch"}, "probs": {0: "batch"}, "cams": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    splits = pd.read_csv(SPLITS)
    # The CAM decomposition must reproduce the model's own logits.
    sample_id = splits.query("split == 'val'").id_code.iloc[0]
    x = to_tensor_array(Image.open(PROCESSED / f"{sample_id}.jpg"))[None]
    with torch.no_grad():
        ref_probs, ref_cams = (t.numpy() for t in model(torch.from_numpy(x)))
        logits = net(torch.from_numpy(x)).numpy()
    implied = ref_cams.mean((2, 3)) + (net.head.fc.weight @ net.head.norm.bias + net.head.fc.bias).detach().numpy()
    assert np.allclose(implied, logits, atol=1e-3), "CAM decomposition does not match logits"

    sess = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    probs, cams = sess.run(None, {"image": x})
    assert np.allclose(probs, ref_probs, atol=1e-4), "probs mismatch between PyTorch and ONNX"
    assert np.allclose(cams, ref_cams, atol=1e-2), "cams mismatch between PyTorch and ONNX"
    print(f"fp32 {fp32.stat().st_size / 1e6:.0f} MB, cams {cams.shape[2:]}, parity ok")

    small = out_dir / "netra.f16w.onnx"
    store_weights_fp16(fp32, small)
    half = ort.InferenceSession(str(small), providers=["CPUExecutionProvider"])
    half_probs, _ = half.run(None, {"image": x})
    print(f"f16w {small.stat().st_size / 1e6:.0f} MB, max prob diff vs fp32 {np.abs(half_probs - probs).max():.5f}")


if __name__ == "__main__":
    main()
