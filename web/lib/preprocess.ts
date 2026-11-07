// Mirrors ml/common.py: crop_retina -> resize -> normalize. If you change one,
// change the other, or the model will see inputs it was never trained on.

import type { ModelMeta } from "./model";

export type Prepared = {
  tensor: Float32Array; // [3, size, size], normalized, CHW
  preview: HTMLCanvasElement; // the square crop the model saw, for display + storage
  quality: ImageQuality;
};

export type ImageQuality = {
  fundusCoverage: number; // share of the square crop that is retina, not black border
  sharpness: number; // variance of the Laplacian on a 256px grayscale copy
  ok: boolean;
  issues: string[];
};

// Calibrated loosely on APTOS images: sharp, well-exposed fundus photos sit far
// above these; they only catch photos that are clearly unusable.
const MIN_COVERAGE = 0.45;
const MIN_SHARPNESS = 12;

export async function loadImage(file: Blob): Promise<ImageBitmap> {
  return createImageBitmap(file, { imageOrientation: "from-image" });
}

function canvas(w: number, h: number): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}

/** Bounding box of pixels brighter than the background threshold. */
function retinaBox(img: ImageBitmap, darkThreshold: number) {
  const c = canvas(img.width, img.height);
  const ctx = c.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0);
  const { data } = ctx.getImageData(0, 0, img.width, img.height);
  let x0 = img.width,
    y0 = img.height,
    x1 = -1,
    y1 = -1;
  for (let y = 0; y < img.height; y++) {
    for (let x = 0; x < img.width; x++) {
      const i = (y * img.width + x) * 4;
      if ((data[i] + data[i + 1] + data[i + 2]) / 3 > darkThreshold) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  if (x1 < 0) return { x: 0, y: 0, w: img.width, h: img.height };
  return { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 };
}

function assessQuality(square: HTMLCanvasElement, darkThreshold: number): ImageQuality {
  const n = 256;
  const small = canvas(n, n);
  const ctx = small.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(square, 0, 0, n, n);
  const { data } = ctx.getImageData(0, 0, n, n);
  const gray = new Float32Array(n * n);
  let bright = 0;
  for (let i = 0; i < n * n; i++) {
    gray[i] = (data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2]) / 3;
    if (gray[i] > darkThreshold) bright++;
  }
  // Laplacian variance over retina pixels only, so the black border doesn't
  // inflate it with a sharp circular edge.
  let sum = 0,
    sumSq = 0,
    count = 0;
  for (let y = 2; y < n - 2; y++) {
    for (let x = 2; x < n - 2; x++) {
      const i = y * n + x;
      if (gray[i - 2] <= darkThreshold || gray[i + 2] <= darkThreshold) continue;
      if (gray[i - 2 * n] <= darkThreshold || gray[i + 2 * n] <= darkThreshold) continue;
      const lap = 4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - n] - gray[i + n];
      sum += lap;
      sumSq += lap * lap;
      count++;
    }
  }
  const sharpness = count ? sumSq / count - (sum / count) ** 2 : 0;
  const fundusCoverage = bright / (n * n);
  const issues: string[] = [];
  if (fundusCoverage < MIN_COVERAGE) issues.push("The retina fills too little of the photo, or the photo is too dark.");
  if (sharpness < MIN_SHARPNESS) issues.push("The photo looks blurred or out of focus.");
  return { fundusCoverage, sharpness, ok: issues.length === 0, issues };
}

/**
 * Square RGBA resize matching PIL's Image.resize(..., BILINEAR), which the model
 * was evaluated with. When shrinking, PIL widens the triangle filter by the scale
 * factor so every source pixel contributes; canvas drawImage smoothing is
 * browser-defined and drifted the model's output by up to 10 points.
 */
function resizeLikePil(src: Uint8ClampedArray, from: number, to: number): Uint8ClampedArray<ArrayBuffer> {
  const scale = from / to;
  const support = Math.max(scale, 1);
  // Per output coordinate: first source index and normalized weights.
  const starts = new Int32Array(to);
  const weights: Float32Array[] = [];
  for (let i = 0; i < to; i++) {
    const center = (i + 0.5) * scale;
    const lo = Math.max(0, Math.floor(center - support + 0.5));
    const hi = Math.min(from, Math.floor(center + support + 0.5));
    const w = new Float32Array(hi - lo);
    let total = 0;
    for (let j = lo; j < hi; j++) {
      const t = Math.max(0, 1 - Math.abs((j - center + 0.5) / support));
      w[j - lo] = t;
      total += t;
    }
    for (let k = 0; k < w.length; k++) w[k] /= total || 1;
    starts[i] = lo;
    weights.push(w);
  }

  // Horizontal pass (from x from -> to x from), rounded to 8 bits like PIL.
  const mid = new Uint8ClampedArray(to * from * 4);
  for (let y = 0; y < from; y++) {
    for (let x = 0; x < to; x++) {
      const w = weights[x];
      let r = 0,
        g = 0,
        b = 0;
      let p = (y * from + starts[x]) * 4;
      for (let k = 0; k < w.length; k++, p += 4) {
        r += src[p] * w[k];
        g += src[p + 1] * w[k];
        b += src[p + 2] * w[k];
      }
      const o = (y * to + x) * 4;
      mid[o] = Math.round(r);
      mid[o + 1] = Math.round(g);
      mid[o + 2] = Math.round(b);
    }
  }
  // Vertical pass.
  const out = new Uint8ClampedArray(new ArrayBuffer(to * to * 4));
  for (let y = 0; y < to; y++) {
    const w = weights[y];
    for (let x = 0; x < to; x++) {
      let r = 0,
        g = 0,
        b = 0;
      let p = (starts[y] * to + x) * 4;
      for (let k = 0; k < w.length; k++, p += to * 4) {
        r += mid[p] * w[k];
        g += mid[p + 1] * w[k];
        b += mid[p + 2] * w[k];
      }
      const o = (y * to + x) * 4;
      out[o] = Math.round(r);
      out[o + 1] = Math.round(g);
      out[o + 2] = Math.round(b);
      out[o + 3] = 255;
    }
  }
  return out;
}

export function prepare(img: ImageBitmap, meta: ModelMeta): Prepared {
  const box = retinaBox(img, meta.darkThreshold);
  const side = Math.max(box.w, box.h);
  const square = canvas(side, side);
  const sctx = square.getContext("2d")!;
  sctx.fillStyle = "#000";
  sctx.fillRect(0, 0, side, side);
  sctx.drawImage(img, box.x, box.y, box.w, box.h, Math.floor((side - box.w) / 2), Math.floor((side - box.h) / 2), box.w, box.h);

  const size = meta.inputSize;
  const full = sctx.getImageData(0, 0, side, side).data;
  const data = resizeLikePil(full, side, size);
  const input = canvas(size, size);
  input.getContext("2d")!.putImageData(new ImageData(data, size, size), 0, 0);

  const plane = size * size;
  const tensor = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      tensor[c * plane + i] = (data[i * 4 + c] / 255 - meta.mean[c]) / meta.std[c];
    }
  }
  return { tensor, preview: input, quality: assessQuality(square, meta.darkThreshold) };
}
