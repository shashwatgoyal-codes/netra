// Turns a low-resolution class activation map into a translucent overlay.

// "inferno" colormap stops (matplotlib), so app heatmaps match the training report.
const STOPS: [number, number, number][] = [
  [0, 0, 4],
  [40, 11, 84],
  [101, 21, 110],
  [159, 42, 99],
  [212, 72, 66],
  [245, 125, 21],
  [250, 193, 39],
  [252, 255, 164],
];

function color(t: number): [number, number, number] {
  const x = Math.min(Math.max(t, 0), 1) * (STOPS.length - 1);
  const i = Math.min(Math.floor(x), STOPS.length - 2);
  const f = x - i;
  const [a, b] = [STOPS[i], STOPS[i + 1]];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/** Positive evidence only, scaled to 0..1. */
export function normalizeCam(cam: ArrayLike<number>): Float32Array {
  let max = 0;
  for (let i = 0; i < cam.length; i++) max = Math.max(max, cam[i]);
  const out = new Float32Array(cam.length);
  if (max <= 0) return out;
  for (let i = 0; i < cam.length; i++) out[i] = Math.max(cam[i], 0) / max;
  return out;
}

/** Draws the map at its native size, then lets the browser upscale it smoothly. */
export function renderHeatmap(cam: ArrayLike<number>, [h, w]: [number, number], size: number): HTMLCanvasElement {
  const norm = normalizeCam(cam);
  const small = document.createElement("canvas");
  small.width = w;
  small.height = h;
  const sctx = small.getContext("2d")!;
  const img = sctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const [r, g, b] = color(norm[i]);
    img.data[i * 4] = r;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = b;
    // Weak evidence fades out so the retina stays readable underneath.
    img.data[i * 4 + 3] = Math.round(255 * Math.min(1, norm[i] * 1.4) ** 1.5);
  }
  sctx.putImageData(img, 0, 0);

  const out = document.createElement("canvas");
  out.width = size;
  out.height = size;
  const ctx = out.getContext("2d")!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(small, 0, 0, size, size);
  return out;
}
