// On-device inference with ONNX Runtime Web. The runtime is served from
// /ort/ (copied there by scripts/copy-ort.mjs) and imported at runtime, not
// bundled, so its WASM files resolve next to it and the service worker can
// cache them for offline use.

import type * as ORT from "onnxruntime-web";

export type ModelMeta = {
  inputSize: number;
  mean: [number, number, number];
  std: [number, number, number];
  darkThreshold: number;
  grades: string[];
  referableFrom: number;
  threshold: number;
  camSize: [number, number];
  version: string;
};

export type Prediction = {
  probs: number[];
  grade: number;
  referableProb: number;
  refer: boolean;
  cam: Float32Array; // activation map of the predicted grade, [h * w]
  backend: string;
  ms: number;
};

export const MODEL_URL = "/model/netra.onnx";
export const META_URL = "/model/model-meta.json";
const ORT_URL = "/ort/ort.wasm.min.mjs";

type Loaded = { ort: typeof ORT; session: ORT.InferenceSession; meta: ModelMeta; backend: string };
let loading: Promise<Loaded> | null = null;

async function load(): Promise<Loaded> {
  // CPU (WASM) build only: its runtime is 15 MB against 29 MB for the WebGPU
  // build, and the low-cost Android phones this targets rarely expose WebGPU.
  // One image per screening keeps CPU latency acceptable.
  const ort: typeof ORT = await import(/* webpackIgnore: true */ /* turbopackIgnore: true */ ORT_URL);
  ort.env.wasm.wasmPaths = "/ort/";
  // Static hosting can't send the COOP/COEP headers multithreading needs.
  ort.env.wasm.numThreads = 1;
  // Run inference in ORT's own worker so the UI stays responsive.
  ort.env.wasm.proxy = true;

  const meta: ModelMeta = await fetch(META_URL).then((r) => r.json());
  const buffer: ArrayBuffer = await (await fetch(MODEL_URL)).arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  return { ort, session, meta, backend: "wasm" };
}

export function loadModel(): Promise<Loaded> {
  loading ??= load().catch((err) => {
    loading = null; // let the next attempt retry, e.g. after coming back online
    throw err;
  });
  return loading;
}

export async function predict(tensor: Float32Array): Promise<Prediction> {
  const { ort, session, meta, backend } = await loadModel();
  const size = meta.inputSize;
  const start = performance.now();
  const out = await session.run({ image: new ort.Tensor("float32", tensor, [1, 3, size, size]) });
  const ms = performance.now() - start;

  const probs = Array.from(out.probs.data as Float32Array);
  const grade = probs.indexOf(Math.max(...probs));
  const referableProb = probs.slice(meta.referableFrom).reduce((a, b) => a + b, 0);
  const [h, w] = meta.camSize;
  const cams = out.cams.data as Float32Array;
  const cam = cams.slice(grade * h * w, (grade + 1) * h * w);
  // Same rule as ml/evaluate.py `refer_decision`: never show a referable top
  // grade next to a "no referral" verdict.
  const refer = referableProb >= meta.threshold || grade >= meta.referableFrom;
  return { probs, grade, referableProb, refer, cam, backend, ms };
}
