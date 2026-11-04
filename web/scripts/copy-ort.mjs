// Copies the ONNX Runtime Web CPU build into public/ort so it is served as
// plain static files (and cached by the service worker) instead of bundled.
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// The package's "exports" hides package.json, so resolve its CommonJS entry
// (dist/ort.min.js) and take that file's folder.
const dist = dirname(fileURLToPath(import.meta.resolve("onnxruntime-web")));
const out = join(import.meta.dirname, "..", "public", "ort");
mkdirSync(out, { recursive: true });
for (const file of ["ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]) {
  copyFileSync(join(dist, file), join(out, file));
}
console.log(`copied onnxruntime-web to ${out}`);
