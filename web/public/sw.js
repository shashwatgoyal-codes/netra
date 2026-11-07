// Offline support. Bump VERSION whenever the app shell changes so clinics
// pick up the new files the next time they are online.
const VERSION = "netra-v4";
// The model itself is cached by the page (lib/model.ts), once, under its own
// version key, so it is downloaded a single time with visible progress.
const MODEL_CACHE = "netra-model";
const MODEL_FILE = "/model/netra.onnx";
const PRECACHE = [
  "/",
  "/queue/",
  "/records/",
  "/records/view/",
  "/about/",
  "/manifest.webmanifest",
  "/model/model-meta.json",
  "/ort/ort.wasm.min.mjs",
  "/ort/ort-wasm-simd-threaded.mjs",
  "/ort/ort-wasm-simd-threaded.wasm",
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(VERSION).then((c) => c.addAll(PRECACHE)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION && k !== MODEL_CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

const networkFirst = (request) =>
  fetch(request)
    .then((res) => {
      if (res.ok) {
        const copy = res.clone();
        caches.open(VERSION).then((c) => c.put(request, copy));
      }
      return res;
    })
    .catch(() => caches.match(request, { ignoreSearch: true }).then((r) => r || caches.match("/")));

self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);
  if (request.method !== "GET" || url.origin !== self.location.origin) return;

  // The page streams and caches the model itself.
  if (url.pathname === MODEL_FILE) return;

  // Pages and model metadata: network first so updates arrive, cached copy offline.
  if (request.mode === "navigate" || url.pathname.startsWith("/model/")) {
    event.respondWith(networkFirst(request));
    return;
  }

  // Everything else (hashed JS/CSS, runtime): cache first.
  event.respondWith(
    caches.match(request).then(
      (hit) =>
        hit ||
        fetch(request).then((res) => {
          if (res.ok) {
            const copy = res.clone();
            caches.open(VERSION).then((c) => c.put(request, copy));
          }
          return res;
        }),
    ),
  );
});
