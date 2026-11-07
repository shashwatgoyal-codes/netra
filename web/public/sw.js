// Offline support. Bump VERSION whenever the model or app shell changes so
// clinics pick up the new files the next time they are online.
const VERSION = "netra-v3";
const PRECACHE = [
  "/",
  "/queue/",
  "/records/",
  "/records/view/",
  "/about/",
  "/manifest.webmanifest",
  "/model/netra.onnx",
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
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);
  if (request.method !== "GET" || url.origin !== self.location.origin) return;

  // Pages: network first so updates arrive, cached copy when offline.
  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request)
        .then((res) => {
          const copy = res.clone();
          caches.open(VERSION).then((c) => c.put(request, copy));
          return res;
        })
        .catch(() => caches.match(request, { ignoreSearch: true }).then((r) => r || caches.match("/"))),
    );
    return;
  }

  // Everything else (hashed JS/CSS, model, runtime): cache first.
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
