"use client";

import { useEffect } from "react";

export default function ServiceWorker() {
  useEffect(() => {
    // Dev server assets change on every edit; only cache the production build.
    if (process.env.NODE_ENV !== "production" || !("serviceWorker" in navigator)) return;
    navigator.serviceWorker.register("/sw.js").catch(() => {});
  }, []);
  return null;
}
