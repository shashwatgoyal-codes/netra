"use client";

import { useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { listScreenings, type Screening } from "./db";
import type { ModelMeta } from "./model";
import { META_URL } from "./model";

function subscribeOnline(cb: () => void) {
  window.addEventListener("online", cb);
  window.addEventListener("offline", cb);
  return () => {
    window.removeEventListener("online", cb);
    window.removeEventListener("offline", cb);
  };
}

export function useOnline() {
  return useSyncExternalStore(subscribeOnline, () => navigator.onLine, () => true);
}

/** All screenings, newest first. Call `reload` after a change. */
export function useScreenings() {
  const [rows, setRows] = useState<Screening[] | null>(null);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let alive = true;
    listScreenings().then((r) => alive && setRows(r));
    return () => {
      alive = false;
    };
  }, [tick]);
  return { rows, reload: () => setTick((t) => t + 1) };
}

export function useMeta() {
  const [meta, setMeta] = useState<ModelMeta | null>(null);
  useEffect(() => {
    fetch(META_URL)
      .then((r) => r.json())
      .then(setMeta)
      .catch(() => setMeta(null));
  }, []);
  return meta;
}

/** Object URL for a stored Blob, revoked on unmount. */
export function useBlobUrl(blob: Blob | undefined) {
  const url = useMemo(() => (blob ? URL.createObjectURL(blob) : undefined), [blob]);
  useEffect(
    () => () => {
      if (url) URL.revokeObjectURL(url);
    },
    [url],
  );
  return url;
}
