"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import ResultView from "@/components/ResultView";
import { saveScreening, type Eye, type Patient, type Screening } from "@/lib/db";
import { loadModel, predict, type ModelMeta, type Prediction } from "@/lib/model";
import { loadImage, prepare, type Prepared } from "@/lib/preprocess";

type ModelState = { status: "loading" } | { status: "ready"; meta: ModelMeta } | { status: "error"; message: string };

const EMPTY_PATIENT: Patient = { name: "", patientId: "", age: null, sex: "", diabetesYears: null, phone: "" };

function toBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("Could not encode image"))), "image/jpeg", 0.92));
}

function num(v: string): number | null {
  const n = Number(v);
  return v.trim() === "" || Number.isNaN(n) ? null : n;
}

export default function ScreenFlow() {
  const [model, setModel] = useState<ModelState>({ status: "loading" });
  const [patient, setPatient] = useState<Patient>(EMPTY_PATIENT);
  const [eye, setEye] = useState<Eye>("right");
  const [prepared, setPrepared] = useState<Prepared | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const [result, setResult] = useState<{ prediction: Prediction; saved: Screening } | null>(null);
  const [done, setDone] = useState<Eye[]>([]);
  const cameraRef = useRef<HTMLInputElement>(null);
  const galleryRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadModel()
      .then(({ meta }) => setModel({ status: "ready", meta }))
      .catch((e) => setModel({ status: "error", message: e instanceof Error ? e.message : String(e) }));
  }, []);

  const set = <K extends keyof Patient>(k: K, v: Patient[K]) => setPatient((p) => ({ ...p, [k]: v }));
  const canCapture = patient.patientId.trim() !== "" && patient.name.trim() !== "";

  async function onFile(file: File | undefined) {
    if (!file || model.status !== "ready") return;
    setError(undefined);
    setResult(null);
    try {
      const bitmap = await loadImage(file);
      const p = prepare(bitmap, model.meta);
      bitmap.close();
      setPrepared(p);
      setPreviewUrl(p.preview.toDataURL("image/jpeg", 0.9));
    } catch {
      setError("Could not read that image. Try another photo.");
    }
  }

  async function analyse() {
    if (!prepared || model.status !== "ready") return;
    setBusy(true);
    setError(undefined);
    try {
      const prediction = await predict(prepared.tensor);
      const now = Date.now();
      const screening: Screening = {
        id: crypto.randomUUID(),
        createdAt: now,
        updatedAt: now,
        patient,
        eye,
        image: await toBlob(prepared.preview),
        probs: prediction.probs,
        grade: prediction.grade,
        referableProb: prediction.referableProb,
        refer: prediction.refer,
        threshold: model.meta.threshold,
        cam: Array.from(prediction.cam),
        camSize: model.meta.camSize,
        modelVersion: model.meta.version,
        quality: { sharpness: prepared.quality.sharpness, fundusCoverage: prepared.quality.fundusCoverage, ok: prepared.quality.ok },
        status: prediction.refer ? "pending" : "routine",
        notes: "",
      };
      await saveScreening(screening);
      setResult({ prediction, saved: screening });
      setDone((d) => (d.includes(eye) ? d : [...d, eye]));
    } catch (e) {
      setError(`Analysis failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  function otherEye() {
    setEye((e) => (e === "right" ? "left" : "right"));
    setPrepared(null);
    setPreviewUrl(undefined);
    setResult(null);
  }

  function newPatient() {
    setPatient(EMPTY_PATIENT);
    setEye("right");
    setPrepared(null);
    setPreviewUrl(undefined);
    setResult(null);
    setDone([]);
    window.scrollTo({ top: 0 });
  }

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">New screening</h1>
        <p className="mt-1 text-sm text-muted">Runs entirely on this device. Photos and results never leave it.</p>
      </div>

      <ModelBadge state={model} />

      <section className="card space-y-3 p-4">
        <h2 className="font-semibold">1. Patient</h2>
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="space-y-1 text-sm">
            <span className="text-muted">Patient ID *</span>
            <input className="field" value={patient.patientId} onChange={(e) => set("patientId", e.target.value)} placeholder="e.g. PHC-0142" />
          </label>
          <label className="space-y-1 text-sm">
            <span className="text-muted">Name *</span>
            <input className="field" value={patient.name} onChange={(e) => set("name", e.target.value)} autoComplete="off" />
          </label>
          <label className="space-y-1 text-sm">
            <span className="text-muted">Age</span>
            <input className="field" inputMode="numeric" value={patient.age ?? ""} onChange={(e) => set("age", num(e.target.value))} />
          </label>
          <label className="space-y-1 text-sm">
            <span className="text-muted">Sex</span>
            <select className="field" value={patient.sex} onChange={(e) => set("sex", e.target.value as Patient["sex"])}>
              <option value="">–</option>
              <option value="F">Female</option>
              <option value="M">Male</option>
              <option value="O">Other</option>
            </select>
          </label>
          <label className="space-y-1 text-sm">
            <span className="text-muted">Years with diabetes</span>
            <input className="field" inputMode="numeric" value={patient.diabetesYears ?? ""} onChange={(e) => set("diabetesYears", num(e.target.value))} />
          </label>
          <label className="space-y-1 text-sm">
            <span className="text-muted">Phone (for follow-up)</span>
            <input className="field" inputMode="tel" value={patient.phone} onChange={(e) => set("phone", e.target.value)} />
          </label>
        </div>
      </section>

      <section className={`card space-y-3 p-4 ${canCapture ? "" : "opacity-60"}`}>
        <div className="flex items-center justify-between">
          <h2 className="font-semibold">2. Retina photo</h2>
          <div className="flex rounded-lg bg-surface-2 p-0.5 text-sm" role="radiogroup" aria-label="Eye">
            {(["right", "left"] as Eye[]).map((e) => (
              <button
                key={e}
                type="button"
                role="radio"
                aria-checked={eye === e}
                className={`rounded-md px-3 py-1 font-medium capitalize ${eye === e ? "bg-surface text-ink shadow-sm" : "text-muted"}`}
                onClick={() => {
                  setEye(e);
                  setResult(null);
                }}
              >
                {e} eye{done.includes(e) ? " ✓" : ""}
              </button>
            ))}
          </div>
        </div>

        {!canCapture && <p className="text-sm text-muted">Enter the patient ID and name first.</p>}

        <div className="flex flex-wrap gap-2">
          <button type="button" className="btn btn-primary" disabled={!canCapture || model.status !== "ready"} onClick={() => cameraRef.current?.click()}>
            Take photo
          </button>
          <button type="button" className="btn btn-ghost" disabled={!canCapture || model.status !== "ready"} onClick={() => galleryRef.current?.click()}>
            Choose from device
          </button>
          <input ref={cameraRef} type="file" accept="image/*" capture="environment" className="hidden" onChange={(e) => onFile(e.target.files?.[0])} />
          <input ref={galleryRef} type="file" accept="image/*" className="hidden" data-testid="upload" onChange={(e) => onFile(e.target.files?.[0])} />
        </div>

        {prepared && previewUrl && !result && (
          <div className="space-y-3">
            <div className="mx-auto aspect-square w-full max-w-sm overflow-hidden rounded-xl bg-black">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={previewUrl} alt="Cropped retina photo" className="h-full w-full object-contain" />
            </div>
            {!prepared.quality.ok && (
              <div className="rounded-xl bg-warn-soft p-3 text-sm text-warn">
                <p className="font-semibold">Photo quality check</p>
                <ul className="mt-1 list-disc pl-5">
                  {prepared.quality.issues.map((i) => (
                    <li key={i}>{i}</li>
                  ))}
                </ul>
                <p className="mt-1">Retake if you can. Poor photos make the result less reliable.</p>
              </div>
            )}
            <button type="button" className="btn btn-primary w-full" disabled={busy} onClick={analyse}>
              {busy ? "Analysing on device…" : prepared.quality.ok ? "Analyse" : "Analyse anyway"}
            </button>
          </div>
        )}
        {error && <p className="rounded-xl bg-refer-soft p-3 text-sm text-refer">{error}</p>}
      </section>

      {result && model.status === "ready" && previewUrl && (
        <section className="space-y-4">
          <h2 className="font-semibold">
            3. Result · <span className="capitalize">{eye}</span> eye
          </h2>
          <ResultView
            imageUrl={previewUrl}
            cam={result.prediction.cam}
            camSize={model.meta.camSize}
            probs={result.prediction.probs}
            grade={result.prediction.grade}
            referableProb={result.prediction.referableProb}
            refer={result.prediction.refer}
            grades={model.meta.grades}
            referableFrom={model.meta.referableFrom}
          />
          <p className="text-xs text-muted">
            Saved to this device · analysed in {Math.round(result.prediction.ms)} ms ·{" "}
            <Link className="underline" href={`/records/view/?id=${result.saved.id}`}>
              open record
            </Link>
          </p>
          <div className="grid gap-2 sm:grid-cols-2">
            {done.length < 2 && (
              <button type="button" className="btn btn-primary" onClick={otherEye}>
                Screen {eye === "right" ? "left" : "right"} eye
              </button>
            )}
            <button type="button" className="btn btn-ghost" onClick={newPatient}>
              Next patient
            </button>
          </div>
        </section>
      )}
    </div>
  );
}

function ModelBadge({ state }: { state: ModelState }) {
  if (state.status === "loading")
    return <p className="card flex items-center gap-2 p-3 text-sm text-muted"><span className="h-2 w-2 animate-pulse rounded-full bg-accent" />Loading the AI model onto this device…</p>;
  if (state.status === "error")
    return <p className="rounded-xl bg-refer-soft p-3 text-sm text-refer">The AI model could not load ({state.message}). Connect to the internet once so it can be downloaded.</p>;
  return (
    <p className="card flex items-center gap-2 p-3 text-sm text-muted">
      <span className="h-2 w-2 rounded-full bg-routine" />
      Model ready, works offline · <span className="font-mono text-xs">{state.meta.version}</span>
    </p>
  );
}
