"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import ResultView from "@/components/ResultView";
import { STATUS_LABEL } from "@/components/ScreeningRow";
import { deleteScreening, getScreening, updateScreening, type Screening, type Status } from "@/lib/db";
import { useBlobUrl, useMeta } from "@/lib/hooks";

function Record() {
  const id = useSearchParams().get("id");
  const router = useRouter();
  const meta = useMeta();
  const [s, setS] = useState<Screening | null | undefined>();
  const [notes, setNotes] = useState("");
  const url = useBlobUrl(s?.image);

  useEffect(() => {
    if (!id) return;
    getScreening(id).then((r) => {
      setS(r ?? null);
      setNotes(r?.notes ?? "");
    });
  }, [id]);

  if (id && (s === undefined || !meta)) return <p className="text-sm text-muted">Loading…</p>;
  if (!id || !s || !meta)
    return (
      <p className="text-sm text-muted">
        Record not found.{" "}
        <Link href="/records/" className="text-accent underline">
          Back to records
        </Link>
      </p>
    );

  async function save(patch: { status?: Status; notes?: string }) {
    await updateScreening(s!.id, patch);
    setS({ ...s!, ...patch });
  }

  async function remove() {
    if (!confirm("Delete this screening from the device? This cannot be undone.")) return;
    await deleteScreening(s!.id);
    router.push("/records/");
  }

  const p = s.patient;
  return (
    <div className="space-y-5">
      <div>
        <Link href="/records/" className="text-sm text-muted hover:text-ink">
          ← Records
        </Link>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight">{p.name}</h1>
        <p className="text-sm text-muted">
          <span className="font-mono">{p.patientId}</span>
          {p.age != null && ` · ${p.age} y`}
          {p.sex && ` · ${p.sex}`}
          {p.diabetesYears != null && ` · diabetic ${p.diabetesYears} y`}
          {p.phone && ` · ${p.phone}`}
        </p>
        <p className="text-sm text-muted">
          <span className="capitalize">{s.eye}</span> eye · {new Date(s.createdAt).toLocaleString()} · model <span className="font-mono text-xs">{s.modelVersion}</span>
        </p>
      </div>

      {url && (
        <ResultView
          imageUrl={url}
          cam={s.cam}
          camSize={s.camSize}
          probs={s.probs}
          grade={s.grade}
          referableProb={s.referableProb}
          refer={s.refer}
          grades={meta.grades}
          referableFrom={meta.referableFrom}
        />
      )}
      {!s.quality.ok && <p className="rounded-xl bg-warn-soft p-3 text-sm text-warn">This photo failed the quality check when it was taken. Consider rescreening.</p>}

      <section className="card space-y-3 p-4">
        <h2 className="font-semibold">Follow-up</h2>
        <label className="block space-y-1 text-sm">
          <span className="text-muted">Status</span>
          <select className="field" value={s.status} onChange={(e) => save({ status: e.target.value as Status })}>
            {(Object.keys(STATUS_LABEL) as Status[]).map((k) => (
              <option key={k} value={k}>
                {STATUS_LABEL[k]}
              </option>
            ))}
          </select>
        </label>
        <label className="block space-y-1 text-sm">
          <span className="text-muted">Notes</span>
          <textarea className="field min-h-24" value={notes} onChange={(e) => setNotes(e.target.value)} onBlur={() => notes !== s.notes && save({ notes })} />
        </label>
        <button type="button" className="text-sm text-refer underline" onClick={remove}>
          Delete record
        </button>
      </section>
    </div>
  );
}

export default function RecordPage() {
  return (
    <Suspense fallback={<p className="text-sm text-muted">Loading…</p>}>
      <Record />
    </Suspense>
  );
}
