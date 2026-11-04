"use client";

import { useState } from "react";
import ScreeningRow from "@/components/ScreeningRow";
import { toCsv } from "@/lib/db";
import { useMeta, useScreenings } from "@/lib/hooks";

export default function RecordsPage() {
  const { rows } = useScreenings();
  const meta = useMeta();
  const [q, setQ] = useState("");
  const grades = meta?.grades ?? [];
  const needle = q.trim().toLowerCase();
  const shown = (rows ?? []).filter(
    (s) => !needle || s.patient.name.toLowerCase().includes(needle) || s.patient.patientId.toLowerCase().includes(needle),
  );

  function exportCsv() {
    const blob = new Blob([toCsv(shown, grades)], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `netra-screenings-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  const referable = (rows ?? []).filter((s) => s.refer).length;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Records</h1>
          <p className="mt-1 text-sm text-muted">{rows === null ? "Loading…" : `${rows.length} ${rows.length === 1 ? "eye" : "eyes"} screened · ${referable} referable`}</p>
        </div>
        <button type="button" className="btn btn-ghost" disabled={!shown.length} onClick={exportCsv}>
          Export CSV
        </button>
      </div>
      <input className="field" placeholder="Search by name or patient ID" value={q} onChange={(e) => setQ(e.target.value)} />
      {rows !== null && shown.length === 0 && <p className="card p-6 text-center text-sm text-muted">No screenings{needle ? " match that search" : " yet"}.</p>}
      <ul className="space-y-3">
        {shown.map((s) => (
          <ScreeningRow key={s.id} s={s} grades={grades} />
        ))}
      </ul>
    </div>
  );
}
