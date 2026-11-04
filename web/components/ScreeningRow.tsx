"use client";

import Link from "next/link";
import type { Screening, Status } from "@/lib/db";
import { useBlobUrl } from "@/lib/hooks";

export const STATUS_LABEL: Record<Status, string> = {
  pending: "Needs referral",
  referred: "Referred",
  seen: "Seen by specialist",
  routine: "Routine rescreen",
};

export function StatusPill({ status }: { status: Status }) {
  const tone =
    status === "pending" ? "bg-refer-soft text-refer" : status === "referred" ? "bg-warn-soft text-warn" : status === "seen" ? "bg-accent-soft text-accent" : "bg-routine-soft text-routine";
  return <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${tone}`}>{STATUS_LABEL[status]}</span>;
}

export default function ScreeningRow({ s, grades, children }: { s: Screening; grades: string[]; children?: React.ReactNode }) {
  const url = useBlobUrl(s.image);
  return (
    <li className="card flex gap-3 p-3">
      <Link href={`/records/view/?id=${s.id}`} className="h-16 w-16 shrink-0 overflow-hidden rounded-lg bg-black">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        {url && <img src={url} alt="" className="h-full w-full object-cover" />}
      </Link>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <Link href={`/records/view/?id=${s.id}`} className="truncate font-semibold hover:underline">
            {s.patient.name}
          </Link>
          <span className="font-mono text-xs text-muted">{s.patient.patientId}</span>
          <StatusPill status={s.status} />
        </div>
        <p className="mt-0.5 text-sm text-muted">
          <span className="capitalize">{s.eye}</span> eye · {grades[s.grade] ?? `Grade ${s.grade}`} · referable {Math.round(s.referableProb * 100)}%
        </p>
        <p className="text-xs text-muted">
          {new Date(s.createdAt).toLocaleString()}
          {s.patient.phone && ` · ${s.patient.phone}`}
        </p>
        {children}
      </div>
    </li>
  );
}
