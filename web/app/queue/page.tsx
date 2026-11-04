"use client";

import Link from "next/link";
import ScreeningRow from "@/components/ScreeningRow";
import { updateScreening, type Status } from "@/lib/db";
import { useMeta, useScreenings } from "@/lib/hooks";

const NEXT: Partial<Record<Status, { to: Status; label: string }>> = {
  pending: { to: "referred", label: "Mark referred" },
  referred: { to: "seen", label: "Seen by specialist" },
};

export default function QueuePage() {
  const { rows, reload } = useScreenings();
  const meta = useMeta();
  const grades = meta?.grades ?? [];
  const open = (rows ?? []).filter((s) => s.status === "pending" || s.status === "referred");
  // Most urgent first: highest grade, then oldest.
  open.sort((a, b) => b.grade - a.grade || a.createdAt - b.createdAt);
  const pending = open.filter((s) => s.status === "pending").length;

  async function advance(id: string, to: Status) {
    await updateScreening(id, { status: to });
    reload();
  }

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Referrals</h1>
        <p className="mt-1 text-sm text-muted">
          {rows === null ? "Loading…" : `${pending} waiting to be referred · ${open.length - pending} awaiting specialist visit`}
        </p>
      </div>
      {rows !== null && open.length === 0 && (
        <div className="card p-6 text-center text-sm text-muted">
          No open referrals.{" "}
          <Link href="/" className="text-accent underline">
            Start a screening
          </Link>
        </div>
      )}
      <ul className="space-y-3">
        {open.map((s) => {
          const next = NEXT[s.status];
          return (
            <ScreeningRow key={s.id} s={s} grades={grades}>
              <div className="mt-2 flex flex-wrap gap-2">
                {s.patient.phone && (
                  <a className="btn btn-ghost px-3 py-1.5 text-xs" href={`tel:${s.patient.phone}`}>
                    Call
                  </a>
                )}
                {next && (
                  <button type="button" className="btn btn-primary px-3 py-1.5 text-xs" onClick={() => advance(s.id, next.to)}>
                    {next.label}
                  </button>
                )}
              </div>
            </ScreeningRow>
          );
        })}
      </ul>
    </div>
  );
}
