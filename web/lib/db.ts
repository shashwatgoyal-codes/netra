// Screenings live only on this device, in IndexedDB. Nothing is uploaded.

import { openDB, type DBSchema, type IDBPDatabase } from "idb";

export type Eye = "left" | "right";
export type Status = "pending" | "referred" | "seen" | "routine";

export type Patient = {
  name: string;
  patientId: string;
  age: number | null;
  sex: "F" | "M" | "O" | "";
  diabetesYears: number | null;
  phone: string;
};

export type Screening = {
  id: string;
  createdAt: number;
  patient: Patient;
  eye: Eye;
  image: Blob; // the square crop the model saw (JPEG)
  probs: number[];
  grade: number;
  referableProb: number;
  refer: boolean;
  threshold: number;
  cam: number[];
  camSize: [number, number];
  modelVersion: string;
  quality: { sharpness: number; fundusCoverage: number; ok: boolean };
  status: Status;
  notes: string;
  updatedAt: number;
};

interface NetraDB extends DBSchema {
  screenings: {
    key: string;
    value: Screening;
    indexes: { createdAt: number; status: Status; patientId: string };
  };
}

let dbPromise: Promise<IDBPDatabase<NetraDB>> | null = null;

function db() {
  dbPromise ??= openDB<NetraDB>("netra", 1, {
    upgrade(d) {
      const store = d.createObjectStore("screenings", { keyPath: "id" });
      store.createIndex("createdAt", "createdAt");
      store.createIndex("status", "status");
      store.createIndex("patientId", "patient.patientId");
    },
  });
  return dbPromise;
}

export async function saveScreening(s: Screening) {
  await (await db()).put("screenings", s);
}

export async function getScreening(id: string) {
  return (await db()).get("screenings", id);
}

export async function listScreenings(): Promise<Screening[]> {
  const all = await (await db()).getAllFromIndex("screenings", "createdAt");
  return all.reverse();
}

export async function updateScreening(id: string, patch: Partial<Pick<Screening, "status" | "notes">>) {
  const d = await db();
  const s = await d.get("screenings", id);
  if (!s) return;
  await d.put("screenings", { ...s, ...patch, updatedAt: Date.now() });
}

export async function deleteScreening(id: string) {
  await (await db()).delete("screenings", id);
}

const CSV_COLUMNS = [
  "date",
  "patient_id",
  "name",
  "age",
  "sex",
  "diabetes_years",
  "phone",
  "eye",
  "predicted_grade",
  "referable_probability",
  "refer",
  "status",
  "notes",
  "model_version",
] as const;

function csvCell(v: unknown): string {
  const s = v == null ? "" : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function toCsv(rows: Screening[], grades: string[]): string {
  const lines = rows.map((s) =>
    [
      new Date(s.createdAt).toISOString(),
      s.patient.patientId,
      s.patient.name,
      s.patient.age,
      s.patient.sex,
      s.patient.diabetesYears,
      s.patient.phone,
      s.eye,
      grades[s.grade],
      s.referableProb.toFixed(3),
      s.refer ? "yes" : "no",
      s.status,
      s.notes,
      s.modelVersion,
    ]
      .map(csvCell)
      .join(","),
  );
  return [CSV_COLUMNS.join(","), ...lines].join("\n");
}
