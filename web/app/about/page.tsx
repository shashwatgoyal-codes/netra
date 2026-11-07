import { readFile } from "node:fs/promises";
import { join } from "node:path";

// Read at build time: the model card always describes the model shipped in
// this build, and the page works offline like the rest of the app.

type Metrics = {
  model: string;
  train_sources?: string[];
  variant: string;
  modelMB: number;
  dataset: string;
  n_test: number;
  n_referable: number;
  threshold: number;
  target_sensitivity: number;
  referral: { auc: number; sensitivity: number; specificity: number; ppv: number; npv: number; tp: number; tn: number; fp: number; fn: number };
  grading: { qwk: number; accuracy: number; confusion: number[][] };
  roc: { fpr: number[]; tpr: number[] };
  external?: {
    key: string;
    dataset: string;
    n: number;
    n_referable: number;
    referral: { auc: number; sensitivity: number; specificity: number };
    grading: { qwk: number };
  }[];
};

type Meta = { grades: string[]; version: string; inputSize: number };

const BACKBONES: Record<string, string> = {
  dinov3: "ConvNeXt-Tiny with Meta DINOv3 pretraining (2025): self-supervised on 1.7 billion images, distilled from a 7-billion-parameter teacher",
  cnv2: "ConvNeXt V2-Nano (2023), FCMAE self-supervised pretraining, fine-tuned on ImageNet-22k at 384 px",
};

async function load<T>(file: string): Promise<T> {
  return JSON.parse(await readFile(join(process.cwd(), "public", "model", file), "utf8"));
}

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

const STAT_NAMES = ["Sensitivity", "Specificity", "AUC", "Grading κ"] as const;

function statRow(sens: number, spec: number, auc: number, qwk: number): [string, string][] {
  return [
    ["Sensitivity", pct(sens)],
    ["Specificity", pct(spec)],
    ["AUC", auc.toFixed(3)],
    ["Grading κ", qwk.toFixed(3)],
  ];
}

function Roc({ m }: { m: Metrics }) {
  const s = 240;
  const pts = m.roc.fpr.map((f, i) => `${(f * s).toFixed(1)},${((1 - m.roc.tpr[i]) * s).toFixed(1)}`).join(" ");
  const op = { x: (1 - m.referral.specificity) * s, y: (1 - m.referral.sensitivity) * s };
  return (
    <svg viewBox={`-30 -10 ${s + 45} ${s + 45}`} className="w-full max-w-xs" role="img" aria-label="ROC curve for the referral decision">
      <rect x={0} y={0} width={s} height={s} fill="none" stroke="var(--line)" />
      <line x1={0} y1={s} x2={s} y2={0} stroke="var(--line)" strokeDasharray="4 4" />
      <polyline points={pts} fill="none" stroke="var(--accent)" strokeWidth={2.5} />
      <circle cx={op.x} cy={op.y} r={5} fill="var(--refer)" />
      <text x={op.x + 8} y={op.y + 14} fontSize={11} fill="var(--ink)">
        operating point
      </text>
      <text x={s / 2} y={s + 28} fontSize={11} textAnchor="middle" fill="var(--muted)">
        False positive rate
      </text>
      <text x={-18} y={s / 2} fontSize={11} textAnchor="middle" fill="var(--muted)" transform={`rotate(-90 -18 ${s / 2})`}>
        Sensitivity
      </text>
    </svg>
  );
}

export default async function AboutPage() {
  const [m, meta] = await Promise.all([load<Metrics>("metrics.json"), load<Meta>("model-meta.json")]);
  const placeholder = meta.version.startsWith("placeholder");
  const r = m.referral;
  const stats = [
    { label: "Sensitivity", value: pct(r.sensitivity), hint: `${r.tp} of ${r.tp + r.fn} referable eyes caught` },
    { label: "Specificity", value: pct(r.specificity), hint: `${r.tn} of ${r.tn + r.fp} healthy eyes cleared` },
    { label: "AUC", value: r.auc.toFixed(3), hint: "Refer vs. don't refer" },
    { label: "Grading κ", value: m.grading.qwk.toFixed(3), hint: "Quadratic weighted kappa, 5 grades" },
  ];
  const maxCell = Math.max(...m.grading.confusion.flat());
  const sites = [
    {
      key: "aptos",
      label: "APTOS held-out test (Aravind, India)",
      detail: `${m.n_test} photos, ${m.n_referable} referable`,
      stats: statRow(r.sensitivity, r.specificity, r.auc, m.grading.qwk),
    },
    ...(m.external ?? []).map((e) => ({
      key: e.key,
      label: e.dataset,
      detail: `${e.n} photos, ${e.n_referable} referable`,
      stats: statRow(e.referral.sensitivity, e.referral.specificity, e.referral.auc, e.grading.qwk),
    })),
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">How reliable is Netra?</h1>
        <p className="mt-1 text-sm text-muted">
          Measured on {m.n_test} held-out photos the model never saw in training ({m.n_referable} referable), using exactly the file shipped in this app.
        </p>
      </div>

      {placeholder && (
        <p className="rounded-xl bg-warn-soft p-3 text-sm text-warn">
          This build ships an untrained placeholder model for development. These numbers are not meaningful.
        </p>
      )}

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {stats.map((s) => (
          <div key={s.label} className="card p-4">
            <p className="text-xs font-medium uppercase tracking-wide text-muted">{s.label}</p>
            <p className="mt-1 font-mono text-2xl font-semibold">{s.value}</p>
            <p className="mt-1 text-xs text-muted">{s.hint}</p>
          </div>
        ))}
      </div>

      {m.external && m.external.length > 0 && (
        <section className="card space-y-3 p-4">
          <div>
            <h2 className="font-semibold">Beyond the training data</h2>
            <p className="mt-1 text-sm text-muted">
              The same model and threshold on photos from other hospitals, cameras and graders, with no retuning. This is the closest check here to what a
              new clinic would get.
              {m.train_sources?.includes("ddr") &&
                " DDR is also in the training data, so its row is held-out photos from hospitals the model has seen. IDRiD and Messidor-2 are the truly new sites."}
            </p>
          </div>
          {/* Phones: one card per test set. Wider screens: a table. */}
          <ul className="space-y-2 sm:hidden">
            {sites.map((s) => (
              <li key={s.key} className="rounded-xl bg-surface-2 p-3">
                <p className="text-sm font-medium">{s.label}</p>
                <p className="text-xs text-muted">{s.detail}</p>
                <dl className="mt-2 grid grid-cols-4 gap-2 text-center">
                  {s.stats.map(([name, value]) => (
                    <div key={name}>
                      <dt className="text-[11px] text-muted">{name}</dt>
                      <dd className="font-mono text-sm font-semibold">{value}</dd>
                    </div>
                  ))}
                </dl>
              </li>
            ))}
          </ul>
          <table className="hidden w-full text-sm sm:table">
            <thead>
              <tr className="text-left text-xs text-muted">
                <th className="pb-2 font-medium" />
                {STAT_NAMES.map((n) => (
                  <th key={n} className="pb-2 font-medium">
                    {n}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="font-mono">
              {sites.map((s) => (
                <tr key={s.key}>
                  <td className="py-1 font-sans text-muted">
                    {s.label}
                    <span className="block text-xs">{s.detail}</span>
                  </td>
                  {s.stats.map(([name, value]) => (
                    <td key={name}>{value}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <section className="card grid gap-6 p-4 sm:grid-cols-2">
        <div>
          <h2 className="font-semibold">Referral decision</h2>
          <p className="mt-1 text-sm text-muted">
            An eye is referred when the predicted chance of moderate-or-worse retinopathy reaches{" "}
            <span className="font-mono">{pct(m.threshold)}</span>. That threshold was set on a separate validation split to catch at least{" "}
            {pct(m.target_sensitivity)} of referable disease, because a missed case costs far more than an extra referral.
          </p>
          <p className="mt-2 text-sm text-muted">
            Of the eyes it referred, {pct(r.ppv)} truly needed it. Of the eyes it cleared, {pct(r.npv)} were truly fine.
          </p>
        </div>
        <Roc m={m} />
      </section>

      <section className="card overflow-x-auto p-4">
        <h2 className="font-semibold">Grade confusion (rows: true grade, columns: predicted)</h2>
        <table className="mt-3 text-sm">
          <thead>
            <tr>
              <th />
              {meta.grades.map((g) => (
                <th key={g} className="px-2 pb-2 text-left text-xs font-medium text-muted">
                  {g}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {m.grading.confusion.map((row, i) => (
              <tr key={meta.grades[i]}>
                <th className="pr-3 text-left text-xs font-medium text-muted">{meta.grades[i]}</th>
                {row.map((v, j) => (
                  <td key={j} className="px-1 py-0.5">
                    <span
                      className={`block min-w-12 rounded-md px-2 py-1.5 text-center font-mono ${i === j ? "font-semibold" : ""}`}
                      style={{ background: `color-mix(in srgb, var(--accent) ${Math.round((v / maxCell) * 70)}%, var(--surface-2))` }}
                    >
                      {v}
                    </span>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="card space-y-2 p-4 text-sm">
        <h2 className="font-semibold">Model</h2>
        <p className="text-muted">{BACKBONES[m.model] ?? m.model}. Fine-tuned to grade diabetic retinopathy on 5 levels.</p>
        <p className="text-muted">
          Runs on the phone with ONNX Runtime Web: {m.modelMB} MB, weights stored in fp16, input {meta.inputSize}×{meta.inputSize}. Heatmaps are exact class
          activation maps computed in the same forward pass.
        </p>
        <p className="text-muted">Data: {m.dataset}.</p>
      </section>

      <section className="card space-y-2 p-4 text-sm">
        <h2 className="font-semibold">Limits</h2>
        <ul className="list-disc space-y-1 pl-5 text-muted">
          <li>
            Trained on photos from one hospital network in India (APTOS 2019). {m.external ? "The external test above shows how much that matters. " : ""}Local
            cameras, smartphone adapters and populations need their own validation.
          </li>
          <li>Not validated in a prospective clinical study. It is a screening aid. A trained grader or ophthalmologist must confirm referrals.</li>
          <li>Does not detect other eye disease (glaucoma, cataract, macular degeneration). It does not grade macular edema separately.</li>
          <li>Ungradable photos (blurred, dark, poorly centred) reduce accuracy. The app warns, but cannot always tell.</li>
        </ul>
      </section>
    </div>
  );
}
