"use client";

import { useMemo, useState } from "react";
import { renderHeatmap } from "@/lib/heatmap";

type Props = {
  imageUrl: string;
  cam: ArrayLike<number>;
  camSize: [number, number];
  probs: number[];
  grade: number;
  referableProb: number;
  refer: boolean;
  grades: string[];
  referableFrom: number;
};

// Advice wording stays generic on purpose: programmes set their own referral
// intervals, so the app states the urgency band and defers to local protocol.
function advice(refer: boolean, grade: number, referableFrom: number) {
  if (!refer) return { title: "No referable retinopathy detected", body: "Rescreen at the next routine diabetic eye check, usually within 12 months.", tone: "routine" as const };
  if (grade >= referableFrom + 1) return { title: "Urgent referral to an eye specialist", body: "Signs suggest severe or proliferative disease. Refer as a priority under your programme's protocol.", tone: "refer" as const };
  return { title: "Refer to an eye specialist", body: "Signs suggest moderate or worse retinopathy. Arrange a routine referral under your programme's protocol.", tone: "refer" as const };
}

export default function ResultView({ imageUrl, cam, camSize, probs, grade, referableProb, refer, grades, referableFrom }: Props) {
  const [showHeat, setShowHeat] = useState(true);
  const a = advice(refer, grade, referableFrom);
  const evidence = grade > 0; // a "No DR" map highlights nothing clinically meaningful
  const heatUrl = useMemo(() => (evidence ? renderHeatmap(cam, camSize, 512).toDataURL() : undefined), [cam, camSize, evidence]);

  return (
    <div className="space-y-4">
      <div className={`rounded-2xl p-4 ${a.tone === "refer" ? "bg-refer-soft text-refer" : "bg-routine-soft text-routine"}`}>
        <p className="text-lg font-semibold">{a.title}</p>
        <p className="mt-1 text-sm text-ink/80">{a.body}</p>
      </div>

      <div className="card overflow-hidden">
        <div className="relative aspect-square w-full bg-black">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={imageUrl} alt="Retina photo analysed" className="absolute inset-0 h-full w-full object-contain" />
          {evidence && heatUrl && showHeat && (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={heatUrl} alt="Regions that drove the result" className="absolute inset-0 h-full w-full object-contain opacity-80 mix-blend-screen" />
          )}
        </div>
        <div className="flex items-center justify-between gap-3 p-3 text-sm">
          <p className="text-muted">{evidence ? "Bright areas drove the model's grade. Check them on the image." : "No lesion evidence to highlight."}</p>
          {evidence && (
            <button type="button" className="btn btn-ghost shrink-0 px-3 py-1.5 text-xs" onClick={() => setShowHeat((s) => !s)}>
              {showHeat ? "Hide heatmap" : "Show heatmap"}
            </button>
          )}
        </div>
      </div>

      <div className="card p-4">
        <div className="mb-3 flex items-baseline justify-between">
          <p className="font-semibold">Most likely grade: {grades[grade]}</p>
          <p className="text-sm text-muted">
            Referable: <span className="font-mono font-semibold text-ink">{Math.round(referableProb * 100)}%</span>
          </p>
        </div>
        <ul className="space-y-2">
          {grades.map((g, i) => (
            <li key={g} className="grid grid-cols-[8.5rem_1fr_3rem] items-center gap-3 text-sm">
              <span className={i === grade ? "font-semibold" : "text-muted"}>{g}</span>
              <span className="h-2 overflow-hidden rounded-full bg-surface-2">
                <span className={`block h-full rounded-full ${i >= referableFrom ? "bg-refer" : "bg-routine"}`} style={{ width: `${Math.max(probs[i] * 100, 1)}%` }} />
              </span>
              <span className="text-right font-mono text-xs">{Math.round(probs[i] * 100)}%</span>
            </li>
          ))}
        </ul>
      </div>

      <p className="text-xs leading-relaxed text-muted">
        Netra is a screening aid, not a diagnosis. It has not been clinically validated in your setting. A trained grader or
        ophthalmologist should confirm every referral decision.
      </p>
    </div>
  );
}
