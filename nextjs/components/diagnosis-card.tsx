"use client";

import { useState } from "react";

export interface DiagnosisCardProps {
  rank: number;
  name: string;
  probability: number;
  icd10: string;
  explanation: string;
  isExpanded?: boolean;
}

export default function DiagnosisCard({
  rank,
  name,
  probability,
  icd10,
  explanation,
}: DiagnosisCardProps) {
  const [isHovered, setIsHovered] = useState(false);

  const getProbabilityColor = (prob: number) => {
    if (prob >= 80) return "bg-emerald-500";
    if (prob >= 60) return "bg-blue-500";
    return "bg-amber-500";
  };

  const getProbabilityLabel = (prob: number) => {
    if (prob >= 80) return "High";
    if (prob >= 60) return "Moderate";
    return "Low-Moderate";
  };

  return (
    <div
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className="group cursor-pointer rounded-xl border border-neutral-200/50 bg-white/60 p-6 transition hover:border-blue-300/50 hover:bg-white/80 hover:shadow-lg dark:border-neutral-800/50 dark:bg-neutral-900/60 dark:hover:border-blue-700/50 dark:hover:bg-neutral-900/80 dark:hover:shadow-lg dark:hover:shadow-blue-900/20"
    >
      <div className="flex items-start gap-4">
        {/* Rank Badge */}
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-linear-to-br from-blue-600 to-cyan-600 text-sm font-bold text-white dark:from-blue-400 dark:to-cyan-300 dark:text-neutral-900">
          {rank}
        </div>

        {/* Content */}
        <div className="flex-1">
          <div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-center">
            <div>
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-white">
                {name}
              </h3>
              <p className="text-sm text-neutral-500 dark:text-neutral-400">
                ICD-10: <span className="font-mono font-semibold text-blue-600 dark:text-blue-400">{icd10}</span>
              </p>
            </div>

            {/* Probability */}
            <div className="text-right">
              <div className="text-2xl font-bold text-neutral-900 dark:text-white">
                {probability}%
              </div>
              <div className="text-xs text-neutral-500 dark:text-neutral-400">
                {getProbabilityLabel(probability)}
              </div>
            </div>
          </div>

          {/* Probability Bar */}
          <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-800">
            <div
              className={`h-full transition-all duration-500 ${getProbabilityColor(probability)}`}
              style={{ width: `${probability}%` }}
            />
          </div>

          {/* Expandable Explanation */}
          {isHovered && (
            <div className="mt-4 border-t border-neutral-200 pt-4 dark:border-neutral-800">
              <p className="text-sm italic text-neutral-700 dark:text-neutral-300">
                {explanation}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
