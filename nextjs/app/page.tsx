"use client";

import Link from "next/link";
import { useState } from "react";

interface Diagnosis {
  rank: number;
  name: string;
  probability: number;
  icd10: string;
  explanation: string;
}

export default function Home() {
  const [isHovered, setIsHovered] = useState<number | null>(null);

  const exampleDiagnoses: Diagnosis[] = [
    {
      rank: 1,
      name: "Acute Bronchitis",
      probability: 87,
      icd10: "J20.9",
      explanation: "Inflammation of the bronchial tubes, commonly following viral upper respiratory infections. Managed with supportive care and symptomatic treatment per Kazakhstan clinical protocols.",
    },
    {
      rank: 2,
      name: "Viral Upper Respiratory Infection",
      probability: 72,
      icd10: "J06.9",
      explanation: "Common viral infection affecting nose, throat, and larynx. Self-limiting condition typically resolving within 7-10 days with supportive treatment.",
    },
    {
      rank: 3,
      name: "Community-Acquired Pneumonia",
      probability: 43,
      icd10: "J18.9",
      explanation: "Infectious inflammation of lung parenchyma. Diagnosis confirmed with chest imaging, managed with antibiotics according to severity and local resistance patterns.",
    },
  ];

  return (
    <main className="min-h-screen bg-linear-to-br from-neutral-50 via-blue-50 to-neutral-50 dark:from-neutral-950 dark:via-neutral-900 dark:to-neutral-950">
      <div className="mx-auto max-w-4xl px-5 py-16 sm:py-24">

        <div className="mb-16 space-y-6 text-center sm:mb-20">
          <div className="inline-block">
            <div className="rounded-full bg-blue-100/50 px-4 py-1.5 text-sm font-medium text-blue-700 dark:bg-blue-900/30 dark:text-blue-300">
              Clinical Decision Support
            </div>
          </div>

          <h1 className="text-4xl font-bold tracking-tight text-neutral-900 dark:text-white sm:text-5xl lg:text-6xl">
            Differential Diagnosis
            <span className="block bg-linear-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent dark:from-blue-400 dark:to-cyan-300">
              Assistant
            </span>
          </h1>

          <p className="mx-auto max-w-2xl text-lg text-neutral-600 dark:text-neutral-300">
            Evidence-based clinical recommendations with ICD-10 codes and Kazakhstan protocol guidelines to support your diagnostic process.
          </p>

          <div className="flex flex-col justify-center gap-3 sm:flex-row">
            <Link
              href="/symptoms"
              className="inline-flex items-center justify-center rounded-lg bg-blue-600 px-6 py-3 font-medium text-white transition hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 sm:px-8"
            >
              Start Diagnosis
              <svg className="ml-2 h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </Link>
            <Link href="/about" className="inline-flex items-center justify-center rounded-lg border border-neutral-300 px-6 py-3 font-medium text-neutral-700 transition hover:bg-neutral-50 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-800/50 sm:px-8">
              Learn More
            </Link>
          </div>
        </div>

        {/* Features Section */}
        <div className="mb-16 grid gap-6 sm:mb-20 sm:grid-cols-3">
          <div className="rounded-xl border border-neutral-200/50 bg-white/40 p-6 backdrop-blur dark:border-neutral-800/50 dark:bg-neutral-900/40">
            <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-lg bg-blue-100 dark:bg-blue-900/50">
              <svg className="h-5 w-5 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h3 className="mb-2 font-semibold text-neutral-900 dark:text-white">Ranked Diagnoses</h3>
            <p className="text-sm text-neutral-600 dark:text-neutral-400">
              Top-N probable diagnoses ranked by likelihood based on symptom analysis
            </p>
          </div>

          <div className="rounded-xl border border-neutral-200/50 bg-white/40 p-6 backdrop-blur dark:border-neutral-800/50 dark:bg-neutral-900/40">
            <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-lg bg-cyan-100 dark:bg-cyan-900/50">
              <svg className="h-5 w-5 text-cyan-600 dark:text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="mb-2 font-semibold text-neutral-900 dark:text-white">ICD-10 Codes</h3>
            <p className="text-sm text-neutral-600 dark:text-neutral-400">
              International Classification of Diseases codes for each diagnosis
            </p>
          </div>

          <div className="rounded-xl border border-neutral-200/50 bg-white/40 p-6 backdrop-blur dark:border-neutral-800/50 dark:bg-neutral-900/40">
            <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-lg bg-emerald-100 dark:bg-emerald-900/50">
              <svg className="h-5 w-5 text-emerald-600 dark:text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C6.5 6.253 2 10.998 2 17.25m20 0c0-6.252-4.5-10.997-10-10.997z" />
              </svg>
            </div>
            <h3 className="mb-2 font-semibold text-neutral-900 dark:text-white">Clinical Protocols</h3>
            <p className="text-sm text-neutral-600 dark:text-neutral-400">
              Evidence-based explanations based on Kazakhstan clinical guidelines
            </p>
          </div>
        </div>

        {/* Example Diagnoses Section */}
        <div className="mb-12 sm:mb-16">
          <div className="mb-8 text-center">
            <h2 className="text-2xl font-bold text-neutral-900 dark:text-white sm:text-3xl">
              Example Results
            </h2>
            <p className="mt-2 text-neutral-600 dark:text-neutral-400">
              See how diagnoses are presented with probability scores and clinical information
            </p>
          </div>

          <div className="space-y-4">
            {exampleDiagnoses.map((diagnosis) => (
              <div
                key={diagnosis.rank}
                onMouseEnter={() => setIsHovered(diagnosis.rank)}
                onMouseLeave={() => setIsHovered(null)}
                className="group cursor-pointer rounded-xl border border-neutral-200/50 bg-white/60 p-6 transition hover:border-blue-300/50 hover:bg-white/80 hover:shadow-lg dark:border-neutral-800/50 dark:bg-neutral-900/60 dark:hover:border-blue-700/50 dark:hover:bg-neutral-900/80 dark:hover:shadow-lg dark:hover:shadow-blue-900/20"
              >
                <div className="flex items-start gap-4">
                  {/* Rank Badge */}
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-linear-to-br from-blue-600 to-cyan-600 text-sm font-bold text-white dark:from-blue-400 dark:to-cyan-300 dark:text-neutral-900">
                    {diagnosis.rank}
                  </div>

                  {/* Content */}
                  <div className="flex-1">
                    <div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-center">
                      <div>
                        <h3 className="text-lg font-semibold text-neutral-900 dark:text-white">
                          {diagnosis.name}
                        </h3>
                        <p className="text-sm text-neutral-500 dark:text-neutral-400">
                          ICD-10: <span className="font-mono font-semibold text-blue-600 dark:text-blue-400">{diagnosis.icd10}</span>
                        </p>
                      </div>

                      {/* Probability */}
                      <div className="text-right">
                        <div className="text-2xl font-bold text-neutral-900 dark:text-white">
                          {diagnosis.probability}%
                        </div>
                        <div className="text-xs text-neutral-500 dark:text-neutral-400">
                          Likelihood
                        </div>
                      </div>
                    </div>

                    {/* Probability Bar */}
                    <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-800">
                      <div
                        className={`h-full transition-all duration-500 ${
                          diagnosis.probability >= 80
                            ? "bg-emerald-500"
                            : diagnosis.probability >= 60
                              ? "bg-blue-500"
                              : "bg-amber-500"
                        }`}
                        style={{ width: `${diagnosis.probability}%` }}
                      />
                    </div>

                    {/* Expandable Explanation */}
                    {isHovered === diagnosis.rank && (
                      <div className="mt-4 border-t border-neutral-200 pt-4 dark:border-neutral-800">
                        <p className="text-sm italic text-neutral-700 dark:text-neutral-300">
                          {diagnosis.explanation}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* CTA Section */}
        <div className="rounded-2xl border border-neutral-200/50 bg-linear-to-br from-blue-50 to-cyan-50 p-8 text-center dark:border-neutral-800/50 dark:from-blue-900/20 dark:to-cyan-900/20 sm:p-12">
          <h2 className="text-2xl font-bold text-neutral-900 dark:text-white sm:text-3xl">
            Ready to get started?
          </h2>
          <p className="mx-auto mt-3 max-w-xl text-neutral-600 dark:text-neutral-300">
            Enter your symptoms and receive a comprehensive analysis with differential diagnoses, ICD-10 codes, and clinical recommendations.
          </p>
          <Link
            href="/symptoms"
            className="mt-6 inline-flex items-center rounded-lg bg-blue-600 px-6 py-3 font-medium text-white transition hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600"
          >
            Begin Diagnosis Assessment
            <svg className="ml-2 h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </Link>
        </div>
      </div>
    </main>
  );
}
