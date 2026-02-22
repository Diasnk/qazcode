

'use client';

import { useState } from 'react';

interface Diagnosis {
  rank: number;
  diagnosis: string;
  icd10_code: string;
  explanation: string;
}

interface DiagnosisResponse {
  diagnoses: Diagnosis[];
}

export default function SymptomsPage() {
  const [symptoms, setSymptoms] = useState('');
  const [onset, setOnset] = useState('');
  const [notes, setNotes] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<Diagnosis[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!symptoms.trim()) {
      setError('Please describe your symptoms');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/diagnose', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symptoms: symptoms.trim(),
          onset: onset.trim(),
          notes: notes.trim(),
        }),
      });

      if (!response.ok) {
        throw new Error(`Backend error: ${response.status}`);
      }

      const data: DiagnosisResponse = await response.json();
      setResults(data.diagnoses);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get diagnosis');
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-linear-to-br from-neutral-50 via-blue-50 to-neutral-50 dark:from-neutral-950 dark:via-neutral-900 dark:to-neutral-950">
      <div className="mx-auto max-w-5xl px-5 py-12">
        <div className="mb-12 space-y-4 text-center sm:mb-16">
          <div className="inline-block">
            <div className="rounded-full bg-blue-100/50 px-4 py-1.5 text-sm font-medium text-blue-700 dark:bg-blue-900/30 dark:text-blue-300">
              Patient Assessment
            </div>
          </div>
          <h1 className="text-4xl font-bold tracking-tight text-neutral-900 dark:text-white sm:text-5xl">
            Symptoms & Diagnosis
          </h1>
          <p className="mx-auto max-w-2xl text-lg text-neutral-600 dark:text-neutral-400">
            Describe your symptoms to receive evidence-based differential diagnoses with ICD-10 codes and clinical guidance.
          </p>
        </div>

        <section className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          {/* Input Section */}
          <div className="rounded-xl border border-neutral-200/50 bg-white/60 p-6 backdrop-blur dark:border-neutral-800/50 dark:bg-neutral-900/60 sm:p-8">
            <h2 className="mb-6 text-xl font-semibold text-neutral-900 dark:text-white">
              Tell us about your symptoms
            </h2>

            <form className="grid gap-5" onSubmit={handleSubmit}>
              <label className="grid gap-2 text-sm text-neutral-700 dark:text-neutral-300">
                <span className="font-medium text-neutral-900 dark:text-white">Primary symptoms</span>
                <textarea
                  name="symptoms"
                  placeholder="Example: sore throat, fever, fatigue"
                  rows={6}
                  value={symptoms}
                  onChange={(e) => setSymptoms(e.target.value)}
                  className="w-full resize-y rounded-lg border border-neutral-300/50 bg-white/50 px-4 py-3 text-neutral-900 placeholder:text-neutral-400 transition focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/10 dark:border-neutral-700/50 dark:bg-neutral-950/50 dark:text-neutral-100 dark:placeholder:text-neutral-500 dark:focus:border-blue-400 dark:focus:ring-blue-400/10"
                />
              </label>

              <label className="grid gap-2 text-sm text-neutral-700 dark:text-neutral-300">
                <span className="font-medium text-neutral-900 dark:text-white">When did it start?</span>
                <input
                  type="text"
                  name="onset"
                  placeholder="Example: 3 days ago"
                  value={onset}
                  onChange={(e) => setOnset(e.target.value)}
                  className="w-full rounded-lg border border-neutral-300/50 bg-white/50 px-4 py-3 text-neutral-900 placeholder:text-neutral-400 transition focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/10 dark:border-neutral-700/50 dark:bg-neutral-950/50 dark:text-neutral-100 dark:placeholder:text-neutral-500 dark:focus:border-blue-400 dark:focus:ring-blue-400/10"
                />
              </label>

              <label className="grid gap-2 text-sm text-neutral-700 dark:text-neutral-300">
                <span className="font-medium text-neutral-900 dark:text-white">Additional notes</span>
                <input
                  type="text"
                  name="notes"
                  placeholder="Example: recent travel, allergies, medications"
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  className="w-full rounded-lg border border-neutral-300/50 bg-white/50 px-4 py-3 text-neutral-900 placeholder:text-neutral-400 transition focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/10 dark:border-neutral-700/50 dark:bg-neutral-950/50 dark:text-neutral-100 dark:placeholder:text-neutral-500 dark:focus:border-blue-400 dark:focus:ring-blue-400/10"
                />
              </label>

              <button
                type="submit"
                disabled={loading}
                className="mt-2 w-full rounded-lg bg-blue-600 px-6 py-3 font-medium text-white transition hover:bg-blue-700 disabled:bg-blue-400 dark:bg-blue-500 dark:hover:bg-blue-600 dark:disabled:bg-blue-600"
              >
                {loading ? 'Checking diagnosis...' : 'Check probable diagnosis'}
              </button>
            </form>
          </div>

          {/* Results Section */}
          <div className="rounded-xl border border-neutral-200/50 bg-white/60 p-6 backdrop-blur dark:border-neutral-800/50 dark:bg-neutral-900/60 sm:p-8">
            <h2 className="mb-6 text-xl font-semibold text-neutral-900 dark:text-white">Probable diagnosis</h2>
            <div
              className="grid min-h-56 gap-3 rounded-lg border border-dashed border-neutral-300/50 bg-linear-to-br from-neutral-50/50 to-blue-50/30 p-5 text-neutral-700 dark:border-neutral-700/50 dark:from-neutral-950/50 dark:to-blue-950/30 dark:text-neutral-300"
              aria-live="polite"
            >
              {error && (
                <div className="rounded-lg bg-red-50 p-4 text-red-700 dark:bg-red-950/30 dark:text-red-300">
                  <p className="font-semibold">Error</p>
                  <p className="text-sm">{error}</p>
                </div>
              )}

              {loading && (
                <div className="flex items-center justify-center">
                  <div className="space-y-3 w-full">
                    <p className="font-semibold text-neutral-900 dark:text-neutral-100">Analyzing symptoms...</p>
                    <div className="h-2 w-full rounded-full bg-neutral-200 dark:bg-neutral-700 overflow-hidden">
                      <div className="h-full w-20 bg-blue-600 rounded-full animate-pulse"></div>
                    </div>
                  </div>
                </div>
              )}

              {!loading && !error && !results && (
                <>
                  <p className="font-semibold text-neutral-900 dark:text-neutral-100">No assessment yet</p>
                  <p className="text-sm">
                    Submit your symptoms to generate a differential diagnosis summary with:
                  </p>
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-start gap-2">
                      <svg className="mt-0.5 h-4 w-4 shrink-0 text-blue-600 dark:text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                      <span>Ranked differential diagnoses</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <svg className="mt-0.5 h-4 w-4 shrink-0 text-blue-600 dark:text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                      <span>ICD-10 codes for each diagnosis</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <svg className="mt-0.5 h-4 w-4 shrink-0 text-blue-600 dark:text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                      <span>Clinical guidance per Kazakhstan protocols</span>
                    </li>
                  </ul>
                </>
              )}

              {!loading && results && results.length > 0 && (
                <div className="space-y-4">
                  <p className="font-semibold text-neutral-900 dark:text-neutral-100">Top diagnoses:</p>
                  {results.map((diagnosis) => (
                    <div
                      key={diagnosis.rank}
                      className="rounded-lg border border-neutral-200/50 bg-white/50 p-4 dark:border-neutral-700/50 dark:bg-neutral-950/50"
                    >
                      <div className="mb-2 flex items-start justify-between gap-2">
                        <p className="font-semibold text-neutral-900 dark:text-neutral-100">
                          {diagnosis.rank}. {diagnosis.diagnosis}
                        </p>
                        <span className="inline-block whitespace-nowrap rounded bg-blue-100/50 px-2 py-1 text-xs font-medium text-blue-700 dark:bg-blue-900/30 dark:text-blue-300">
                          {diagnosis.icd10_code}
                        </span>
                      </div>
                      <p className="text-sm text-neutral-600 dark:text-neutral-400">{diagnosis.explanation}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}