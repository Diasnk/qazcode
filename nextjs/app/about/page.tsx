import Link from "next/link";

export default function About() {
  return (
    <main className="min-h-screen bg-linear-to-br from-neutral-50 via-blue-50 to-neutral-50 dark:from-neutral-950 dark:via-neutral-900 dark:to-neutral-950">
      <div className="mx-auto max-w-4xl px-5 py-16 sm:py-24">
        {/* Header */}
        <div className="mb-16 space-y-4 text-center sm:mb-20">
          <h1 className="text-4xl font-bold tracking-tight text-neutral-900 dark:text-white sm:text-5xl">
            About Qaz Health
          </h1>
          <p className="text-lg text-neutral-600 dark:text-neutral-400">
            Supporting clinical decision-making through evidence-based diagnostic assistance
          </p>
        </div>

        {/* Mission Section */}
        <div className="mb-16 rounded-xl border border-neutral-200/50 bg-white/60 p-8 backdrop-blur dark:border-neutral-800/50 dark:bg-neutral-900/60 sm:mb-20">
          <h2 className="mb-4 text-2xl font-semibold text-neutral-900 dark:text-white">
            Our Mission
          </h2>
          <p className="text-neutral-700 dark:text-neutral-300">
            Qaz Health is dedicated to enhancing clinical diagnostics by providing healthcare professionals with evidence-based differential diagnosis support. We combine international medical standards (ICD-10) with Kazakhstan's clinical protocols to deliver reliable, accessible diagnostic recommendations based on patient symptoms.
          </p>
        </div>

        {/* Core Values */}
        <div className="mb-16 sm:mb-20">
          <h2 className="mb-8 text-center text-2xl font-semibold text-neutral-900 dark:text-white">
            Core Values
          </h2>
          <div className="grid gap-6 sm:grid-cols-3">
            <div className="rounded-lg border border-neutral-200/50 bg-white/40 p-6 backdrop-blur dark:border-neutral-800/50 dark:bg-neutral-900/40">
              <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-lg bg-blue-100 dark:bg-blue-900/50">
                <svg
                  className="h-6 w-6 text-blue-600 dark:text-blue-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12l2 2 4-4m7 0a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>
              <h3 className="mb-2 font-semibold text-neutral-900 dark:text-white">
                Evidence-Based
              </h3>
              <p className="text-sm text-neutral-600 dark:text-neutral-400">
                All diagnoses backed by clinical research and official medical protocols
              </p>
            </div>

            <div className="rounded-lg border border-neutral-200/50 bg-white/40 p-6 backdrop-blur dark:border-neutral-800/50 dark:bg-neutral-900/40">
              <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-lg bg-cyan-100 dark:bg-cyan-900/50">
                <svg
                  className="h-6 w-6 text-cyan-600 dark:text-cyan-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M13 10V3L4 14h7v7l9-11h-7z"
                  />
                </svg>
              </div>
              <h3 className="mb-2 font-semibold text-neutral-900 dark:text-white">
                Accessible
              </h3>
              <p className="text-sm text-neutral-600 dark:text-neutral-400">
                Simple interface making diagnostic support available to all practitioners
              </p>
            </div>

            <div className="rounded-lg border border-neutral-200/50 bg-white/40 p-6 backdrop-blur dark:border-neutral-800/50 dark:bg-neutral-900/40">
              <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-lg bg-emerald-100 dark:bg-emerald-900/50">
                <svg
                  className="h-6 w-6 text-emerald-600 dark:text-emerald-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"
                  />
                </svg>
              </div>
              <h3 className="mb-2 font-semibold text-neutral-900 dark:text-white">
                Localized
              </h3>
              <p className="text-sm text-neutral-600 dark:text-neutral-400">
                Built specifically for Kazakhstan clinical practices and protocols
              </p>
            </div>
          </div>
        </div>

        {/* How It Works */}
        <div className="mb-16 sm:mb-20">
          <h2 className="mb-8 text-center text-2xl font-semibold text-neutral-900 dark:text-white">
            How It Works
          </h2>
          <div className="space-y-6">
            <div className="flex gap-4 sm:gap-6">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-blue-100 font-semibold text-blue-600 dark:bg-blue-900/50 dark:text-blue-400 sm:h-12 sm:w-12">
                1
              </div>
              <div>
                <h3 className="mb-1 font-semibold text-neutral-900 dark:text-white">
                  Enter Symptoms
                </h3>
                <p className="text-neutral-600 dark:text-neutral-400">
                  Describe patient symptoms in a simple, intuitive interface
                </p>
              </div>
            </div>

            <div className="flex gap-4 sm:gap-6">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-blue-100 font-semibold text-blue-600 dark:bg-blue-900/50 dark:text-blue-400 sm:h-12 sm:w-12">
                2
              </div>
              <div>
                <h3 className="mb-1 font-semibold text-neutral-900 dark:text-white">
                  Receive Analysis
                </h3>
                <p className="text-neutral-600 dark:text-neutral-400">
                  Get ranked differential diagnoses with probability scores
                </p>
              </div>
            </div>

            <div className="flex gap-4 sm:gap-6">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-blue-100 font-semibold text-blue-600 dark:bg-blue-900/50 dark:text-blue-400 sm:h-12 sm:w-12">
                3
              </div>
              <div>
                <h3 className="mb-1 font-semibold text-neutral-900 dark:text-white">
                  Review Evidence
                </h3>
                <p className="text-neutral-600 dark:text-neutral-400">
                  Explore ICD-10 codes and clinical explanations for each diagnosis
                </p>
              </div>
            </div>

            <div className="flex gap-4 sm:gap-6">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-blue-100 font-semibold text-blue-600 dark:bg-blue-900/50 dark:text-blue-400 sm:h-12 sm:w-12">
                4
              </div>
              <div>
                <h3 className="mb-1 font-semibold text-neutral-900 dark:text-white">
                  Make Informed Decisions
                </h3>
                <p className="text-neutral-600 dark:text-neutral-400">
                  Use diagnostic support to guide clinical decision-making
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Standards Section */}
        <div className="mb-16 rounded-xl border border-neutral-200/50 bg-white/60 p-8 backdrop-blur dark:border-neutral-800/50 dark:bg-neutral-900/60 sm:mb-20">
          <h2 className="mb-4 text-2xl font-semibold text-neutral-900 dark:text-white">
            Medical Standards
          </h2>
          <p className="mb-6 text-neutral-700 dark:text-neutral-300">
            Qaz Health adheres to internationally recognized diagnostic standards:
          </p>
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="flex items-start gap-3">
              <svg className="h-5 w-5 shrink-0 text-blue-600 dark:text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
              <div>
                <p className="font-semibold text-neutral-900 dark:text-white">ICD-10-CM</p>
                <p className="text-sm text-neutral-600 dark:text-neutral-400">International Classification of Diseases</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <svg className="h-5 w-5 shrink-0 text-blue-600 dark:text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
              <div>
                <p className="font-semibold text-neutral-900 dark:text-white">Kazakhstan Protocols</p>
                <p className="text-sm text-neutral-600 dark:text-neutral-400">Local clinical guidelines and best practices</p>
              </div>
            </div>
          </div>
        </div>

        {/* CTA Section */}
        <div className="rounded-2xl border border-neutral-200/50 bg-linear-to-br from-blue-50 to-cyan-50 p-8 text-center dark:border-neutral-800/50 dark:from-blue-900/20 dark:to-cyan-900/20 sm:p-12">
          <h2 className="text-2xl font-bold text-neutral-900 dark:text-white sm:text-3xl">
            Ready to get started?
          </h2>
          <p className="mx-auto mt-3 max-w-xl text-neutral-600 dark:text-neutral-300">
            Experience evidence-based diagnostic support with ICD-10 codes and Kazakhstan protocols.
          </p>
          <Link
            href="/symptoms"
            className="mt-6 inline-flex items-center rounded-lg bg-blue-600 px-6 py-3 font-medium text-white transition hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600"
          >
            Begin Assessment
            <svg className="ml-2 h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </Link>
        </div>
      </div>
    </main>
  );
}
