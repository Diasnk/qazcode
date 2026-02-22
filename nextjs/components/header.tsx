import Link from "next/link";

const navLinks = [
	{ href: "/", label: "Home" },
	{ href: "/symptoms", label: "Symptoms" },
	{ href: "/about", label: "About" },
];

export default function Header() {
	return (
		<header className="border-b border-neutral-200/70 bg-white/80 backdrop-blur dark:border-neutral-800/70 dark:bg-neutral-950/80">
			<div className="mx-auto flex max-w-5xl items-center justify-between px-5 py-4">
				<Link href="/" className="text-sm font-semibold uppercase tracking-[0.2em] text-neutral-900 dark:text-white">
					Qaz Health
				</Link>

				<nav className="hidden items-center gap-6 text-sm text-neutral-600 dark:text-neutral-300 md:flex">
					{navLinks.map((link) => (
						<Link
							key={link.href}
							href={link.href}
							className="transition hover:text-neutral-900 dark:hover:text-white"
						>
							{link.label}
						</Link>
					))}
				</nav>
			</div>
		</header>
	);
}
