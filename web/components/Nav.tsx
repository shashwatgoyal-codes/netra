"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useOnline } from "@/lib/hooks";

const LINKS = [
  { href: "/", label: "Screen", icon: "M12 5c-5 0-9 4.5-10 7 1 2.5 5 7 10 7s9-4.5 10-7c-1-2.5-5-7-10-7Zm0 10.5a3.5 3.5 0 1 1 0-7 3.5 3.5 0 0 1 0 7Z" },
  { href: "/queue/", label: "Referrals", icon: "M5 4h14v4H5zM5 10h14v4H5zM5 16h9v4H5z" },
  { href: "/records/", label: "Records", icon: "M6 3h9l4 4v14H6zM14 3v5h5M9 12h7M9 16h7" },
  { href: "/about/", label: "Model", icon: "M4 19V9M10 19V5M16 19v-7M22 19H2" },
];

function Icon({ d }: { d: string }) {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinejoin="round" strokeLinecap="round" aria-hidden>
      <path d={d} />
    </svg>
  );
}

export default function Nav() {
  const path = usePathname();
  const online = useOnline();
  const active = (href: string) => (href === "/" ? path === "/" : path.startsWith(href));

  return (
    <>
      <header className="sticky top-0 z-20 border-b border-line bg-bg/85 backdrop-blur">
        <div className="mx-auto flex max-w-3xl items-center justify-between px-4 py-3">
          <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight">
            <span className="grid h-8 w-8 place-items-center rounded-full bg-accent text-accent-ink">
              <svg viewBox="0 0 24 24" className="h-5 w-5" fill="currentColor" aria-hidden>
                <circle cx="12" cy="12" r="4" />
                <path d="M12 5c-5 0-9 4.5-10 7 1 2.5 5 7 10 7s9-4.5 10-7c-1-2.5-5-7-10-7Zm0 12a5 5 0 1 1 0-10 5 5 0 0 1 0 10Z" opacity=".55" />
              </svg>
            </span>
            Netra
          </Link>
          <nav className="hidden gap-1 sm:flex">
            {LINKS.map((l) => (
              <Link key={l.href} href={l.href} className={`rounded-lg px-3 py-1.5 text-sm font-medium ${active(l.href) ? "bg-accent-soft text-accent" : "text-muted hover:text-ink"}`}>
                {l.label}
              </Link>
            ))}
          </nav>
          <span className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${online ? "bg-surface-2 text-muted" : "bg-warn-soft text-warn"}`}>
            <span className={`h-1.5 w-1.5 rounded-full ${online ? "bg-routine" : "bg-warn"}`} />
            {online ? "Online" : "Offline"}
          </span>
        </div>
      </header>
      <nav className="fixed inset-x-0 bottom-0 z-20 grid grid-cols-4 border-t border-line bg-surface/95 pb-[env(safe-area-inset-bottom)] backdrop-blur sm:hidden">
        {LINKS.map((l) => (
          <Link key={l.href} href={l.href} className={`flex flex-col items-center gap-0.5 py-2.5 text-[11px] font-medium ${active(l.href) ? "text-accent" : "text-muted"}`}>
            <Icon d={l.icon} />
            {l.label}
          </Link>
        ))}
      </nav>
    </>
  );
}
