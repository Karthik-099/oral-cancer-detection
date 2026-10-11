import { cn } from "@/lib/utils";

/** Mark: an open mouth arc with a scan line, drawn as a single stroke. */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={cn("h-7 w-7", className)} aria-hidden>
      <rect width="32" height="32" rx="9" className="fill-ink" />
      <path d="M8 13c2.2 6.5 13.8 6.5 16 0" className="stroke-bg" strokeWidth="2.4" fill="none" strokeLinecap="round" />
      <path d="M9 19.5c2.4 3.4 11.6 3.4 14 0" className="stroke-bg/50" strokeWidth="1.6" fill="none" strokeLinecap="round" />
      <circle cx="23.5" cy="9" r="2.2" className="fill-brand" />
    </svg>
  );
}

export function Logo({ className }: { className?: string }) {
  return (
    <span className={cn("inline-flex items-center gap-2", className)}>
      <LogoMark />
      <span className="text-[17px] font-semibold tracking-tight">OralAID</span>
    </span>
  );
}
