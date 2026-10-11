import { useState } from "react";
import { Check, Copy } from "lucide-react";
import { cn } from "@/lib/utils";

async function copyText(text: string) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      type="button"
      onClick={async () => {
        if (await copyText(text)) {
          setDone(true);
          setTimeout(() => setDone(false), 1600);
        }
      }}
      className="inline-flex h-7 items-center gap-1.5 rounded-md px-2 text-[11px] font-medium text-muted transition hover:bg-surface-2 hover:text-ink"
      aria-label={done ? "Copied" : "Copy code"}
    >
      {done ? <Check className="h-3.5 w-3.5 text-low" /> : <Copy className="h-3.5 w-3.5" />}
      {done ? "Copied" : "Copy"}
    </button>
  );
}

/**
 * Code sample on a dark surface in both themes. The `dark` class scopes the dark
 * token values to this element, so token colours still apply.
 */
export function CodeBlock({ code, title, className }: { code: string; title?: string; className?: string }) {
  return (
    <div className={cn("dark overflow-hidden rounded-2xl border border-line bg-surface text-ink shadow-soft", className)}>
      <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-2">
        <span className="truncate font-mono text-[11px] text-muted">{title ?? "example"}</span>
        <CopyButton text={code} />
      </div>
      <pre className="overflow-x-auto p-4 text-[12.5px] leading-relaxed"><code className="font-mono">{code}</code></pre>
    </div>
  );
}

/** Tabbed code samples (curl / Python / JavaScript). */
export function CodeTabs({ tabs, className }: { tabs: { label: string; code: string; title?: string }[]; className?: string }) {
  const [active, setActive] = useState(0);
  const tab = tabs[active];
  return (
    <div className={cn("dark overflow-hidden rounded-2xl border border-line bg-surface text-ink shadow-soft", className)}>
      <div className="flex items-center justify-between gap-2 border-b border-line px-2">
        <div role="tablist" aria-label="Language" className="flex overflow-x-auto">
          {tabs.map((t, i) => (
            <button
              key={t.label} role="tab" type="button" aria-selected={i === active} onClick={() => setActive(i)}
              className={cn("relative whitespace-nowrap px-3 py-2.5 text-xs font-medium transition",
                i === active ? "text-ink" : "text-muted hover:text-ink")}
            >
              {t.label}
              {i === active && <span className="absolute inset-x-3 -bottom-px h-px bg-brand" />}
            </button>
          ))}
        </div>
        <CopyButton text={tab.code} />
      </div>
      {tab.title && <div className="border-b border-line px-4 py-1.5 font-mono text-[11px] text-muted">{tab.title}</div>}
      <pre role="tabpanel" className="overflow-x-auto p-4 text-[12.5px] leading-relaxed"><code className="font-mono">{tab.code}</code></pre>
    </div>
  );
}
