import { Check } from "lucide-react";
import { cn } from "@/lib/utils";

export function passwordChecks(password: string, email = "") {
  const local = email.split("@")[0]?.toLowerCase() ?? "";
  return [
    { label: "At least 10 characters", ok: password.length >= 10 },
    { label: "Letters and numbers", ok: /[a-z]/i.test(password) && /\d/.test(password) },
    { label: "Doesn't contain your email name", ok: !local || local.length < 3 || !password.toLowerCase().includes(local) },
  ];
}

export function PasswordChecklist({ password, email }: { password: string; email?: string }) {
  const checks = passwordChecks(password, email);
  const score = checks.filter((c) => c.ok).length + (password.length >= 14 ? 1 : 0) + (/[^a-z0-9]/i.test(password) ? 1 : 0);
  const strength = !password ? 0 : Math.min(4, Math.max(1, score - 1));
  const labels = ["", "Weak", "Fair", "Good", "Strong"];
  const tones = ["bg-line", "bg-high", "bg-moderate", "bg-low", "bg-low"];
  return (
    <div className="mt-2.5 space-y-2.5" aria-live="polite">
      <div className="flex items-center gap-2">
        <div className="flex flex-1 gap-1">
          {[1, 2, 3, 4].map((i) => (
            <span key={i} className={cn("h-1 flex-1 rounded-full transition-colors duration-300", i <= strength ? tones[strength] : "bg-surface-2")} />
          ))}
        </div>
        <span className="w-12 text-right text-xs text-muted">{labels[strength]}</span>
      </div>
      <ul className="space-y-1">
        {checks.map((c) => (
          <li key={c.label} className={cn("flex items-center gap-2 text-xs transition-colors", c.ok ? "text-low" : "text-muted")}>
            <span className={cn("grid h-4 w-4 place-items-center rounded-full border transition", c.ok ? "border-low bg-low text-white dark:text-bg" : "border-line")}>
              {c.ok && <Check className="h-2.5 w-2.5" strokeWidth={3} />}
            </span>
            {c.label}
          </li>
        ))}
      </ul>
    </div>
  );
}
