import { forwardRef, type ButtonHTMLAttributes, type HTMLAttributes, type ReactNode } from "react";
import { Link, type LinkProps } from "react-router-dom";
import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

type Variant = "primary" | "secondary" | "ghost" | "danger" | "ink";
type Size = "sm" | "md" | "lg";

const variants: Record<Variant, string> = {
  primary: "text-white bg-[linear-gradient(110deg,rgb(var(--glow-1)),rgb(var(--glow-3))_50%,rgb(var(--glow-2)))] bg-[length:200%_100%] bg-left hover:bg-right shadow-[0_1px_0_rgb(255_255_255/.25)_inset,0_10px_30px_-10px_rgb(var(--glow-3)/.8)] [transition:background-position_.6s,transform_.15s]",
  ink: "bg-ink text-bg hover:bg-ink/90",
  secondary: "border border-line bg-surface text-ink hover:bg-surface-2",
  ghost: "text-ink hover:bg-surface-2",
  danger: "bg-high text-white hover:bg-high/90",
};
const sizes: Record<Size, string> = {
  sm: "h-8 px-3 text-[13px] gap-1.5 rounded-lg",
  md: "h-10 px-4 text-sm gap-2 rounded-xl",
  lg: "h-12 px-6 text-[15px] gap-2 rounded-xl",
};

export const buttonClass = (variant: Variant = "primary", size: Size = "md", className?: string) =>
  cn(
    "inline-flex items-center justify-center whitespace-nowrap font-medium transition active:scale-[.98] disabled:pointer-events-none disabled:opacity-50",
    variants[variant],
    sizes[size],
    className,
  );

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant, size, loading, className, children, disabled, ...props }, ref) => (
    <button ref={ref} className={buttonClass(variant, size, className)} disabled={disabled || loading} {...props}>
      {loading && <Loader2 className="h-4 w-4 animate-spin" />}
      {children}
    </button>
  ),
);
Button.displayName = "Button";

export function ButtonLink({ variant, size, className, ...props }: LinkProps & { variant?: Variant; size?: Size }) {
  return <Link className={buttonClass(variant, size, className)} {...props} />;
}

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("card", className)} {...props} />;
}

export function Badge({ className, tone = "neutral", ...props }: HTMLAttributes<HTMLSpanElement> & {
  tone?: "neutral" | "brand" | "low" | "moderate" | "high" | "accent";
}) {
  const tones = {
    neutral: "bg-surface-2 text-muted ring-line",
    brand: "bg-brand/10 text-brand ring-brand/20",
    accent: "bg-accent/10 text-accent ring-accent/20",
    low: "bg-low/10 text-low ring-low/25",
    moderate: "bg-moderate/10 text-moderate ring-moderate/25",
    high: "bg-high/10 text-high ring-high/25",
  };
  return (
    <span
      className={cn("inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset", tones[tone], className)}
      {...props}
    />
  );
}

export function Field({ label, hint, error, children, className }: {
  label: string; hint?: ReactNode; error?: string | null; children: ReactNode; className?: string;
}) {
  return (
    <label className={cn("block", className)}>
      <span className="label">{label}</span>
      {children}
      {error ? <span className="mt-1.5 block text-xs text-high">{error}</span>
        : hint ? <span className="mt-1.5 block text-xs text-muted">{hint}</span> : null}
    </label>
  );
}

export function Spinner({ className }: { className?: string }) {
  return <Loader2 className={cn("h-5 w-5 animate-spin text-muted", className)} />;
}

export function EmptyState({ icon, title, body, action }: { icon: ReactNode; title: string; body: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-line px-6 py-14 text-center">
      <div className="mb-4 grid h-12 w-12 place-items-center rounded-2xl bg-surface-2 text-muted">{icon}</div>
      <h3 className="text-base font-semibold">{title}</h3>
      <p className="mt-1 max-w-sm text-sm text-muted">{body}</p>
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="rounded-md border border-line bg-surface-2 px-1.5 py-0.5 font-mono text-[11px] text-muted">{children}</kbd>;
}

export function ErrorNote({ children }: { children: ReactNode }) {
  if (!children) return null;
  return <div role="alert" className="rounded-xl border border-high/25 bg-high/5 px-3.5 py-2.5 text-sm text-high">{children}</div>;
}
