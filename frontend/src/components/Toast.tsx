import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle2, AlertCircle, X } from "lucide-react";
import { create } from "zustand";

interface Toast { id: number; message: string; tone: "success" | "error" }
interface ToastState { toasts: Toast[]; push: (message: string, tone?: Toast["tone"]) => void; dismiss: (id: number) => void }

export const useToast = create<ToastState>((set) => ({
  toasts: [],
  push: (message, tone = "success") => {
    const id = Date.now() + Math.random();
    set((s) => ({ toasts: [...s.toasts, { id, message, tone }] }));
    setTimeout(() => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })), 4200);
  },
  dismiss: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
}));

export const toast = {
  success: (m: string) => useToast.getState().push(m, "success"),
  error: (m: string) => useToast.getState().push(m, "error"),
};

export function Toaster() {
  const { toasts, dismiss } = useToast();
  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-4 z-[100] flex flex-col items-center gap-2 px-4" aria-live="polite">
      <AnimatePresence>
        {toasts.map((t) => (
          <motion.div
            key={t.id}
            initial={{ opacity: 0, y: 12, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 6, scale: 0.98 }}
            className="pointer-events-auto flex max-w-md items-center gap-3 rounded-xl border border-line bg-surface px-4 py-3 text-sm shadow-lift"
          >
            {t.tone === "success" ? <CheckCircle2 className="h-4 w-4 shrink-0 text-low" /> : <AlertCircle className="h-4 w-4 shrink-0 text-high" />}
            <span className="flex-1">{t.message}</span>
            <button onClick={() => dismiss(t.id)} className="text-muted hover:text-ink" aria-label="Dismiss"><X className="h-4 w-4" /></button>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
