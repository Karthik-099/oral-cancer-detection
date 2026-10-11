import { useEffect, type ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";

export function Modal({ open, onClose, title, children, className }: {
  open: boolean; onClose: () => void; title?: string; children: ReactNode; className?: string;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.removeEventListener("keydown", onKey); document.body.style.overflow = prev; };
  }, [open, onClose]);

  return (
    <AnimatePresence>
      {open && (
        <motion.div className="fixed inset-0 z-[90] flex items-end justify-center p-0 sm:items-center sm:p-6"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
          <div className="absolute inset-0 bg-ink/30 backdrop-blur-sm" onClick={onClose} />
          <motion.div
            role="dialog" aria-modal="true" aria-label={title}
            className={cn("relative max-h-[92vh] w-full overflow-auto rounded-t-3xl border border-line bg-surface p-6 shadow-lift sm:max-w-lg sm:rounded-3xl", className)}
            initial={{ y: 24, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: 16, opacity: 0 }}
            transition={{ type: "spring", damping: 28, stiffness: 320 }}
          >
            {title && (
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-semibold">{title}</h2>
                <button onClick={onClose} className="rounded-lg p-1 text-muted hover:bg-surface-2 hover:text-ink" aria-label="Close"><X className="h-5 w-5" /></button>
              </div>
            )}
            {children}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
