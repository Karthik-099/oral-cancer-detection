import { useEffect } from "react";
import { motion, useReducedMotion, useScroll, useSpring, useTransform } from "framer-motion";
import Lenis from "lenis";

/** Drifting gradient light behind the whole site, with gentle scroll parallax. */
export function Aurora() {
  const { scrollYProgress } = useScroll();
  const y1 = useTransform(scrollYProgress, [0, 1], ["0%", "-30%"]);
  const y2 = useTransform(scrollYProgress, [0, 1], ["0%", "25%"]);
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <div className="absolute inset-0 bg-bg" />
      <motion.div style={{ y: y1 }} className="absolute -left-[15%] -top-[20%] h-[70vmax] w-[70vmax] animate-drift rounded-full bg-[radial-gradient(circle,rgb(var(--glow-1)/.20),transparent_62%)] blur-3xl" />
      <motion.div style={{ y: y2 }} className="absolute -right-[20%] top-[10%] h-[65vmax] w-[65vmax] animate-drift rounded-full bg-[radial-gradient(circle,rgb(var(--glow-2)/.18),transparent_62%)] blur-3xl [animation-delay:-8s]" />
      <div className="absolute bottom-[-30%] left-[20%] h-[60vmax] w-[60vmax] animate-drift rounded-full bg-[radial-gradient(circle,rgb(var(--glow-3)/.12),transparent_62%)] blur-3xl [animation-delay:-14s]" />
      <div className="absolute inset-0 bg-[linear-gradient(180deg,transparent,rgb(var(--bg)/.6)_70%,rgb(var(--bg)))]" />
    </div>
  );
}

/** Thin gradient bar showing reading progress. */
export function ScrollProgress() {
  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, { stiffness: 120, damping: 24, restDelta: 0.001 });
  return (
    <motion.div aria-hidden style={{ scaleX }}
      className="fixed inset-x-0 top-0 z-[60] h-[2px] origin-left bg-[linear-gradient(90deg,rgb(var(--glow-1)),rgb(var(--glow-3)),rgb(var(--glow-2)))]" />
  );
}

/** Inertial smooth scrolling for the marketing site (skipped for reduced motion). */
export function useSmoothScroll() {
  const reduce = useReducedMotion();
  useEffect(() => {
    if (reduce) return;
    const lenis = new Lenis({ duration: 1.15, easing: (t) => 1 - Math.pow(1 - t, 4), smoothWheel: true });
    let id = 0;
    const raf = (time: number) => { lenis.raf(time); id = requestAnimationFrame(raf); };
    id = requestAnimationFrame(raf);
    return () => { cancelAnimationFrame(id); lenis.destroy(); };
  }, [reduce]);
}

/** Feeds the cursor position to .spotlight surfaces (one listener for the whole page). */
export function useSpotlight() {
  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      const el = (e.target as Element | null)?.closest?.(".spotlight") as HTMLElement | null;
      if (!el) return;
      const r = el.getBoundingClientRect();
      el.style.setProperty("--mx", `${e.clientX - r.left}px`);
      el.style.setProperty("--my", `${e.clientY - r.top}px`);
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    return () => window.removeEventListener("pointermove", onMove);
  }, []);
}
