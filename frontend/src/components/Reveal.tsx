import { useEffect, useRef } from "react";
import { motion, useAnimationControls, useInView, useReducedMotion, type HTMLMotionProps } from "framer-motion";
import { scrollDirection } from "@/lib/scroll";

const EASE = [0.16, 1, 0.3, 1] as const;

/**
 * Scroll-driven entrance that replays in both directions: content rises from
 * below while scrolling down and drops in from above while scrolling up, with
 * a slight 3D tilt and blur that resolve as it settles.
 */
export function Reveal({ delay = 0, style, children, ...props }: HTMLMotionProps<"div"> & { delay?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { amount: 0.12, margin: "0px 0px -6% 0px" });
  const reduce = useReducedMotion();
  const controls = useAnimationControls();

  useEffect(() => {
    const d = scrollDirection();
    const hidden = reduce
      ? { opacity: 0 }
      : { opacity: 0, y: 64 * d, rotateX: 14 * d, scale: 0.96, filter: "blur(8px)" };
    if (inView) {
      controls.set(hidden);
      controls.start({ opacity: 1, y: 0, rotateX: 0, scale: 1, filter: "blur(0px)", transition: { duration: 0.9, delay, ease: EASE } });
    } else {
      controls.start({ ...hidden, transition: { duration: 0.35, ease: "easeIn" } });
    }
  }, [inView, reduce, delay, controls]);

  return (
    <motion.div ref={ref} initial={{ opacity: 0 }} animate={controls}
      style={{ transformPerspective: 1200, transformOrigin: "50% 100%", willChange: "transform, opacity", ...style }} {...props}>
      {children}
    </motion.div>
  );
}

/** Mouse-driven 3D tilt with a moving highlight, for hero visuals. */
export function Tilt({ children, className, max = 10 }: { children: React.ReactNode; className?: string; max?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const controls = useAnimationControls();
  const reduce = useReducedMotion();
  return (
    <motion.div
      ref={ref}
      className={className}
      animate={controls}
      style={{ transformPerspective: 1100, transformStyle: "preserve-3d" }}
      onPointerMove={(e) => {
        if (reduce || e.pointerType !== "mouse") return;
        const b = ref.current!.getBoundingClientRect();
        const x = (e.clientX - b.left) / b.width - 0.5;
        const y = (e.clientY - b.top) / b.height - 0.5;
        controls.start({ rotateY: x * max * 2, rotateX: -y * max * 2, transition: { type: "spring", stiffness: 150, damping: 18 } });
      }}
      onPointerLeave={() => controls.start({ rotateX: 0, rotateY: 0, transition: { type: "spring", stiffness: 120, damping: 16 } })}
    >
      {children}
    </motion.div>
  );
}
