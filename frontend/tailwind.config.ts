import type { Config } from "tailwindcss";

const c = (v: string) => `rgb(var(--${v}) / <alpha-value>)`;

const config: Config = {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    container: { center: true, padding: "1.25rem", screens: { "2xl": "1240px" } },
    extend: {
      colors: {
        bg: c("bg"),
        surface: c("surface"),
        "surface-2": c("surface-2"),
        ink: c("ink"),
        muted: c("muted"),
        line: c("line"),
        brand: c("brand"),
        "brand-soft": c("brand-soft"),
        accent: c("accent"),
        low: c("low"),
        moderate: c("moderate"),
        high: c("high"),
      },
      fontFamily: {
        sans: ["'Space Grotesk'", "ui-sans-serif", "system-ui", "sans-serif"],
        display: ["'Chakra Petch'", "'Space Grotesk'", "ui-sans-serif", "sans-serif"],
        mono: ["'JetBrains Mono'", "ui-monospace", "monospace"],
      },
      borderRadius: { xl: "14px", "2xl": "20px", "3xl": "28px" },
      boxShadow: {
        soft: "0 1px 2px rgb(0 0 0 / 0.04), 0 8px 24px -12px rgb(0 0 0 / 0.12)",
        lift: "0 2px 4px rgb(0 0 0 / 0.04), 0 24px 48px -24px rgb(0 0 0 / 0.25)",
      },
      keyframes: {
        "fade-up": { from: { opacity: "0", transform: "translateY(8px)" }, to: { opacity: "1", transform: "none" } },
        scan: { "0%": { transform: "translateY(-100%)" }, "100%": { transform: "translateY(100%)" } },
        shimmer: { "100%": { transform: "translateX(100%)" } },
        drift: {
          "0%,100%": { transform: "translate3d(0,0,0) scale(1)" },
          "33%": { transform: "translate3d(6%,-4%,0) scale(1.08)" },
          "66%": { transform: "translate3d(-5%,5%,0) scale(.95)" },
        },
        "spin-slow": { to: { transform: "rotate(360deg)" } },
        "gradient-x": { "0%,100%": { backgroundPosition: "0% 50%" }, "50%": { backgroundPosition: "100% 50%" } },
      },
      animation: {
        "fade-up": "fade-up .5s cubic-bezier(.2,.7,.2,1) both",
        scan: "scan 2.4s cubic-bezier(.6,0,.4,1) infinite",
        shimmer: "shimmer 1.6s infinite",
        drift: "drift 22s ease-in-out infinite",
        "spin-slow": "spin-slow 40s linear infinite",
        "gradient-x": "gradient-x 8s ease infinite",
      },
    },
  },
  plugins: [],
};

export default config;
