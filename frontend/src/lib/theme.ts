import { create } from "zustand";

// One shared store so every toggle (site header, sidebar, settings) stays in sync.
const useThemeStore = create<{ dark: boolean; set: (dark: boolean) => void }>((set) => ({
  dark: typeof document !== "undefined" && document.documentElement.classList.contains("dark"),
  set: (dark) => {
    document.documentElement.classList.toggle("dark", dark);
    try { localStorage.setItem("oralaid-theme", dark ? "dark" : "light"); } catch { /* private mode */ }
    set({ dark });
  },
}));

export function useTheme() {
  const { dark, set } = useThemeStore();
  return { dark, toggle: () => set(!dark), setDark: set };
}
