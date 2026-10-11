import { create } from "zustand";
import { api, onUnauthorized, tokenStore, type User } from "./api";

interface AuthState {
  user: User | null;
  status: "idle" | "loading" | "ready";
  setSession: (token: string, user: User) => void;
  setUser: (user: User) => void;
  bootstrap: () => Promise<void>;
  signOut: () => void;
}

export const useAuth = create<AuthState>((set, get) => ({
  user: null,
  status: "idle",
  setSession: (token, user) => {
    tokenStore.set(token);
    set({ user, status: "ready" });
  },
  setUser: (user) => set({ user }),
  bootstrap: async () => {
    if (get().status !== "idle") return;
    if (!tokenStore.get()) return set({ status: "ready" });
    set({ status: "loading" });
    try {
      set({ user: await api.me(), status: "ready" });
    } catch {
      tokenStore.set(null);
      set({ user: null, status: "ready" });
    }
  },
  signOut: () => {
    tokenStore.set(null);
    set({ user: null, status: "ready" });
  },
}));

onUnauthorized(() => useAuth.getState().signOut());
