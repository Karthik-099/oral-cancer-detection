import { Component, type ErrorInfo, type ReactNode } from "react";

/** Keeps one broken page from blanking the whole site. */
export class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Page crashed", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="grid min-h-[70vh] place-items-center px-6 text-center">
        <div className="max-w-md">
          <p className="eyebrow mb-3">Something went wrong</p>
          <h1 className="display text-4xl">This page hit an error.</h1>
          <p className="mt-3 text-muted">Reload to try again. If it keeps happening, the service may be temporarily unavailable.</p>
          <div className="mt-6 flex justify-center gap-3">
            <button onClick={() => location.reload()} className="rounded-xl bg-ink px-5 py-2.5 text-sm font-medium text-bg">Reload</button>
            <a href="/" className="rounded-xl border border-line px-5 py-2.5 text-sm font-medium">Home</a>
          </div>
        </div>
      </div>
    );
  }
}
