import { useEffect, useRef, useState } from "react";
import { MousePointer2, Pentagon, Square, Trash2, Undo2 } from "lucide-react";
import type { Shape } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Button } from "./ui";

/** Labels a clinician can draw on a photo. Must match ANNOTATION_LABELS in the API. */
export const ANNOTATION_LABELS = [
  "Lesion", "Lip", "Gingiva", "Teeth", "Tongue (tip)", "Tongue (dorsal)", "Tongue (lateral)", "Tongue (ventral)",
  "Buccal mucosa", "Floor of mouth", "Hard palate", "Soft palate", "Retromolar trigone", "Other",
] as const;

const COLORS: Record<string, string> = {
  Lesion: "#ef4444", Lip: "#f472b6", Gingiva: "#fb923c", Teeth: "#e5e7eb", "Tongue (tip)": "#a78bfa",
  "Tongue (dorsal)": "#8b5cf6", "Tongue (lateral)": "#c084fc", "Tongue (ventral)": "#7c3aed", "Buccal mucosa": "#22d3ee",
  "Floor of mouth": "#34d399", "Hard palate": "#facc15", "Soft palate": "#fde047", "Retromolar trigone": "#60a5fa", Other: "#94a3b8",
};
export const labelColor = (l: string) => COLORS[l] ?? "#94a3b8";

type Tool = "select" | "rect" | "polygon";
type Pt = [number, number];

/** Read-only overlay of saved shapes, sized to whatever image it sits on. */
export function AnnotationOverlay({ shapes, className }: { shapes: Shape[]; className?: string }) {
  return (
    <svg viewBox="0 0 1 1" preserveAspectRatio="none" className={cn("pointer-events-none absolute inset-0 h-full w-full", className)}>
      {shapes.map((s, i) => <ShapeView key={i} shape={s} />)}
    </svg>
  );
}

function ShapeView({ shape, selected, onSelect }: { shape: Shape; selected?: boolean; onSelect?: () => void }) {
  const c = labelColor(shape.label);
  const common = {
    fill: c, fillOpacity: selected ? 0.32 : 0.18, stroke: c, strokeWidth: selected ? 3 : 2,
    vectorEffect: "non-scaling-stroke" as const, onPointerDown: onSelect ? (e: React.PointerEvent) => { e.stopPropagation(); onSelect(); } : undefined,
    style: { cursor: onSelect ? "pointer" : undefined },
  };
  if (shape.type === "rect") {
    const [[x1, y1], [x2, y2]] = shape.points;
    return <rect x={Math.min(x1, x2)} y={Math.min(y1, y2)} width={Math.abs(x2 - x1)} height={Math.abs(y2 - y1)} {...common} />;
  }
  return <polygon points={shape.points.map((p) => p.join(",")).join(" ")} {...common} />;
}

export function ImageAnnotator({ src, value, onChange, onDone }: {
  src: string; value: Shape[]; onChange: (s: Shape[]) => void; onDone?: () => void;
}) {
  const [tool, setTool] = useState<Tool>("rect");
  const [label, setLabel] = useState<string>("Lesion");
  const [draft, setDraft] = useState<Pt[] | null>(null);
  const [hoverPt, setHoverPt] = useState<Pt | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const dragStart = useRef<Pt | null>(null);

  const toPt = (e: React.PointerEvent): Pt => {
    const r = svgRef.current!.getBoundingClientRect();
    return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))];
  };
  const add = (shape: Shape) => { onChange([...value, shape]); setSelected(null); };

  const finishPolygon = () => {
    if (draft && draft.length >= 3) add({ type: "polygon", label, points: draft });
    setDraft(null);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { setDraft(null); dragStart.current = null; }
      if (e.key === "Enter") finishPolygon();
      if ((e.key === "Delete" || e.key === "Backspace") && selected !== null && !(e.target as HTMLElement).closest("input,select,textarea")) {
        onChange(value.filter((_, i) => i !== selected)); setSelected(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const onDown = (e: React.PointerEvent) => {
    const p = toPt(e);
    if (tool === "select") { setSelected(null); return; }
    (e.target as Element).setPointerCapture?.(e.pointerId);
    if (tool === "rect") { dragStart.current = p; setDraft([p, p]); return; }
    // polygon: click to add points; click near the first point to close
    if (draft && draft.length >= 3 && Math.hypot(p[0] - draft[0][0], p[1] - draft[0][1]) < 0.025) return finishPolygon();
    setDraft([...(draft ?? []), p]);
  };
  const onMove = (e: React.PointerEvent) => {
    const p = toPt(e);
    setHoverPt(p);
    if (tool === "rect" && dragStart.current) setDraft([dragStart.current, p]);
  };
  const onUp = (e: React.PointerEvent) => {
    if (tool !== "rect" || !dragStart.current) return;
    const p = toPt(e);
    const [a] = [dragStart.current];
    dragStart.current = null;
    setDraft(null);
    if (Math.abs(p[0] - a[0]) > 0.01 && Math.abs(p[1] - a[1]) > 0.01) add({ type: "rect", label, points: [a, p] });
  };

  const relabel = (i: number, l: string) => onChange(value.map((s, j) => (j === i ? { ...s, label: l } : s)));

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_260px]">
      <div>
        <div className="mb-3 flex flex-wrap items-center gap-2">
          {([["select", MousePointer2, "Select"], ["rect", Square, "Box"], ["polygon", Pentagon, "Polygon"]] as const).map(([t, Icon, name]) => (
            <button type="button" key={t} onClick={() => { setTool(t); setDraft(null); }} aria-pressed={tool === t}
              className={cn("inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs transition",
                tool === t ? "border-brand bg-brand/15 text-ink" : "border-line text-muted hover:text-ink")}>
              <Icon className="h-3.5 w-3.5" />{name}
            </button>
          ))}
          <button type="button" onClick={() => { onChange(value.slice(0, -1)); setSelected(null); }} disabled={!value.length}
            className="inline-flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-xs text-muted hover:text-ink disabled:opacity-40">
            <Undo2 className="h-3.5 w-3.5" />Undo
          </button>
          <span className="ml-auto text-[11px] text-muted">
            {tool === "rect" ? "Drag to draw a box" : tool === "polygon" ? "Click points, click the first point or press Enter to close" : "Click a shape to select it"}
          </span>
        </div>
        <div className="relative mx-auto w-fit select-none overflow-hidden rounded-xl bg-surface-2">
          <img src={src} alt="Photo to annotate" className="block max-h-[62vh] w-auto" draggable={false} />
          <svg ref={svgRef} viewBox="0 0 1 1" preserveAspectRatio="none"
            className={cn("absolute inset-0 h-full w-full touch-none", tool !== "select" && "cursor-crosshair")}
            onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerLeave={() => setHoverPt(null)}>
            {value.map((s, i) => <ShapeView key={i} shape={s} selected={selected === i} onSelect={tool === "select" ? () => setSelected(i) : undefined} />)}
            {draft && tool === "rect" && <ShapeView shape={{ type: "rect", label, points: draft }} selected />}
            {draft && tool === "polygon" && (
              <>
                <polyline points={[...draft, ...(hoverPt ? [hoverPt] : [])].map((p) => p.join(",")).join(" ")}
                  fill="none" stroke={labelColor(label)} strokeWidth={2} strokeDasharray="6 4" vectorEffect="non-scaling-stroke" />
                {draft.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={0.008} fill={i === 0 ? "#fff" : labelColor(label)} />)}
              </>
            )}
          </svg>
        </div>
      </div>

      <aside className="space-y-4">
        <div>
          <p className="label">{tool === "select" && selected !== null ? "Change label of selected region" : "Label for new shapes"}</p>
          <div className="flex flex-wrap gap-1.5">
            {ANNOTATION_LABELS.map((l) => (
              <button type="button" key={l} onClick={() => { setLabel(l); if (tool === "select" && selected !== null) relabel(selected, l); }}
                className={cn("inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] transition",
                  label === l ? "border-ink/40 bg-surface-2 text-ink" : "border-line text-muted hover:text-ink")}>
                <span className="h-2 w-2 rounded-full" style={{ background: labelColor(l) }} />{l}
              </button>
            ))}
          </div>
        </div>
        <div>
          <p className="label">Marked regions ({value.length})</p>
          {value.length === 0 ? (
            <p className="text-xs text-muted">Nothing marked yet. Draw a box or polygon on the photo.</p>
          ) : (
            <ul className="max-h-56 space-y-1.5 overflow-auto pr-1">
              {value.map((s, i) => (
                <li key={i} onClick={() => setSelected(i)}
                  className={cn("flex items-center gap-2 rounded-lg border px-2 py-1.5 text-xs", selected === i ? "border-brand/60 bg-brand/5" : "border-line")}>
                  <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: labelColor(s.label) }} />
                  <select value={s.label} onChange={(e) => relabel(i, e.target.value)} className="min-w-0 flex-1 bg-transparent text-xs outline-none" aria-label="Region label">
                    {ANNOTATION_LABELS.map((l) => <option key={l}>{l}</option>)}
                  </select>
                  <span className="text-muted">{s.type === "rect" ? "box" : "poly"}</span>
                  <button type="button" onClick={(e) => { e.stopPropagation(); onChange(value.filter((_, j) => j !== i)); setSelected(null); }}
                    className="text-muted hover:text-high" aria-label="Delete region"><Trash2 className="h-3.5 w-3.5" /></button>
                </li>
              ))}
            </ul>
          )}
        </div>
        {onDone && <Button className="w-full" onClick={onDone}>Done</Button>}
      </aside>
    </div>
  );
}
