import { useEffect, useRef, useState } from "react";
import { Crop, RotateCcw } from "lucide-react";
import { Button } from "./ui";

/**
 * Drag a rectangle to crop out fingers, retractors or background before
 * analysis. Background objects can pull the model's attention away from tissue.
 */
export function ImageCropper({ file, onDone, onCancel }: { file: File; onDone: (f: File) => void; onCancel: () => void }) {
  const [src] = useState(() => URL.createObjectURL(file));
  const imgRef = useRef<HTMLImageElement>(null);
  const [rect, setRect] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const start = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => () => URL.revokeObjectURL(src), [src]);

  const rel = (e: React.PointerEvent) => {
    const b = imgRef.current!.getBoundingClientRect();
    return { x: Math.min(1, Math.max(0, (e.clientX - b.left) / b.width)), y: Math.min(1, Math.max(0, (e.clientY - b.top) / b.height)) };
  };

  const apply = async () => {
    const img = imgRef.current!;
    if (!rect || rect.w < 0.05 || rect.h < 0.05) return onDone(file);
    const canvas = document.createElement("canvas");
    const sx = rect.x * img.naturalWidth, sy = rect.y * img.naturalHeight;
    canvas.width = Math.round(rect.w * img.naturalWidth);
    canvas.height = Math.round(rect.h * img.naturalHeight);
    canvas.getContext("2d")!.drawImage(img, sx, sy, canvas.width, canvas.height, 0, 0, canvas.width, canvas.height);
    const blob: Blob = await new Promise((res) => canvas.toBlob((b) => res(b!), "image/jpeg", 0.95));
    onDone(new File([blob], file.name.replace(/\.\w+$/, "") + "-crop.jpg", { type: "image/jpeg" }));
  };

  return (
    <div className="space-y-4">
      <div
        className="relative mx-auto w-fit touch-none select-none overflow-hidden rounded-xl"
        onPointerDown={(e) => { start.current = rel(e); setRect(null); (e.target as Element).setPointerCapture?.(e.pointerId); }}
        onPointerMove={(e) => {
          if (!start.current) return;
          const p = rel(e), s = start.current;
          setRect({ x: Math.min(s.x, p.x), y: Math.min(s.y, p.y), w: Math.abs(p.x - s.x), h: Math.abs(p.y - s.y) });
        }}
        onPointerUp={() => (start.current = null)}
      >
        <img ref={imgRef} src={src} alt="Image to crop" className="block max-h-[60vh] w-auto cursor-crosshair" draggable={false} />
        {rect && (
          <>
            <div className="pointer-events-none absolute inset-0 bg-black/50" style={{
              clipPath: `polygon(0 0,100% 0,100% 100%,0 100%,0 0,${rect.x * 100}% ${rect.y * 100}%,${rect.x * 100}% ${(rect.y + rect.h) * 100}%,${(rect.x + rect.w) * 100}% ${(rect.y + rect.h) * 100}%,${(rect.x + rect.w) * 100}% ${rect.y * 100}%,${rect.x * 100}% ${rect.y * 100}%)`,
            }} />
            <div className="pointer-events-none absolute border-2 border-white shadow-[0_0_0_1px_rgb(0_0_0/.3)]"
              style={{ left: `${rect.x * 100}%`, top: `${rect.y * 100}%`, width: `${rect.w * 100}%`, height: `${rect.h * 100}%` }} />
          </>
        )}
      </div>
      <p className="text-center text-xs text-muted">Drag across the tissue you want analysed. Leave out fingers, retractors and background.</p>
      <div className="flex justify-end gap-2">
        <Button variant="ghost" onClick={onCancel}>Cancel</Button>
        <Button variant="secondary" onClick={() => setRect(null)}><RotateCcw className="h-4 w-4" />Reset</Button>
        <Button onClick={apply}><Crop className="h-4 w-4" />{rect ? "Apply crop" : "Use full image"}</Button>
      </div>
    </div>
  );
}
