import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, CameraOff, RefreshCw } from "lucide-react";
import { Button, Spinner } from "@/components/ui";

/** Live camera preview (rear camera by default) that captures a JPEG still. */
export function CameraCapture({ onCapture, onClose }: { onCapture: (f: File) => void; onClose: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [facing, setFacing] = useState<"environment" | "user">("environment");
  const [state, setState] = useState<"starting" | "live" | "error">("starting");
  const [error, setError] = useState("");
  const [flash, setFlash] = useState(false);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, []);

  useEffect(() => {
    let cancelled = false;
    setState("starting");
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("This browser can't access the camera. Use the file picker instead.");
      setState("error");
      return;
    }
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: { ideal: facing }, width: { ideal: 1920 }, height: { ideal: 1440 } }, audio: false })
      .then((stream) => {
        if (cancelled) return stream.getTracks().forEach((t) => t.stop());
        stop();
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(() => undefined);
        }
        setState("live");
      })
      .catch((e: DOMException) => {
        if (cancelled) return;
        setError(
          e.name === "NotAllowedError" ? "Camera access was blocked. Allow camera permission in your browser settings, then try again."
            : e.name === "NotFoundError" ? "No camera was found on this device."
              : "The camera couldn't be started. It may be in use by another app.",
        );
        setState("error");
      });
    return () => { cancelled = true; };
  }, [facing, stop]);

  useEffect(() => stop, [stop]);

  const capture = () => {
    const v = videoRef.current;
    if (!v || !v.videoWidth) return;
    const canvas = document.createElement("canvas");
    canvas.width = v.videoWidth;
    canvas.height = v.videoHeight;
    canvas.getContext("2d")!.drawImage(v, 0, 0);
    setFlash(true);
    setTimeout(() => setFlash(false), 180);
    canvas.toBlob((b) => {
      if (!b) return;
      const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
      onCapture(new File([b], `capture-${stamp}.jpg`, { type: "image/jpeg" }));
    }, "image/jpeg", 0.92);
  };

  return (
    <div className="space-y-4">
      <div className="relative aspect-[4/3] overflow-hidden rounded-2xl bg-black">
        <video ref={videoRef} playsInline muted className="h-full w-full object-cover" style={facing === "user" ? { transform: "scaleX(-1)" } : undefined} />
        {state === "starting" && <div className="absolute inset-0 grid place-items-center"><Spinner className="text-white/70" /></div>}
        {state === "error" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6 text-center text-sm text-white/80">
            <CameraOff className="h-6 w-6" />{error}
          </div>
        )}
        {state === "live" && (
          <div className="pointer-events-none absolute inset-6 rounded-[28px] border-2 border-dashed border-white/40">
            <span className="absolute -top-px left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-black/60 px-2.5 py-0.5 text-[11px] text-white">
              Fill the frame with the lesion area
            </span>
          </div>
        )}
        {flash && <div className="absolute inset-0 bg-white/70" />}
      </div>
      <div className="flex items-center justify-between gap-2">
        <Button variant="ghost" onClick={() => setFacing((f) => (f === "environment" ? "user" : "environment"))} disabled={state === "error"}>
          <RefreshCw className="h-4 w-4" />Flip
        </Button>
        <button type="button" onClick={capture} disabled={state !== "live"} aria-label="Capture photo"
          className="grid h-16 w-16 place-items-center rounded-full border-4 border-line bg-surface transition active:scale-95 disabled:opacity-40">
          <span className="grid h-11 w-11 place-items-center rounded-full bg-brand text-white dark:text-bg"><Camera className="h-5 w-5" /></span>
        </button>
        <Button variant="secondary" onClick={onClose}>Done</Button>
      </div>
      <p className="text-center text-xs text-muted">Each capture is added to the case. Up to 6 images.</p>
    </div>
  );
}
