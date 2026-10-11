// On-device OralNet-B4. The model runs in the browser with ONNX Runtime Web:
// photos never leave the device and nothing is stored on any server.
//
// The ONNX graph (oralaid_app/inference/export_onnx.py) returns logits and a
// Grad-CAM map, so this file only handles pre/post-processing and rendering.
// It mirrors the server pipeline in oralaid_app/inference/engine.py.
import type { ImageResult, Quality, Tier } from "./api";

const INPUT = 380;
const OUT = 512;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
export const THRESHOLD = 0.3;
export const HIGH_THRESHOLD = 0.55;

const TIERS: Record<Tier, { label: string; finding: string; color: string; recommendation: string }> = {
  low: {
    label: "Low risk",
    finding: "No suspicious features detected",
    color: "#10b981",
    recommendation: "Routine recall. Re-screen at the next scheduled visit or sooner if symptoms change.",
  },
  moderate: {
    label: "Moderate risk",
    finding: "Possible suspicious features",
    color: "#f59e0b",
    recommendation: "Clinical re-examination within 2–3 weeks. Consider specialist review if the area persists.",
  },
  high: {
    label: "High risk",
    finding: "Suspicious oral lesion features",
    color: "#ef4444",
    recommendation: "Refer to an oral medicine / maxillofacial specialist for evaluation and possible biopsy.",
  },
};

export class OnDeviceError extends Error {
  constructor(public code: "unsupported" | "not-oral" | "model" | "decode", message: string) {
    super(message);
  }
}

export interface OnDeviceResult extends ImageResult {
  id: string;
  onDevice: true;
}

type Ort = typeof import("onnxruntime-web/wasm");
type Session = import("onnxruntime-web/wasm").InferenceSession;

let sessionPromise: Promise<{ ort: Ort; session: Session }> | null = null;

export interface LoadProgress { loaded: number; total: number; cached: boolean }

/** Download (or read from cache) the model shards and start an inference session. */
export function loadModel(onProgress?: (p: LoadProgress) => void) {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      if (typeof WebAssembly === "undefined") throw new OnDeviceError("unsupported", "This browser can't run the on-device model. Try a recent Chrome, Edge, Firefox or Safari.");
      const ort = await import("onnxruntime-web/wasm");
      ort.env.wasm.wasmPaths = "/ort/";
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
      const bytes = await fetchModel(onProgress);
      const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      return { ort, session };
    })().catch((e) => {
      sessionPromise = null;
      throw e instanceof OnDeviceError ? e : new OnDeviceError("model", "The AI model couldn't be loaded. Check your connection and try again.");
    });
  }
  return sessionPromise;
}

interface Manifest { file: string; size: number; sha256: string; shards: string[] }

async function fetchModel(onProgress?: (p: LoadProgress) => void): Promise<Uint8Array> {
  const res = await fetch("/model/manifest.json", { cache: "no-cache" });
  if (!res.ok || !res.headers.get("content-type")?.includes("json")) {
    throw new OnDeviceError("model", "The on-device model isn't published on this site yet.");
  }
  const manifest = (await res.json()) as Manifest;
  const cacheKey = `/model/${manifest.sha256}.onnx`;

  let cache: Cache | null = null;
  try { cache = await caches.open("oralaid-model-v1"); } catch { /* private mode: no cache */ }
  const hit = await cache?.match(cacheKey);
  if (hit) {
    onProgress?.({ loaded: manifest.size, total: manifest.size, cached: true });
    return new Uint8Array(await hit.arrayBuffer());
  }

  const out = new Uint8Array(manifest.size);
  let offset = 0;
  for (const shard of manifest.shards) {
    const r = await fetch(`/model/${shard}`);
    if (!r.ok || !r.body) throw new OnDeviceError("model", "Model download failed. Please retry.");
    const reader = r.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      out.set(value, offset);
      offset += value.length;
      onProgress?.({ loaded: offset, total: manifest.size, cached: false });
    }
  }
  if (offset !== manifest.size) throw new OnDeviceError("model", "Model download was incomplete. Please retry.");
  const digest = await crypto.subtle.digest("SHA-256", out);
  const hex = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
  if (hex !== manifest.sha256) throw new OnDeviceError("model", "Model integrity check failed. Please reload the page.");
  try { await cache?.put(cacheKey, new Response(out, { headers: { "content-type": "application/octet-stream" } })); } catch { /* quota */ }
  return out;
}

// ── image helpers ──────────────────────────────────────────────────────────

function canvas(w: number, h: number) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}

function draw(bitmap: ImageBitmap, w: number, h: number) {
  const c = canvas(w, h);
  const ctx = c.getContext("2d", { willReadFrequently: true })!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, w, h);
  return { c, ctx, data: ctx.getImageData(0, 0, w, h).data };
}

function quality(bitmap: ImageBitmap): Quality {
  const { data } = draw(bitmap, OUT, OUT);
  const gray = new Float32Array(OUT * OUT);
  let sum = 0;
  for (let i = 0; i < gray.length; i++) {
    gray[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
    sum += gray[i];
  }
  const brightness = sum / gray.length;
  // Laplacian variance (same kernel as cv2.Laplacian ksize=1), interior pixels.
  let n = 0, mean = 0, m2 = 0;
  for (let y = 1; y < OUT - 1; y++) {
    for (let x = 1; x < OUT - 1; x++) {
      const i = y * OUT + x;
      const v = gray[i - 1] + gray[i + 1] + gray[i - OUT] + gray[i + OUT] - 4 * gray[i];
      n++;
      const d = v - mean;
      mean += d / n;
      m2 += d * (v - mean);
    }
  }
  const sharpness = m2 / n;
  const issues: string[] = [];
  if (sharpness < 8) issues.push("Image looks blurry. Hold the camera steady and refocus.");
  if (brightness < 50) issues.push("Image is underexposed. Add light or use the flash.");
  else if (brightness > 215) issues.push("Image is overexposed. Reduce glare or flash intensity.");
  const short = Math.min(bitmap.width, bitmap.height);
  if (short < 224) issues.push("Low resolution. Capture at least 640px on the short side.");
  let score = 100;
  score -= sharpness < 8 ? 35 : sharpness < 12 ? 10 : 0;
  score -= brightness < 50 || brightness > 215 ? 25 : 0;
  score -= short < 224 ? 20 : 0;
  return {
    score: Math.max(score, 0),
    sharpness: Math.round(sharpness * 10) / 10,
    brightness: Math.round(brightness * 10) / 10,
    resolution: `${bitmap.width}×${bitmap.height}`,
    issues,
  };
}

/** Port of models/validator.py: rejects obvious non-oral photos before inference. */
function checkOral(bitmap: ImageBitmap) {
  if (bitmap.width < 50 || bitmap.height < 50) throw new OnDeviceError("not-oral", "Image resolution too low. Please upload a clear intraoral photograph.");
  const { data } = draw(bitmap, 128, 128);
  const px = data.length / 4;
  let pink = 0, white = 0, dark = 0, green = 0, blue = 0;
  const counts = new Map<number, number>();
  const chan = [0, 0, 0], chan2 = [0, 0, 0];
  for (let i = 0; i < px; i++) {
    const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2];
    if (r > 140 && g < 160 && b < 160 && r > g && r > b) pink++;
    if (r > 180 && g > 180 && b > 180) white++;
    if (r < 60 && g < 60 && b < 60) dark++;
    if (g > r * 1.1 && g > b * 1.1 && g > 70) green++;
    if (b > r * 1.1 && b > g * 1.1 && b > 70) blue++;
    const q = ((r >> 5) << 6) | ((g >> 5) << 3) | (b >> 5);
    counts.set(q, (counts.get(q) ?? 0) + 1);
    [r, g, b].forEach((v, k) => { chan[k] += v; chan2[k] += v * v; });
  }
  const std = Math.max(...chan.map((s, k) => Math.sqrt(Math.max(0, chan2[k] / px - (s / px) ** 2))));
  if (std < 5) throw new OnDeviceError("not-oral", "Image appears to be blank or uniform. Please upload a real intraoral photograph.");
  const pinkR = pink / px, whiteR = white / px, darkR = dark / px;
  const scene = Math.max(green, blue) / px;
  const top3 = [...counts.values()].sort((a, b) => b - a).slice(0, 3).reduce((a, b) => a + b, 0) / px;
  if (scene > 0.12 && pinkR < 0.18) throw new OnDeviceError("not-oral", "This looks like a non-oral photo. Please use an intraoral image (mouth, teeth, gums or lesion).");
  if (top3 > 0.75 && whiteR > 0.35 && darkR < 0.15) throw new OnDeviceError("not-oral", "This looks like a logo or graphic. Please use an intraoral photograph.");
  const score = Math.min(1, pinkR * 2 + whiteR * 1.5 + darkR * 0.5);
  if (score < 0.15) throw new OnDeviceError("not-oral", "This image doesn't look like an oral cavity photograph. Please upload a clear intraoral image.");
}

function toTensor(bitmap: ImageBitmap) {
  const { data } = draw(bitmap, INPUT, INPUT);
  const plane = INPUT * INPUT;
  const out = new Float32Array(2 * 3 * plane); // [original, horizontally flipped]
  for (let y = 0; y < INPUT; y++) {
    for (let x = 0; x < INPUT; x++) {
      const src = (y * INPUT + x) * 4;
      const dst = y * INPUT + x;
      const flip = y * INPUT + (INPUT - 1 - x);
      for (let c = 0; c < 3; c++) {
        const v = (data[src + c] / 255 - MEAN[c]) / STD[c];
        out[c * plane + dst] = v;
        out[3 * plane + c * plane + flip] = v;
      }
    }
  }
  return out;
}

// Turbo colormap (Google), polynomial approximation.
function turbo(t: number): [number, number, number] {
  const r = 34.61 + t * (1172.33 - t * (10793.56 - t * (33300.12 - t * (38394.49 - t * 14825.05))));
  const g = 23.31 + t * (557.33 + t * (1225.33 - t * (3574.96 - t * (1073.77 + t * 707.56))));
  const b = 27.2 + t * (3211.1 - t * (15327.97 - t * (27814 - t * (22569.18 - t * 6838.66))));
  const c = (v: number) => Math.max(0, Math.min(255, v));
  return [c(r), c(g), c(b)];
}

/** Upsample the 12×12 CAM to OUT×OUT with smoothing, normalised to 0..1. */
function upsampleCam(cam: Float32Array, size: number) {
  const small = canvas(size, size);
  const sctx = small.getContext("2d")!;
  const img = sctx.createImageData(size, size);
  let max = 0;
  for (const v of cam) max = Math.max(max, v);
  for (let i = 0; i < cam.length; i++) {
    const v = max > 0 ? Math.round((cam[i] / max) * 255) : 0;
    img.data.set([v, v, v, 255], i * 4);
  }
  sctx.putImageData(img, 0, 0);
  const big = canvas(OUT, OUT);
  const bctx = big.getContext("2d", { willReadFrequently: true })!;
  bctx.imageSmoothingEnabled = true;
  bctx.imageSmoothingQuality = "high";
  bctx.filter = `blur(${OUT / 60}px)`;
  bctx.drawImage(small, 0, 0, OUT, OUT);
  const d = bctx.getImageData(0, 0, OUT, OUT).data;
  const out = new Float32Array(OUT * OUT);
  let lo = 255, hi = 0;
  for (let i = 0; i < out.length; i++) { out[i] = d[i * 4]; lo = Math.min(lo, out[i]); hi = Math.max(hi, out[i]); }
  for (let i = 0; i < out.length; i++) out[i] = (out[i] - lo) / (hi - lo + 1e-6);
  return out;
}

function render(base: Uint8ClampedArray, fn: (i: number, px: Uint8ClampedArray) => void) {
  const c = canvas(OUT, OUT);
  const ctx = c.getContext("2d")!;
  const img = ctx.createImageData(OUT, OUT);
  img.data.set(base);
  for (let i = 0; i < OUT * OUT; i++) fn(i, img.data);
  ctx.putImageData(img, 0, 0);
  return c.toDataURL("image/jpeg", 0.9);
}

// ── public API ─────────────────────────────────────────────────────────────

export async function analyzeOnDevice(file: Blob, onProgress?: (p: LoadProgress) => void): Promise<OnDeviceResult> {
  const t0 = performance.now();
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    throw new OnDeviceError("decode", "This image format can't be read in the browser. Use JPEG, PNG or WebP (HEIC: export as JPEG first).");
  }
  checkOral(bitmap);
  const q = quality(bitmap);
  const { ort, session } = await loadModel(onProgress);

  const input = new ort.Tensor("float32", toTensor(bitmap), [2, 3, INPUT, INPUT]);
  const outputs = await session.run({ image: input });
  const logits = outputs.logits.data as Float32Array;
  const camRaw = outputs.cam.data as Float32Array;
  const h = Number(outputs.cam.dims[1]);
  const w = Number(outputs.cam.dims[2]);

  const probs = [0, 1].map((k) => {
    const a = logits[k * 2], b = logits[k * 2 + 1];
    const m = Math.max(a, b);
    return Math.exp(b - m) / (Math.exp(a - m) + Math.exp(b - m));
  });
  const p = (probs[0] + probs[1]) / 2;
  const uncertainty = Math.abs(probs[0] - probs[1]) / 2;

  // Average the CAM of the original and the (un-flipped) flipped view.
  const camAvg = new Float32Array(h * w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      camAvg[y * w + x] = (camRaw[y * w + x] + camRaw[h * w + y * w + (w - 1 - x)]) / 2;
    }
  }
  const cam = upsampleCam(camAvg, w);

  const base = draw(bitmap, OUT, OUT).data;
  const original = render(base, () => {});
  const heatmap = render(base, (i, px) => { const [r, g, b] = turbo(cam[i]); px[i * 4] = r; px[i * 4 + 1] = g; px[i * 4 + 2] = b; });
  const gradcam = render(base, (i, px) => {
    const a = 0.15 + 0.45 * cam[i];
    const [r, g, b] = turbo(cam[i]);
    px[i * 4] = px[i * 4] * (1 - a) + r * a;
    px[i * 4 + 1] = px[i * 4 + 1] * (1 - a) + g * a;
    px[i * 4 + 2] = px[i * 4 + 2] * (1 - a) + b * a;
  });
  // Region of interest: the strongest-attention area with a white outline.
  const inRegion = (i: number) => cam[i] > 0.6;
  let regionCount = 0;
  const segmentation = render(base, (i, px) => {
    if (!inRegion(i)) return;
    regionCount++;
    const x = i % OUT, y = (i / OUT) | 0;
    const edge = x === 0 || y === 0 || x === OUT - 1 || y === OUT - 1 ||
      !inRegion(i - 1) || !inRegion(i + 1) || !inRegion(i - OUT) || !inRegion(i + OUT);
    if (edge) { px[i * 4] = px[i * 4 + 1] = px[i * 4 + 2] = 255; return; }
    px[i * 4] = px[i * 4] * 0.55 + 239 * 0.45;
    px[i * 4 + 1] = px[i * 4 + 1] * 0.55 + 68 * 0.45;
    px[i * 4 + 2] = px[i * 4 + 2] * 0.55 + 68 * 0.45;
  });
  bitmap.close();

  const tier: Tier = p >= HIGH_THRESHOLD ? "high" : p >= THRESHOLD ? "moderate" : "low";
  return {
    id: crypto.randomUUID?.() ?? String(Date.now()),
    onDevice: true,
    index: 0,
    probability: Math.round(p * 10000) / 10000,
    uncertainty: Math.round(uncertainty * 10000) / 10000,
    confidence: Math.round(Math.max(p, 1 - p) * 10000) / 10000,
    tier,
    ...TIERS[tier],
    threshold: THRESHOLD,
    region_area_ratio: regionCount / (OUT * OUT),
    attention_area_ratio: regionCount / (OUT * OUT),
    quality: q,
    files: { original, gradcam, heatmap, segmentation },
    inference_ms: Math.round(performance.now() - t0),
  };
}

/** True when the browser has the model cached (instant analysis). */
export async function isModelCached() {
  try {
    const res = await fetch("/model/manifest.json", { cache: "no-cache" });
    if (!res.ok) return false;
    const m = (await res.json()) as Manifest;
    return !!(await (await caches.open("oralaid-model-v1")).match(`/model/${m.sha256}.onnx`));
  } catch {
    return false;
  }
}
