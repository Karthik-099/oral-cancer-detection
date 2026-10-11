// Runs before `vite build` (and `npm run dev`). Puts two things in public/:
//   public/ort/    ONNX Runtime Web's WASM files (served from our own domain)
//   public/model/  the OralNet-B4 ONNX model split into 8 MB shards + manifest.json
//
// Model source, first match wins:
//   1. ORALAID_ONNX_URL (+ HF_TOKEN for a private Hugging Face repo)  <- Netlify
//   2. ../models/oralnet-b4.onnx                                      <- local dev
// Weights are never committed to git; both output folders are gitignored.
import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";

const root = new URL("../", import.meta.url);
const pub = (p) => new URL(`public/${p}`, root);
const SHARD = 8 * 1024 * 1024;

// 1. ONNX Runtime WASM
mkdirSync(pub("ort"), { recursive: true });
for (const f of ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"]) {
  copyFileSync(new URL(`node_modules/onnxruntime-web/dist/${f}`, root), pub(`ort/${f}`));
}

// 2. Model
async function loadModelBytes() {
  const url = process.env.ORALAID_ONNX_URL;
  if (url) {
    if (!url.startsWith("https://")) throw new Error("ORALAID_ONNX_URL must be https");
    const headers = process.env.HF_TOKEN && url.startsWith("https://huggingface.co/") ? { Authorization: `Bearer ${process.env.HF_TOKEN}` } : {};
    console.log("[prepare-model] downloading model");
    const res = await fetch(url, { headers, redirect: "follow" });
    if (!res.ok) throw new Error(`model download failed: HTTP ${res.status}`);
    return Buffer.from(await res.arrayBuffer());
  }
  const local = new URL("../models/oralnet-b4.onnx", root);
  if (existsSync(local)) return readFileSync(local);
  return null;
}

const bytes = await loadModelBytes();
if (!bytes) {
  console.warn("[prepare-model] no model found (set ORALAID_ONNX_URL). The site will build, but on-device analysis is disabled.");
  process.exit(0);
}
const sha256 = createHash("sha256").update(bytes).digest("hex");
if (process.env.ORALAID_ONNX_SHA256 && process.env.ORALAID_ONNX_SHA256 !== sha256) {
  throw new Error("model checksum mismatch; refusing to publish");
}
rmSync(pub("model"), { recursive: true, force: true });
mkdirSync(pub("model"), { recursive: true });
const shards = [];
for (let i = 0; i * SHARD < bytes.length; i++) {
  const name = `${sha256.slice(0, 12)}.${String(i).padStart(2, "0")}.bin`;
  writeFileSync(pub(`model/${name}`), bytes.subarray(i * SHARD, (i + 1) * SHARD));
  shards.push(name);
}
writeFileSync(pub("model/manifest.json"), JSON.stringify({ file: "oralnet-b4.onnx", size: bytes.length, sha256, shards }, null, 2));
console.log(`[prepare-model] ${shards.length} shards, ${(bytes.length / 1048576).toFixed(1)} MB, sha256 ${sha256.slice(0, 12)}…`);
