import { useEffect, useRef, useState } from "react";
import {
  AmbientLight, CapsuleGeometry, CatmullRomCurve3, Color, DirectionalLight, DoubleSide, Group, Mesh, MeshPhysicalMaterial,
  MeshStandardMaterial, PerspectiveCamera, PointLight, Raycaster, Scene, SphereGeometry, TorusGeometry, TubeGeometry,
  Vector2, Vector3, WebGLRenderer, type BufferGeometry,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Site } from "@/components/MouthMap";

/**
 * Stylised 3D oral cavity built with plain three.js (named imports keep the
 * download small). Every anatomical site is a separate mesh: hover to
 * highlight, click to select, drag to rotate, scroll to zoom. Rendering pauses
 * while the canvas is off-screen.
 */

const TISSUE = new Color("#d9788a");
const TISSUE_DARK = new Color("#b85a6e");
const GUM = new Color("#e48f9c");
const TONGUE = new Color("#d76d82");
const SELECT = new Color("#22d3ee");
const HOVER = new Color("#7dd3fc");
const BLACK = new Color("#000000");
const HEAT = new Color("#ef4444");

type Vec3 = [number, number, number];
interface PartSpec { site: Site; color: Color; geo: () => BufferGeometry; pos: Vec3; rot?: Vec3; scale?: Vec3 }

const archPoint = (theta: number, y: number) => new Vector3(1.22 * Math.sin(theta), y, Math.cos(theta) - 0.05);
const gumGeo = (y: number) => () =>
  new TubeGeometry(new CatmullRomCurve3(Array.from({ length: 30 }, (_, i) => archPoint(-1.42 + (2.84 * i) / 29, y))), 80, 0.11, 12, false);
const sphere = (w = 32, h = 32) => () => new SphereGeometry(1, w, h);
const dome = (w: number, h: number) => () => new SphereGeometry(1, w, h, 0, Math.PI * 2, 0, Math.PI / 2);

const PARTS: PartSpec[] = [
  { site: "Lip", color: TISSUE_DARK, geo: () => new TorusGeometry(1.48, 0.15, 24, 96, Math.PI), pos: [0, 0.22, 0.9], scale: [1, 0.6, 0.5] },
  { site: "Lip", color: TISSUE_DARK, geo: () => new TorusGeometry(1.48, 0.17, 24, 96, Math.PI), pos: [0, -0.22, 0.9], rot: [0, 0, Math.PI], scale: [1, 0.6, 0.5] },
  { site: "Gingiva", color: GUM, geo: gumGeo(0.66), pos: [0, 0, 0] },
  { site: "Gingiva", color: GUM, geo: gumGeo(-0.66), pos: [0, 0, 0] },
  { site: "Hard palate", color: TISSUE, geo: dome(48, 24), pos: [0, 0.68, 0.15], scale: [1.12, 0.42, 0.85] },
  { site: "Soft palate", color: TISSUE_DARK, geo: dome(40, 20), pos: [0, 0.55, -0.82], scale: [0.85, 0.32, 0.38] },
  ...([-1, 1] as const).flatMap((s): PartSpec[] => [
    { site: "Buccal mucosa", color: TISSUE, geo: sphere(), pos: [s * 1.4, 0, 0.05], scale: [0.12, 0.72, 0.92] },
    { site: "Retromolar trigone", color: TISSUE_DARK, geo: sphere(24, 24), pos: [s * 1.12, -0.5, -0.38], scale: [0.2, 0.12, 0.2] },
    { site: "Tongue (lateral)", color: TONGUE, geo: sphere(), pos: [s * 0.58, -0.4, 0.02], scale: [0.12, 0.2, 0.9] },
  ]),
  { site: "Floor of mouth", color: TISSUE, geo: sphere(40, 20), pos: [0, -0.74, 0.12], scale: [1.12, 0.06, 0.9] },
  { site: "Tongue (ventral)", color: TISSUE_DARK, geo: sphere(32, 16), pos: [0, -0.6, 0.62], scale: [0.48, 0.08, 0.28] },
  { site: "Tongue (dorsal)", color: TONGUE, geo: sphere(48, 24), pos: [0, -0.38, 0.05], scale: [0.66, 0.26, 1] },
];

function buildTeeth(y: number, flip: 1 | -1) {
  const group = new Group();
  const mat = new MeshPhysicalMaterial({ color: "#f6f2ea", roughness: 0.18, clearcoat: 1, clearcoatRoughness: 0.1, sheen: 0.3 });
  for (let i = 0; i < 14; i++) {
    const theta = -1.38 + (2.76 * i) / 13;
    const molar = Math.abs(theta) > 0.75;
    const p = archPoint(theta, y);
    const m = new Mesh(new CapsuleGeometry(0.075, (molar ? 0.2 : 0.28) - 0.1, 6, 14), mat);
    m.position.set(p.x, p.y - flip * 0.08, p.z);
    m.rotation.y = theta;
    m.scale.x = (molar ? 0.22 : 0.16) / 0.16;
    group.add(m);
  }
  return group;
}

export default function Mouth3D({ value, onChange, counts, className }: {
  value?: string | null;
  onChange?: (s: Site) => void;
  counts?: Record<string, number>;
  className?: string;
}) {
  const host = useRef<HTMLDivElement>(null);
  const [label, setLabel] = useState<Site | null>(null);
  const [ready, setReady] = useState(false);
  // Live values for the render loop without re-creating the scene.
  const state = useRef({ value, onChange, counts, hovered: null as Site | null });
  state.current.value = value;
  state.current.onChange = onChange;
  state.current.counts = counts;

  useEffect(() => {
    const el = host.current!;
    let renderer: WebGLRenderer;
    try {
      renderer = new WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    } catch {
      return; // no WebGL: the poster image stays visible
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    renderer.domElement.style.cssText = "position:absolute;inset:0;width:100%;height:100%;opacity:0;transition:opacity .7s;touch-action:none";
    renderer.domElement.setAttribute("aria-label", "Interactive 3D model of the oral cavity");
    el.appendChild(renderer.domElement);

    const scene = new Scene();
    const camera = new PerspectiveCamera(40, 1, 0.1, 50);
    camera.position.set(0, 0.9, 5);
    scene.add(new AmbientLight(0xffffff, 0.55));
    const key = new DirectionalLight(0xffffff, 1.6); key.position.set(3, 4, 5); scene.add(key);
    const cyan = new PointLight("#22d3ee", 18); cyan.position.set(-3, 1, 3); scene.add(cyan);
    const violet = new PointLight("#8b5cf6", 14); violet.position.set(3, -1, 2); scene.add(violet);

    const throat = new Mesh(new SphereGeometry(1, 32, 32), new MeshStandardMaterial({ color: "#2a0d16", roughness: 1 }));
    throat.position.set(0, 0, -1.4); throat.scale.set(1.6, 1.2, 0.4); scene.add(throat);
    scene.add(buildTeeth(0.66, 1), buildTeeth(-0.66, -1));

    const meshes: { mesh: Mesh; site: Site; base: Color; mat: MeshPhysicalMaterial }[] = [];
    for (const p of PARTS) {
      const mat = new MeshPhysicalMaterial({ color: p.color, roughness: 0.42, clearcoat: 0.6, clearcoatRoughness: 0.25, sheen: 0.4, side: DoubleSide });
      const mesh = new Mesh(p.geo(), mat);
      mesh.position.set(...p.pos);
      if (p.rot) mesh.rotation.set(...p.rot);
      if (p.scale) mesh.scale.set(...p.scale);
      mesh.userData.site = p.site;
      scene.add(mesh);
      meshes.push({ mesh, site: p.site, base: p.color, mat });
    }

    const controls = new OrbitControls(camera, renderer.domElement);
    Object.assign(controls, {
      enablePan: false, enableDamping: true, minDistance: 3.6, maxDistance: 7.5,
      minPolarAngle: Math.PI * 0.22, maxPolarAngle: Math.PI * 0.62, minAzimuthAngle: -1.1, maxAzimuthAngle: 1.1, autoRotateSpeed: 0.6,
    });

    const ray = new Raycaster();
    const ptr = new Vector2();
    const pick = (e: PointerEvent): Site | null => {
      const r = renderer.domElement.getBoundingClientRect();
      ptr.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      ray.setFromCamera(ptr, camera);
      const hit = ray.intersectObjects(meshes.map((m) => m.mesh), false)[0];
      return (hit?.object.userData.site as Site) ?? null;
    };
    const onMove = (e: PointerEvent) => {
      const s = pick(e);
      if (s !== state.current.hovered) {
        state.current.hovered = s;
        setLabel(s);
        renderer.domElement.style.cursor = s && state.current.onChange ? "pointer" : "grab";
      }
    };
    let down = { x: 0, y: 0 };
    const onDown = (e: PointerEvent) => { down = { x: e.clientX, y: e.clientY }; };
    const onUp = (e: PointerEvent) => {
      if (Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) return; // it was a drag
      const s = pick(e);
      if (s) state.current.onChange?.(s);
    };
    const onLeave = () => { state.current.hovered = null; setLabel(null); };
    const dom = renderer.domElement;
    dom.addEventListener("pointermove", onMove);
    dom.addEventListener("pointerdown", onDown);
    dom.addEventListener("pointerup", onUp);
    dom.addEventListener("pointerleave", onLeave);

    const resize = () => {
      const { clientWidth: w, clientHeight: h } = el;
      renderer.setSize(w, h, false);
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();

    let visible = true;
    const io = new IntersectionObserver(([en]) => { visible = en.isIntersecting; });
    io.observe(el);

    let frame = 0;
    let first = true;
    const tick = () => {
      frame = requestAnimationFrame(tick);
      if (!visible) return;
      const { value: sel, counts: c, hovered } = state.current;
      const max = c ? Math.max(1, ...Object.values(c)) : 1;
      for (const m of meshes) {
        const isSel = sel === m.site;
        const isHov = hovered === m.site;
        m.mat.color.copy(isSel ? SELECT : c ? m.base.clone().lerp(HEAT, (c[m.site] ?? 0) / max) : m.base);
        m.mat.emissive.copy(isSel ? SELECT : isHov ? HOVER : BLACK);
        m.mat.emissiveIntensity = isSel ? 0.55 : isHov ? 0.35 : 0;
      }
      controls.autoRotate = !hovered && !sel;
      controls.update();
      renderer.render(scene, camera);
      if (first) { first = false; dom.style.opacity = "1"; setReady(true); }
    };
    tick();

    return () => {
      cancelAnimationFrame(frame);
      ro.disconnect();
      io.disconnect();
      controls.dispose();
      dom.removeEventListener("pointermove", onMove);
      dom.removeEventListener("pointerdown", onDown);
      dom.removeEventListener("pointerup", onUp);
      dom.removeEventListener("pointerleave", onLeave);
      scene.traverse((o) => {
        if (o instanceof Mesh) {
          o.geometry.dispose();
          (Array.isArray(o.material) ? o.material : [o.material]).forEach((mt) => mt.dispose());
        }
      });
      renderer.dispose();
      dom.remove();
    };
  }, []);

  return (
    <div ref={host} className={className} style={{ position: "relative" }}>
      <img src="/mouth-3d-poster.webp" alt="" aria-hidden decoding="async"
        className="pointer-events-none absolute inset-0 h-full w-full object-contain transition-opacity duration-700"
        style={{ opacity: ready ? 0 : 1 }} />
      <div className="pointer-events-none absolute left-3 top-3 z-10 rounded-lg border border-line bg-bg/70 px-2.5 py-1 text-xs backdrop-blur">
        {label ?? value ?? (onChange ? "Hover and click a region" : "Drag to rotate")}
        {label && counts ? ` · ${counts[label] ?? 0} case${(counts[label] ?? 0) === 1 ? "" : "s"}` : ""}
      </div>
      <div className="pointer-events-none absolute bottom-3 right-3 z-10 text-[11px] text-muted">Drag to rotate · scroll to zoom</div>
    </div>
  );
}
