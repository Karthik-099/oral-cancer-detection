import { useMemo, useState } from "react";
import { Canvas, type ThreeEvent } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { Site } from "@/components/MouthMap";

/**
 * Stylised 3D oral cavity. Every anatomical site the API accepts is a separate
 * mesh: hover to highlight, click to select. Drag to rotate, scroll to zoom.
 */

const TISSUE = new THREE.Color("#d9788a");
const TISSUE_DARK = new THREE.Color("#b85a6e");
const GUM = new THREE.Color("#e48f9c");
const TONGUE = new THREE.Color("#d76d82");
const SELECT = new THREE.Color("#22d3ee");
const HOVER = new THREE.Color("#7dd3fc");

interface PartProps {
  site: Site;
  hovered: Site | null;
  selected?: string | null;
  setHovered: (s: Site | null) => void;
  onPick?: (s: Site) => void;
  color: THREE.Color;
  children: React.ReactNode; // geometry
  position?: [number, number, number];
  rotation?: [number, number, number];
  scale?: [number, number, number];
  heat?: number;
}

function Part({ site, hovered, selected, setHovered, onPick, color, children, heat, ...t }: PartProps) {
  const isSel = selected === site;
  const isHov = hovered === site;
  const base = heat !== undefined ? color.clone().lerp(new THREE.Color("#ef4444"), heat) : color;
  return (
    <mesh
      {...t}
      onPointerOver={(e: ThreeEvent<PointerEvent>) => { e.stopPropagation(); setHovered(site); document.body.style.cursor = onPick ? "pointer" : "default"; }}
      onPointerOut={() => { setHovered(null); document.body.style.cursor = "default"; }}
      onClick={(e: ThreeEvent<MouseEvent>) => { e.stopPropagation(); onPick?.(site); }}
    >
      {children}
      <meshPhysicalMaterial
        color={isSel ? SELECT : base}
        emissive={isSel ? SELECT : isHov ? HOVER : new THREE.Color("#000000")}
        emissiveIntensity={isSel ? 0.55 : isHov ? 0.35 : 0}
        roughness={0.42}
        clearcoat={0.6}
        clearcoatRoughness={0.25}
        sheen={0.4}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

/** Points along the dental arch (horseshoe) at a given height. */
function archPoint(theta: number, y: number, rx = 1.22, rz = 1.0): THREE.Vector3 {
  return new THREE.Vector3(rx * Math.sin(theta), y, rz * Math.cos(theta) - 0.05);
}

function Teeth({ y, flip = 1 }: { y: number; flip?: 1 | -1 }) {
  const teeth = useMemo(() => {
    const n = 14;
    return Array.from({ length: n }, (_, i) => {
      const theta = -1.38 + (2.76 * i) / (n - 1);
      const p = archPoint(theta, y);
      const molar = Math.abs(theta) > 0.75;
      return { p, theta, w: molar ? 0.22 : 0.16, h: molar ? 0.2 : 0.28 };
    });
  }, [y]);
  return (
    <group>
      {teeth.map((t, i) => (
        <mesh key={i} position={[t.p.x, t.p.y - flip * 0.08, t.p.z]} rotation={[0, t.theta, 0]} scale={[t.w / 0.16, 1, 1]}>
          <capsuleGeometry args={[0.075, t.h - 0.1, 6, 14]} />
          <meshPhysicalMaterial color="#f6f2ea" roughness={0.18} clearcoat={1} clearcoatRoughness={0.1} sheen={0.3} />
        </mesh>
      ))}
    </group>
  );
}

function Gum({ y, ...p }: Omit<PartProps, "site" | "color" | "children"> & { y: number }) {
  const geom = useMemo(() => {
    const pts = Array.from({ length: 30 }, (_, i) => archPoint(-1.42 + (2.84 * i) / 29, y));
    return new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 80, 0.11, 12, false);
  }, [y]);
  return <Part site="Gingiva" color={GUM} {...p}><primitive object={geom} attach="geometry" /></Part>;
}

function Scene({ value, onChange, counts, onHover }: {
  value?: string | null; onChange?: (s: Site) => void; counts?: Record<string, number>; onHover: (s: Site | null) => void;
}) {
  const [hovered, setHoveredState] = useState<Site | null>(null);
  const setHovered = (s: Site | null) => { setHoveredState(s); onHover(s); };
  const max = counts ? Math.max(1, ...Object.values(counts)) : 1;
  const common = { hovered, selected: value, setHovered, onPick: onChange };
  const heat = (s: Site) => (counts ? (counts[s] ?? 0) / max : undefined);

  return (
    <>
      <ambientLight intensity={0.55} />
      <directionalLight position={[3, 4, 5]} intensity={1.6} />
      <pointLight position={[-3, 1, 3]} intensity={18} color="#22d3ee" />
      <pointLight position={[3, -1, 2]} intensity={14} color="#8b5cf6" />

      {/* throat: gives depth */}
      <mesh position={[0, 0, -1.4]} scale={[1.6, 1.2, 0.4]}>
        <sphereGeometry args={[1, 32, 32]} />
        <meshStandardMaterial color="#2a0d16" roughness={1} />
      </mesh>

      {/* lips */}
      <Part site="Lip" color={TISSUE_DARK} {...common} heat={heat("Lip")} position={[0, 0.22, 0.9]} scale={[1.0, 0.6, 0.5]}>
        <torusGeometry args={[1.48, 0.15, 24, 96, Math.PI]} />
      </Part>
      <Part site="Lip" color={TISSUE_DARK} {...common} heat={heat("Lip")} position={[0, -0.22, 0.9]} rotation={[0, 0, Math.PI]} scale={[1.0, 0.6, 0.5]}>
        <torusGeometry args={[1.48, 0.17, 24, 96, Math.PI]} />
      </Part>

      {/* gums + teeth */}
      <Gum y={0.66} {...common} heat={heat("Gingiva")} />
      <Gum y={-0.66} {...common} heat={heat("Gingiva")} />
      <Teeth y={0.66} flip={1} />
      <Teeth y={-0.66} flip={-1} />

      {/* palate */}
      <Part site="Hard palate" color={TISSUE} {...common} heat={heat("Hard palate")} position={[0, 0.68, 0.15]} scale={[1.12, 0.42, 0.85]}>
        <sphereGeometry args={[1, 48, 24, 0, Math.PI * 2, 0, Math.PI / 2]} />
      </Part>
      <Part site="Soft palate" color={TISSUE_DARK} {...common} heat={heat("Soft palate")} position={[0, 0.55, -0.82]} scale={[0.85, 0.32, 0.38]}>
        <sphereGeometry args={[1, 40, 20, 0, Math.PI * 2, 0, Math.PI / 2]} />
      </Part>

      {/* cheeks */}
      {[-1, 1].map((s) => (
        <Part key={`b${s}`} site="Buccal mucosa" color={TISSUE} {...common} heat={heat("Buccal mucosa")} position={[s * 1.4, 0, 0.05]} scale={[0.12, 0.72, 0.92]}>
          <sphereGeometry args={[1, 48, 48]} />
        </Part>
      ))}
      {[-1, 1].map((s) => (
        <Part key={`r${s}`} site="Retromolar trigone" color={TISSUE_DARK} {...common} heat={heat("Retromolar trigone")} position={[s * 1.12, -0.5, -0.38]} scale={[0.2, 0.12, 0.2]}>
          <sphereGeometry args={[1, 24, 24]} />
        </Part>
      ))}

      {/* tongue and floor of mouth */}
      <Part site="Floor of mouth" color={TISSUE} {...common} heat={heat("Floor of mouth")} position={[0, -0.74, 0.12]} scale={[1.12, 0.06, 0.9]}>
        <sphereGeometry args={[1, 40, 20]} />
      </Part>
      <Part site="Tongue (ventral)" color={TISSUE_DARK} {...common} heat={heat("Tongue (ventral)")} position={[0, -0.6, 0.62]} scale={[0.48, 0.08, 0.28]}>
        <sphereGeometry args={[1, 32, 16]} />
      </Part>
      <Part site="Tongue (dorsal)" color={TONGUE} {...common} heat={heat("Tongue (dorsal)")} position={[0, -0.38, 0.05]} scale={[0.66, 0.26, 1.0]}>
        <sphereGeometry args={[1, 64, 32]} />
      </Part>
      {[-1, 1].map((s) => (
        <Part key={`t${s}`} site="Tongue (lateral)" color={TONGUE} {...common} heat={heat("Tongue (lateral)")} position={[s * 0.58, -0.4, 0.02]} scale={[0.12, 0.2, 0.9]}>
          <sphereGeometry args={[1, 32, 32]} />
        </Part>
      ))}

      <OrbitControls
        enablePan={false}
        minDistance={3.6}
        maxDistance={7.5}
        minPolarAngle={Math.PI * 0.22}
        maxPolarAngle={Math.PI * 0.62}
        minAzimuthAngle={-1.1}
        maxAzimuthAngle={1.1}
        autoRotate={!hovered && !value}
        autoRotateSpeed={0.6}
        enableDamping
      />
    </>
  );
}

export default function Mouth3D({ value, onChange, counts, className }: {
  value?: string | null;
  onChange?: (s: Site) => void;
  counts?: Record<string, number>;
  className?: string;
}) {
  const [label, setLabel] = useState<Site | null>(null);
  return (
    <div className={className} style={{ position: "relative" }}>
      <Canvas dpr={[1, 2]} camera={{ position: [0, 0.9, 5], fov: 40 }} gl={{ antialias: true, alpha: true }}
        aria-label="Interactive 3D model of the oral cavity">
        <Scene value={value} onChange={onChange} counts={counts} onHover={setLabel} />
      </Canvas>
      <div className="pointer-events-none absolute left-3 top-3 rounded-lg border border-line bg-bg/70 px-2.5 py-1 text-xs backdrop-blur">
        {label ?? value ?? (onChange ? "Hover and click a region" : "Drag to rotate")}
        {label && counts ? ` · ${counts[label] ?? 0} case${(counts[label] ?? 0) === 1 ? "" : "s"}` : ""}
      </div>
      <div className="pointer-events-none absolute bottom-3 right-3 text-[11px] text-muted">Drag to rotate · scroll to zoom</div>
    </div>
  );
}
