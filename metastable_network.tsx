import { useState, useRef, useEffect, useCallback } from "react";

const MODULES = ["S", "A", "R"];
const MODULE_NAMES = { S: "sensory", A: "association", R: "readout" };
const K = 4;
const NPC = 10;
const NMOD = MODULES.length;
const NG = NMOD * K;
const N = NMOD * K * NPC;
const DT = 0.001;
const SAMPLE_EVERY = 10;
const BUF = 1000;

const CROSS = [
  [0.0, 1.0, 0.15],
  [0.45, 0.0, 1.0],
  [0.1, 0.5, 0.0],
];

const CLUSTER_COLORS = [
  "#0B4F8C", "#1A6FB8", "#3A90D6", "#6FB4E8",
  "#8C1F1C", "#B03B2F", "#CC6248", "#E28C6B",
  "#14624A", "#2E7D5B", "#4E9E76", "#7CBD9A",
];

const PAPER = "#E6E9E6";
const INK = "#14181B";
const GRID = "#C6CDCB";
const MUTED = "#5E6A68";
const FLAG = "#B8860B";

const PRESETS = {
  itinerant: { label: "Itinerant", I_bg: 0.30, g_I: 6.5, w_plus: 3.2, w_minus: -1.0, w_cross: 0.8, beta: 2.8, tau_a: 0.5, tau_r: 0.02, sigma: 0.06 },
  frozen: { label: "No self-perturbation", I_bg: 0.30, g_I: 6.5, w_plus: 3.2, w_minus: -1.0, w_cross: 0.8, beta: 0.0, tau_a: 0.5, tau_r: 0.02, sigma: 0.06 },
  saturated: { label: "Runaway", I_bg: 0.30, g_I: 0.5, w_plus: 3.2, w_minus: -1.0, w_cross: 0.8, beta: 2.8, tau_a: 0.5, tau_r: 0.02, sigma: 0.06 },
  silent: { label: "Silent", I_bg: -0.2, g_I: 6.5, w_plus: 3.2, w_minus: -1.0, w_cross: 0.8, beta: 2.8, tau_a: 0.5, tau_r: 0.02, sigma: 0.06 },
  hyper: { label: "Past the peak", I_bg: 0.30, g_I: 6.5, w_plus: 3.2, w_minus: -1.0, w_cross: 0.8, beta: 2.8, tau_a: 0.05, tau_r: 0.02, sigma: 0.06 },
};

const modOf = (i) => Math.floor(i / (K * NPC));
const cluOf = (i) => Math.floor((i % (K * NPC)) / NPC);
const gidOf = (i) => modOf(i) * K + cluOf(i);

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeGauss(rand) {
  let spare = null;
  return function () {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u = 0, v = 0, s = 0;
    do {
      u = rand() * 2 - 1; v = rand() * 2 - 1; s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const m = Math.sqrt((-2 * Math.log(s)) / s);
    spare = v * m;
    return u * m;
  };
}

function buildW(wPlus, wMinus, wCross, seed) {
  const rand = mulberry32(seed);
  const g = makeGauss(rand);
  const W = new Float32Array(N * N);
  for (let i = 0; i < N; i++) {
    const mi = modOf(i), ci = cluOf(i), gi = gidOf(i);
    for (let j = 0; j < N; j++) {
      if (i === j) continue;
      const mj = modOf(j), cj = cluOf(j), gj = gidOf(j);
      let w = 0;
      if (gi === gj) w = wPlus / NPC;
      else if (mi === mj) w = wMinus / (NPC * (K - 1));
      else if (ci === cj) w = (CROSS[mi][mj] * wCross) / NPC;
      if (w !== 0) W[i * N + j] = w * (1 + 0.1 * g());
    }
  }
  return W;
}

function lz76(seq) {
  if (seq.length < 3) return 0;
  const s = seq.map((x) => String.fromCharCode(65 + x + 1)).join("");
  const n = s.length;
  let i = 0, k = 1, l = 1, c = 1, kmax = 1;
  while (true) {
    if (s[i + k - 1] === s[l + k - 1]) {
      k++;
      if (l + k > n) { c++; break; }
    } else {
      if (k > kmax) kmax = k;
      i++;
      if (i === l) {
        c++; l += kmax;
        if (l + 1 > n) break;
        i = 0; k = 1; kmax = 1;
      } else k = 1;
    }
  }
  return c;
}

class Network {
  constructor(params, seed = 7) {
    this.seed = seed;
    this.rand = mulberry32(seed + 1);
    this.gauss = makeGauss(this.rand);
    this.W = buildW(params.w_plus, params.w_minus, params.w_cross, seed);
    this.r = new Float32Array(N);
    this.a = new Float32Array(N);
    for (let i = 0; i < N; i++) this.r[i] = 0.05 + 0.02 * this.rand();
    this.t = 0;
    this.tick = 0;
    this.g = new Float32Array(NG);
    this.hist = new Float32Array(BUF * NG);
    this.winners = new Int8Array(BUF).fill(-1);
    this.filled = 0;
    this.head = 0;
  }

  rewire(p) {
    this.W = buildW(p.w_plus, p.w_minus, p.w_cross, this.seed);
  }

  step(p) {
    const { r, a, W } = this;
    let rbar = 0;
    for (let i = 0; i < N; i++) rbar += r[i];
    rbar /= N;
    const inh = p.g_I * rbar;
    const nAmp = (p.sigma * 0.01) / Math.sqrt(DT);
    const dr = DT / p.tau_r;
    const da = DT / p.tau_a;
    for (let i = 0; i < N; i++) {
      const row = i * N;
      let h = 0;
      for (let j = 0; j < N; j++) {
        const w = W[row + j];
        if (w !== 0) h += w * r[j];
      }
      h += -a[i] - inh + p.I_bg + nAmp * this.gauss();
      const phi = 1 / (1 + Math.exp(-(h - 0.35) / 0.08));
      let v = r[i] + dr * (-r[i] + phi);
      if (v < 0) v = 0; else if (v > 1) v = 1;
      r[i] = v;
      a[i] = a[i] + da * (-a[i] + p.beta * v);
    }
    this.t += DT;
    this.tick++;
    if (this.tick % SAMPLE_EVERY === 0) this.sample();
  }

  sample() {
    const g = this.g;
    g.fill(0);
    for (let i = 0; i < N; i++) g[gidOf(i)] += this.r[i];
    let best = -1, bv = 0;
    for (let k = 0; k < NG; k++) {
      g[k] /= NPC;
      if (g[k] > bv) { bv = g[k]; best = k; }
    }
    const w = bv > 0.15 ? best : -1;
    const o = this.head * NG;
    for (let k = 0; k < NG; k++) this.hist[o + k] = g[k];
    this.winners[this.head] = w;
    this.head = (this.head + 1) % BUF;
    if (this.filled < BUF) this.filled++;
  }

  ordered() {
    const n = this.filled;
    const out = [];
    for (let s = 0; s < n; s++) {
      const idx = (this.head - n + s + BUF) % BUF;
      out.push(idx);
    }
    return out;
  }

  diagnostics() {
    const idxs = this.ordered();
    const n = idxs.length;
    if (n < 50) return null;
    const seq = idxs.map((i) => this.winners[i]);
    let trans = 0;
    const runs = [];
    let run = 1;
    const seen = new Set();
    for (let s = 0; s < n; s++) {
      if (seq[s] >= 0) seen.add(seq[s]);
      if (s > 0) {
        if (seq[s] !== seq[s - 1]) { trans++; runs.push(run); run = 1; }
        else run++;
      }
    }
    const dur = n * SAMPLE_EVERY * DT;
    const dwell = runs.length ? (runs.reduce((x, y) => x + y, 0) / runs.length) * SAMPLE_EVERY * DT : dur;
    const mean = new Float64Array(NG);
    let rateSum = 0, active = 0;
    for (let s = 0; s < n; s++) {
      const o = idxs[s] * NG;
      for (let k = 0; k < NG; k++) { mean[k] += this.hist[o + k]; rateSum += this.hist[o + k]; }
      if (seq[s] >= 0) active++;
    }
    for (let k = 0; k < NG; k++) mean[k] /= n;
    const C = new Float64Array(NG * NG);
    for (let s = 0; s < n; s++) {
      const o = idxs[s] * NG;
      for (let a2 = 0; a2 < NG; a2++) {
        const da2 = this.hist[o + a2] - mean[a2];
        for (let b = a2; b < NG; b++) {
          C[a2 * NG + b] += da2 * (this.hist[o + b] - mean[b]);
        }
      }
    }
    for (let a2 = 0; a2 < NG; a2++)
      for (let b = a2; b < NG; b++) {
        const v = C[a2 * NG + b] / (n - 1);
        C[a2 * NG + b] = v; C[b * NG + a2] = v;
      }
    let tr = 0, fro = 0;
    for (let a2 = 0; a2 < NG; a2++) {
      tr += C[a2 * NG + a2];
      for (let b = 0; b < NG; b++) fro += C[a2 * NG + b] * C[a2 * NG + b];
    }
    const pr = fro > 1e-12 ? (tr * tr) / fro : 0;
    const compressed = [];
    for (let s = 0; s < n; s++) if (s === 0 || seq[s] !== seq[s - 1]) compressed.push(seq[s]);
    return {
      C,
      mean,
      transRate: trans / dur,
      dwell,
      pr,
      states: seen.size,
      meanRate: rateSum / (n * NG),
      fracActive: active / n,
      lz: lz76(compressed),
      window: dur,
    };
  }
}

function topTwoPCs(C) {
  const norm = (v) => {
    let m = 0;
    for (let i = 0; i < NG; i++) m += v[i] * v[i];
    m = Math.sqrt(m) || 1;
    for (let i = 0; i < NG; i++) v[i] /= m;
    return v;
  };
  const mul = (M, v) => {
    const o = new Float64Array(NG);
    for (let i = 0; i < NG; i++) {
      let s = 0;
      for (let j = 0; j < NG; j++) s += M[i * NG + j] * v[j];
      o[i] = s;
    }
    return o;
  };
  let v1 = new Float64Array(NG).map(() => Math.random() - 0.5);
  norm(v1);
  for (let it = 0; it < 60; it++) v1 = norm(mul(C, v1));
  let lam = 0;
  const Cv = mul(C, v1);
  for (let i = 0; i < NG; i++) lam += v1[i] * Cv[i];
  const D = new Float64Array(NG * NG);
  for (let i = 0; i < NG; i++)
    for (let j = 0; j < NG; j++) D[i * NG + j] = C[i * NG + j] - lam * v1[i] * v1[j];
  let v2 = new Float64Array(NG).map(() => Math.random() - 0.5);
  norm(v2);
  for (let it = 0; it < 60; it++) v2 = norm(mul(D, v2));
  return [v1, v2];
}

function verdict(d) {
  if (!d) return { text: "warming up", tone: MUTED };
  if (d.meanRate < 0.02) return { text: "silent â network is dead", tone: MUTED };
  if (d.transRate < 0.25 && d.fracActive > 0.9)
    return { text: "locked â single stable attractor, no self-perturbation", tone: FLAG };
  if (d.transRate < 0.25) return { text: "stereotyped â not exploring", tone: FLAG };
  if (d.transRate > 6) return { text: "past the peak â transition rate has collapsed the repertoire", tone: FLAG };
  if (d.states >= 5 && d.pr > 2.5)
    return { text: "metastable itinerancy â self-perturbed and exploring", tone: "#2E7D5B" };
  return { text: "marginal â transitions present but repertoire is thin", tone: FLAG };
}

const SLIDERS = [
  { key: "I_bg", label: "Background drive", min: -0.4, max: 0.8, step: 0.01, group: "drive" },
  { key: "g_I", label: "Global inhibition", min: 0, max: 14, step: 0.1, group: "drive" },
  { key: "beta", label: "Adaptation strength", min: 0, max: 5, step: 0.05, group: "perturb" },
  { key: "tau_a", label: "Adaptation time constant (s)", min: 0.02, max: 2, step: 0.01, group: "perturb" },
  { key: "sigma", label: "Noise", min: 0, max: 0.3, step: 0.005, group: "perturb" },
  { key: "w_plus", label: "Within-cluster excitation", min: 0, max: 6, step: 0.05, group: "wiring" },
  { key: "w_minus", label: "Cross-cluster suppression", min: -3, max: 0, step: 0.05, group: "wiring" },
  { key: "w_cross", label: "Between-module coupling", min: 0, max: 3, step: 0.05, group: "wiring" },
  { key: "tau_r", label: "Rate time constant (s)", min: 0.005, max: 0.08, step: 0.001, group: "wiring" },
];

const GROUPS = [
  { id: "drive", title: "Drive and competition" },
  { id: "perturb", title: "Endogenous self-perturbation" },
  { id: "wiring", title: "Wiring" },
];

export default function MetastableNetwork() {
  const [params, setParams] = useState(PRESETS.itinerant);
  const [running, setRunning] = useState(true);
  const [speed, setSpeed] = useState(12);
  const [diag, setDiag] = useState(null);
  const [simTime, setSimTime] = useState(0);
  const [openGroup, setOpenGroup] = useState("perturb");

  const netRef = useRef(null);
  const paramsRef = useRef(params);
  const runRef = useRef(running);
  const speedRef = useRef(speed);
  const pcsRef = useRef(null);
  const trailRef = useRef([]);
  const rasterRef = useRef(null);
  const phaseRef = useRef(null);
  const barsRef = useRef(null);
  const colRef = useRef(0);

  paramsRef.current = params;
  runRef.current = running;
  speedRef.current = speed;

  if (netRef.current === null) netRef.current = new Network(params);

  const reset = useCallback(() => {
    netRef.current = new Network(paramsRef.current, Math.floor(Math.random() * 100000));
    trailRef.current = [];
    pcsRef.current = null;
    colRef.current = 0;
    setDiag(null);
    setSimTime(0);
    const c = rasterRef.current;
    if (c) {
      const ctx = c.getContext("2d");
      ctx.fillStyle = PAPER;
      ctx.fillRect(0, 0, c.width, c.height);
    }
  }, []);

  const applyPreset = (key) => {
    const p = PRESETS[key];
    setParams(p);
    netRef.current.rewire(p);
  };

  const setParam = (key, value) => {
    setParams((prev) => {
      const next = { ...prev, [key]: value };
      if (key === "w_plus" || key === "w_minus" || key === "w_cross") netRef.current.rewire(next);
      return next;
    });
  };

  useEffect(() => {
    let raf;
    let frame = 0;
    const loop = () => {
      const net = netRef.current;
      if (runRef.current) {
        for (let s = 0; s < speedRef.current; s++) net.step(paramsRef.current);
      }
      frame++;
      drawRaster(net);
      if (frame % 4 === 0) {
        const d = net.diagnostics();
        if (d) {
          if (frame % 40 === 0 || pcsRef.current === null) pcsRef.current = topTwoPCs(d.C);
          projectTrail(net, d);
          if (frame % 12 === 0) { setDiag(d); setSimTime(net.t); }
        }
        drawPhase();
        drawBars(net);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  function projectTrail(net, d) {
    const pcs = pcsRef.current;
    if (!pcs) return;
    const g = net.g;
    let x = 0, y = 0;
    for (let k = 0; k < NG; k++) {
      const c = g[k] - d.mean[k];
      x += c * pcs[0][k];
      y += c * pcs[1][k];
    }
    const t = trailRef.current;
    t.push([x, y, net.winners[(net.head - 1 + BUF) % BUF]]);
    if (t.length > 420) t.shift();
  }

  function drawRaster(net) {
    const c = rasterRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    const w = c.width, h = c.height;
    const rowH = h / NG;
    const x = colRef.current;
    ctx.fillStyle = PAPER;
    ctx.fillRect(x, 0, 3, h);
    for (let k = 0; k < NG; k++) {
      const v = Math.min(1, net.g[k] / 0.85);
      if (v > 0.03) {
        ctx.globalAlpha = Math.pow(v, 0.7);
        ctx.fillStyle = CLUSTER_COLORS[k];
        ctx.fillRect(x, k * rowH, 2, rowH - 0.5);
      }
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = GRID;
    ctx.fillRect(x + 2, 0, 1, h);
    colRef.current = (x + 2) % w;
  }

  function drawPhase() {
    const c = phaseRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    const w = c.width, h = c.height;
    ctx.fillStyle = PAPER;
    ctx.fillRect(0, 0, w, h);
    const t = trailRef.current;
    ctx.strokeStyle = GRID;
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      ctx.beginPath();
      ctx.moveTo((w * i) / 4, 0); ctx.lineTo((w * i) / 4, h);
      ctx.moveTo(0, (h * i) / 4); ctx.lineTo(w, (h * i) / 4);
      ctx.stroke();
    }
    if (t.length < 4) return;
    let mx = 0;
    for (const p of t) mx = Math.max(mx, Math.abs(p[0]), Math.abs(p[1]));
    mx = mx || 1;
    const sx = (v) => w / 2 + (v / mx) * (w / 2 - 14);
    const sy = (v) => h / 2 - (v / mx) * (h / 2 - 14);
    ctx.lineWidth = 1.6;
    for (let i = 1; i < t.length; i++) {
      const a = i / t.length;
      ctx.globalAlpha = 0.08 + 0.8 * a * a;
      ctx.strokeStyle = t[i][2] >= 0 ? CLUSTER_COLORS[t[i][2]] : MUTED;
      ctx.beginPath();
      ctx.moveTo(sx(t[i - 1][0]), sy(t[i - 1][1]));
      ctx.lineTo(sx(t[i][0]), sy(t[i][1]));
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    const last = t[t.length - 1];
    ctx.fillStyle = last[2] >= 0 ? CLUSTER_COLORS[last[2]] : MUTED;
    ctx.beginPath();
    ctx.arc(sx(last[0]), sy(last[1]), 4.5, 0, Math.PI * 2);
    ctx.fill();
  }

  function drawBars(net) {
    const c = barsRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    const w = c.width, h = c.height;
    ctx.fillStyle = PAPER;
    ctx.fillRect(0, 0, w, h);
    const gap = 3;
    const bw = (w - gap * (NG - 1)) / NG;
    for (let k = 0; k < NG; k++) {
      const v = Math.min(1, net.g[k]);
      const bh = Math.max(1, v * (h - 2));
      ctx.fillStyle = CLUSTER_COLORS[k];
      ctx.globalAlpha = 0.25;
      ctx.fillRect(k * (bw + gap), 0, bw, h);
      ctx.globalAlpha = 1;
      ctx.fillRect(k * (bw + gap), h - bh, bw, bh);
    }
  }

  const v = verdict(diag);

  return (
    <div style={{ background: PAPER, color: INK, minHeight: "100%", padding: "18px 16px 40px", fontFamily: "'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif" }}>
      <style>{`
        .mn-num { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-variant-numeric: tabular-nums; }
        .mn-btn { border: 1px solid ${INK}; background: transparent; color: ${INK}; font: inherit;
                  padding: 7px 12px; border-radius: 2px; cursor: pointer; }
        .mn-btn:focus-visible { outline: 2px solid ${FLAG}; outline-offset: 2px; }
        .mn-btn[data-on="true"] { background: ${INK}; color: ${PAPER}; }
        .mn-chip { border: 1px solid ${GRID}; background: transparent; color: ${MUTED}; font: inherit;
                   font-size: 12px; padding: 6px 9px; border-radius: 2px; cursor: pointer; }
        .mn-chip[data-on="true"] { border-color: ${INK}; color: ${INK}; }
        input[type=range] { width: 100%; accent-color: ${INK}; }
        canvas { width: 100%; display: block; border: 1px solid ${GRID}; background: ${PAPER}; }
      `}</style>

      <div style={{ maxWidth: 620, margin: "0 auto" }}>
        <h1 style={{ fontSize: 21, fontWeight: 600, letterSpacing: "-0.01em", margin: "0 0 4px" }}>
          Step 1 â a network that will not sit still
        </h1>
        <p style={{ margin: "0 0 18px", fontSize: 14, lineHeight: 1.55, color: MUTED, maxWidth: "62ch" }}>
          {NG} clusters across three modules, {N} rate units, zero external input. Adaptation fatigues whichever
          cluster is winning until it loses; noise picks the next one. Nothing is driving this from outside.
        </p>

        <div style={{ border: `1px solid ${INK}`, padding: "12px 12px 14px", marginBottom: 14 }}>
          <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 10, gap: 10 }}>
            <span style={{ fontSize: 13, color: v.tone, fontWeight: 600 }}>{v.text}</span>
            <span className="mn-num" style={{ fontSize: 12, color: MUTED }}>t = {simTime.toFixed(1)} s</span>
          </div>
          <canvas ref={phaseRef} width={580} height={230} style={{ height: 230 }} />
          <div style={{ fontSize: 11.5, color: MUTED, marginTop: 6 }}>
            Trajectory in the top two principal components of cluster activity. Colour marks the cluster currently winning.
          </div>
        </div>

        <canvas ref={rasterRef} width={580} height={168} style={{ height: 168, marginBottom: 6 }} />
        <canvas ref={barsRef} width={580} height={46} style={{ height: 46 }} />
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5, color: MUTED, margin: "6px 0 16px" }}>
          <span>{MODULES.map((m) => `${m} Â· ${MODULE_NAMES[m]}`).join("   ")}</span>
          <span>live cluster rates</span>
        </div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
          <button className="mn-btn" data-on={running} onClick={() => setRunning((x) => !x)}>
            {running ? "Pause" : "Run"}
          </button>
          <button className="mn-btn" onClick={reset}>Reseed</button>
          <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12.5, color: MUTED, flex: "1 1 150px" }}>
            speed
            <input type="range" min={1} max={40} step={1} value={speed} onChange={(e) => setSpeed(+e.target.value)} />
          </label>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(96px, 1fr))", gap: 1, background: GRID, border: `1px solid ${GRID}`, marginBottom: 18 }}>
          {[
            ["transitions / s", diag ? diag.transRate.toFixed(2) : "â"],
            ["mean dwell", diag ? `${(diag.dwell * 1000).toFixed(0)} ms` : "â"],
            ["states visited", diag ? `${diag.states} / ${NG}` : "â"],
            ["dimensionality", diag ? diag.pr.toFixed(2) : "â"],
            ["sequence LZ", diag ? diag.lz : "â"],
            ["mean rate", diag ? diag.meanRate.toFixed(3) : "â"],
          ].map(([k, val]) => (
            <div key={k} style={{ background: PAPER, padding: "9px 10px" }}>
              <div className="mn-num" style={{ fontSize: 17 }}>{val}</div>
              <div style={{ fontSize: 10.5, color: MUTED, marginTop: 2 }}>{k}</div>
            </div>
          ))}
        </div>

        <div style={{ fontSize: 12.5, color: MUTED, marginBottom: 8 }}>Jump to a regime</div>
        <div style={{ display: "flex", gap: 7, flexWrap: "wrap", marginBottom: 20 }}>
          {Object.entries(PRESETS).map(([k, p]) => (
            <button key={k} className="mn-chip" onClick={() => applyPreset(k)}>{p.label}</button>
          ))}
        </div>

        {GROUPS.map((grp) => (
          <div key={grp.id} style={{ borderTop: `1px solid ${GRID}` }}>
            <button
              className="mn-chip"
              data-on={openGroup === grp.id}
              style={{ border: "none", width: "100%", textAlign: "left", padding: "12px 0", fontSize: 13.5 }}
              onClick={() => setOpenGroup(openGroup === grp.id ? null : grp.id)}
            >
              {grp.title}
            </button>
            {openGroup === grp.id && (
              <div style={{ paddingBottom: 14 }}>
                {SLIDERS.filter((s) => s.group === grp.id).map((s) => (
                  <div key={s.key} style={{ marginBottom: 12 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5, marginBottom: 3 }}>
                      <span>{s.label}</span>
                      <span className="mn-num" style={{ color: MUTED }}>{params[s.key].toFixed(s.step < 0.01 ? 3 : 2)}</span>
                    </div>
                    <input
                      type="range" min={s.min} max={s.max} step={s.step}
                      value={params[s.key]}
                      onChange={(e) => setParam(s.key, +e.target.value)}
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}

        <p style={{ fontSize: 12.5, color: MUTED, lineHeight: 1.6, marginTop: 22, maxWidth: "64ch" }}>
          Drag adaptation strength to zero and the trajectory collapses onto one point within a couple of seconds.
          That is the thesis claim made visible: recurrence alone gets you a stable attractor, not a conscious-type regime.
        </p>
      </div>
    </div>
  );
}
