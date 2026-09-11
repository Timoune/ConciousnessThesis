import { useState, useRef, useEffect, useCallback } from "react";

const K = 4;
const NPC = 12;
const NR = 64;
const N_S = K * NPC;
const N_A = NR;
const N_R = K * NPC;
const N = N_S + N_A + N_R;
const OFF_S = 0;
const OFF_A = N_S;
const OFF_R = N_S + N_A;
const DT = 0.001;
const SAMPLE_EVERY = 10;
const BUF = 1200;
const FEAT = K + K + 2;
const NSTATE = 8;

const PAPER = "#E6E9E6";
const INK = "#14181B";
const GRID = "#C6CDCB";
const MUTED = "#5E6A68";
const FLAG = "#B8860B";
const GOOD = "#2E7D5B";

const S_COLORS = ["#0B4F8C", "#1A6FB8", "#3A90D6", "#6FB4E8"];
const R_COLORS = ["#14624A", "#2E7D5B", "#4E9E76", "#7CBD9A"];
const RING_INK = "#B03B2F";

const BASE = {
  tau_r: 0.02, tau_a: 0.5, beta: 2.8, g_I: 6.5, w_plus: 3.2, w_minus: -1.0,
  I_bg: 0.3, sigma: 0.06, ring_J0: -3.4, ring_J1: 5.2, ring_bg: 0.34,
  ring_beta: 2.2, w_SA: 0.9, w_AR: 0.9, w_RA: 0.35, w_AS: 0.25, g_rand: 1.8,
};

const PRESETS = {
  chaotic: { label: "Chaotic itinerancy", p: { ...BASE } },
  cycle: { label: "Limit cycle", p: { ...BASE, g_rand: 0.0 } },
  frozen: { label: "No self-perturbation", p: { ...BASE, beta: 0.0, ring_beta: 0.0 } },
  hyper: { label: "Past the peak", p: { ...BASE, tau_a: 0.03 } },
  silent: { label: "Silent", p: { ...BASE, I_bg: -0.2, ring_bg: -0.1 } },
};

const RING_PHI = Array.from({ length: NR }, (_, i) => (2 * Math.PI * i) / NR);
const CLU_PHI = Array.from({ length: K }, (_, i) => (2 * Math.PI * i) / K);
const CLU_S = Array.from({ length: N_S }, (_, i) => Math.floor(i / NPC));
const CLU_R = Array.from({ length: N_R }, (_, i) => Math.floor(i / NPC));
const CROSS_TUNE = 0.9;

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussFrom(rand) {
  let spare = null;
  return () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u, v, s;
    do { u = rand() * 2 - 1; v = rand() * 2 - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
    const m = Math.sqrt((-2 * Math.log(s)) / s);
    spare = v * m;
    return u * m;
  };
}

function tune(a, b) {
  let d = a - b;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return Math.exp(-(d * d) / (2 * CROSS_TUNE * CROSS_TUNE));
}

function buildW(p, seed) {
  const rand = mulberry32(seed);
  const g = gaussFrom(rand);
  const W = new Float32Array(N * N);
  const put = (i, j, w) => { W[i * N + j] = w; };

  for (let i = 0; i < N_S; i++)
    for (let j = 0; j < N_S; j++)
      put(OFF_S + i, OFF_S + j, CLU_S[i] === CLU_S[j] ? p.w_plus / NPC : p.w_minus / (NPC * (K - 1)));
  for (let i = 0; i < N_R; i++)
    for (let j = 0; j < N_R; j++)
      put(OFF_R + i, OFF_R + j, CLU_R[i] === CLU_R[j] ? p.w_plus / NPC : p.w_minus / (NPC * (K - 1)));
  for (let i = 0; i < NR; i++)
    for (let j = 0; j < NR; j++)
      put(OFF_A + i, OFF_A + j, (p.ring_J0 + p.ring_J1 * Math.cos(RING_PHI[i] - RING_PHI[j])) / NR);
  for (let i = 0; i < NR; i++) {
    for (let j = 0; j < N_S; j++) put(OFF_A + i, OFF_S + j, tune(RING_PHI[i], CLU_PHI[CLU_S[j]]) * (p.w_SA / N_S));
    for (let j = 0; j < N_R; j++) put(OFF_A + i, OFF_R + j, tune(RING_PHI[i], CLU_PHI[CLU_R[j]]) * (p.w_RA / N_R));
  }
  for (let i = 0; i < N_R; i++)
    for (let j = 0; j < NR; j++) put(OFF_R + i, OFF_A + j, tune(CLU_PHI[CLU_R[i]], RING_PHI[j]) * (p.w_AR / NR));
  for (let i = 0; i < N_S; i++)
    for (let j = 0; j < NR; j++) put(OFF_S + i, OFF_A + j, tune(CLU_PHI[CLU_S[i]], RING_PHI[j]) * (p.w_AS / NR));

  const s = 1 / Math.sqrt(N);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      if (i === j) { W[i * N + j] = 0; continue; }
      W[i * N + j] = W[i * N + j] * (1 + 0.08 * g()) + p.g_rand * g() * s;
    }
    W[i * N + i] = 0;
  }
  return W;
}

const act = (h) => 1 / (1 + Math.exp(-(h - 0.35) / 0.08));

class Network {
  constructor(p, seed = 7) {
    this.seed = seed;
    this.rand = mulberry32(seed + 1);
    this.gauss = gaussFrom(this.rand);
    this.W = buildW(p, seed);
    this.r = new Float32Array(N);
    this.a = new Float32Array(N);
    this.h1 = new Float32Array(N);
    this.rp = new Float32Array(N);
    this.ap = new Float32Array(N);
    this.xi = new Float32Array(N);
    this.f1 = new Float32Array(N);
    this.ext = new Float32Array(N);
    for (let i = 0; i < N; i++) this.r[i] = 0.05 + 0.02 * this.rand();
    this.t = 0;
    this.tick = 0;
    this.S = new Float32Array(K);
    this.Rr = new Float32Array(K);
    this.theta = 0;
    this.coh = 0;
    this.feat = new Float32Array(BUF * FEAT);
    this.labels = new Int8Array(BUF).fill(-1);
    this.head = 0;
    this.filled = 0;
    this.stimUntil = -1;
    this.stimCluster = 0;
    this.km = null;
  }

  rewire(p) { this.W = buildW(p, this.seed); }

  stimulate(cluster) {
    this.stimCluster = cluster;
    this.stimUntil = this.t + 0.25;
  }

  drive(out, r, a, p) {
    let mS = 0, mR = 0;
    for (let i = 0; i < N_S; i++) mS += r[OFF_S + i];
    for (let i = 0; i < N_R; i++) mR += r[OFF_R + i];
    mS = (p.g_I * mS) / N_S;
    mR = (p.g_I * mR) / N_R;
    const W = this.W;
    for (let i = 0; i < N; i++) {
      const row = i * N;
      let h = 0;
      for (let j = 0; j < N; j++) h += W[row + j] * r[j];
      const inh = i < OFF_A ? mS : i < OFF_R ? 0 : mR;
      const bg = i >= OFF_A && i < OFF_R ? p.ring_bg : p.I_bg;
      out[i] = h - a[i] - inh + bg + this.xi[i] + this.ext[i];
    }
  }

  step(p) {
    const { r, a, rp, ap, h1, xi, ext, f1 } = this;
    const nAmp = (p.sigma * 0.01) / Math.sqrt(DT);
    for (let i = 0; i < N; i++) xi[i] = nAmp * this.gauss();
    ext.fill(0);
    if (this.t < this.stimUntil)
      for (let i = 0; i < NPC; i++) ext[OFF_S + this.stimCluster * NPC + i] = 0.45;

    const invR = 1 / p.tau_r;
    const invA = 1 / p.tau_a;
    this.drive(h1, r, a, p);
    for (let i = 0; i < N; i++) {
      const bv = i >= OFF_A && i < OFF_R ? p.ring_beta : p.beta;
      const d1 = (-r[i] + act(h1[i])) * invR;
      const g1 = (-a[i] + bv * r[i]) * invA;
      rp[i] = Math.min(1, Math.max(0, r[i] + DT * d1));
      ap[i] = a[i] + DT * g1;
      f1[i] = d1;
    }
    this.drive(h1, rp, ap, p);
    for (let i = 0; i < N; i++) {
      const bv = i >= OFF_A && i < OFF_R ? p.ring_beta : p.beta;
      const f2 = (-rp[i] + act(h1[i])) * invR;
      const g2 = (-ap[i] + bv * rp[i]) * invA;
      const g1 = (-a[i] + bv * r[i]) * invA;
      r[i] = Math.min(1, Math.max(0, r[i] + 0.5 * DT * (f1[i] + f2)));
      a[i] = a[i] + 0.5 * DT * (g1 + g2);
    }
    this.t += DT;
    this.tick++;
    if (this.tick % SAMPLE_EVERY === 0) this.sample();
  }

  sample() {
    const { r } = this;
    this.S.fill(0); this.Rr.fill(0);
    for (let i = 0; i < N_S; i++) this.S[CLU_S[i]] += r[OFF_S + i] / NPC;
    for (let i = 0; i < N_R; i++) this.Rr[CLU_R[i]] += r[OFF_R + i] / NPC;
    let cx = 0, cy = 0, tot = 0;
    for (let i = 0; i < NR; i++) {
      const v = r[OFF_A + i];
      cx += v * Math.cos(RING_PHI[i]);
      cy += v * Math.sin(RING_PHI[i]);
      tot += v;
    }
    this.theta = Math.atan2(cy, cx);
    this.coh = tot > 1e-9 ? Math.hypot(cx, cy) / tot : 0;
    const o = this.head * FEAT;
    for (let k = 0; k < K; k++) { this.feat[o + k] = this.S[k]; this.feat[o + K + k] = this.Rr[k]; }
    this.feat[o + 2 * K] = this.coh * Math.cos(this.theta);
    this.feat[o + 2 * K + 1] = this.coh * Math.sin(this.theta);
    this.labels[this.head] = this.assign(o);
    this.head = (this.head + 1) % BUF;
    if (this.filled < BUF) this.filled++;
  }

  assign(o) {
    if (this.km === null) {
      if (this.filled < 120) return -1;
      this.km = { c: new Float32Array(NSTATE * FEAT), m: new Float32Array(FEAT), s: new Float32Array(FEAT).fill(1) };
      for (let d = 0; d < FEAT; d++) {
        let mu = 0;
        for (let s = 0; s < this.filled; s++) mu += this.feat[s * FEAT + d];
        mu /= this.filled;
        let va = 0;
        for (let s = 0; s < this.filled; s++) { const z = this.feat[s * FEAT + d] - mu; va += z * z; }
        this.km.m[d] = mu;
        this.km.s[d] = Math.sqrt(va / this.filled) || 1;
      }
      for (let q = 0; q < NSTATE; q++) {
        const src = Math.floor((q * this.filled) / NSTATE) * FEAT;
        for (let d = 0; d < FEAT; d++)
          this.km.c[q * FEAT + d] = (this.feat[src + d] - this.km.m[d]) / this.km.s[d];
      }
    }
    const { c, m, s } = this.km;
    let best = 0, bd = Infinity;
    for (let q = 0; q < NSTATE; q++) {
      let d2 = 0;
      for (let d = 0; d < FEAT; d++) {
        const z = (this.feat[o + d] - m[d]) / s[d] - c[q * FEAT + d];
        d2 += z * z;
      }
      if (d2 < bd) { bd = d2; best = q; }
    }
    const lr = 0.02;
    for (let d = 0; d < FEAT; d++) {
      const z = (this.feat[o + d] - m[d]) / s[d];
      c[best * FEAT + d] += lr * (z - c[best * FEAT + d]);
    }
    return best;
  }

  diagnostics() {
    const n = this.filled;
    if (n < 300) return null;
    const idx = (s) => (this.head - n + s + BUF) % BUF;
    const seq = [];
    for (let s = 0; s < n; s++) seq.push(this.labels[idx(s)]);
    let trans = 0;
    const runs = [];
    let run = 1;
    const used = new Set();
    for (let s = 0; s < n; s++) {
      if (seq[s] >= 0) used.add(seq[s]);
      if (s > 0) {
        if (seq[s] !== seq[s - 1]) { trans++; runs.push(run); run = 1; } else run++;
      }
    }
    const dur = n * SAMPLE_EVERY * DT;
    const dwell = runs.length ? (runs.reduce((a, b) => a + b, 0) / runs.length) * SAMPLE_EVERY * DT : dur;

    const mean = new Float64Array(FEAT);
    let rate = 0, coh = 0;
    for (let s = 0; s < n; s++) {
      const o = idx(s) * FEAT;
      for (let d = 0; d < FEAT; d++) mean[d] += this.feat[o + d];
      for (let k = 0; k < 2 * K; k++) rate += this.feat[o + k];
      coh += Math.hypot(this.feat[o + 2 * K], this.feat[o + 2 * K + 1]);
    }
    for (let d = 0; d < FEAT; d++) mean[d] /= n;
    rate /= n * 2 * K;
    coh /= n;

    const C = new Float64Array(FEAT * FEAT);
    for (let s = 0; s < n; s++) {
      const o = idx(s) * FEAT;
      for (let x = 0; x < FEAT; x++) {
        const dx = this.feat[o + x] - mean[x];
        for (let y = x; y < FEAT; y++) C[x * FEAT + y] += dx * (this.feat[o + y] - mean[y]);
      }
    }
    for (let x = 0; x < FEAT; x++)
      for (let y = x; y < FEAT; y++) {
        const v = C[x * FEAT + y] / (n - 1);
        C[x * FEAT + y] = v; C[y * FEAT + x] = v;
      }
    let tr = 0, ss = 0;
    for (let x = 0; x < FEAT; x++) { tr += C[x * FEAT + x]; for (let y = 0; y < FEAT; y++) ss += C[x * FEAT + y] ** 2; }
    const pr = rate < 0.02 || ss <= 0 ? NaN : (tr * tr) / ss;

    const T = new Float64Array(NSTATE * NSTATE);
    for (let s = 1; s < n; s++) if (seq[s] >= 0 && seq[s - 1] >= 0) T[seq[s - 1] * NSTATE + seq[s]]++;
    let tot = 0;
    for (let i = 0; i < NSTATE * NSTATE; i++) tot += T[i];
    let H = 0;
    for (let i = 0; i < NSTATE; i++) {
      let row = 0;
      for (let j = 0; j < NSTATE; j++) row += T[i * NSTATE + j];
      if (row === 0) continue;
      const pi = row / tot;
      for (let j = 0; j < NSTATE; j++) {
        const q = T[i * NSTATE + j] / row;
        if (q > 0) H -= pi * q * Math.log2(q);
      }
    }
    return { transRate: trans / dur, dwell, states: used.size, pr, rate, coh, H, C, mean, window: dur };
  }
}

function verdict(d) {
  if (!d) return { text: "settling", tone: MUTED };
  if (d.rate < 0.02) return { text: "silent â nothing to be conscious of or with", tone: MUTED };
  if (d.transRate < 0.6) return { text: "locked â one state, no endogenous hand-off", tone: FLAG };
  if (d.coh < 0.2) return { text: "bump has dissolved â content code has broken down", tone: FLAG };
  if (d.H < 0.35) return { text: "stereotyped â same itinerary every loop", tone: FLAG };
  if (d.states >= 5) return { text: "chaotic itinerancy â aperiodic, coherent, self-driven", tone: GOOD };
  return { text: "marginal â moving, but the repertoire is thin", tone: FLAG };
}

const SLIDERS = [
  { key: "g_rand", label: "Unstructured weight gain", min: 0, max: 3, step: 0.05, group: "chaos" },
  { key: "beta", label: "Adaptation strength (S, R)", min: 0, max: 5, step: 0.05, group: "chaos" },
  { key: "ring_beta", label: "Adaptation strength (ring)", min: 0, max: 5, step: 0.05, group: "chaos" },
  { key: "tau_a", label: "Adaptation time constant (s)", min: 0.01, max: 1.2, step: 0.005, group: "chaos" },
  { key: "sigma", label: "Noise", min: 0, max: 0.3, step: 0.005, group: "chaos" },
  { key: "ring_J1", label: "Ring local excitation", min: 0, max: 10, step: 0.1, group: "ring" },
  { key: "ring_J0", label: "Ring global inhibition", min: -8, max: 0, step: 0.1, group: "ring" },
  { key: "ring_bg", label: "Ring background drive", min: -0.2, max: 0.8, step: 0.01, group: "ring" },
  { key: "I_bg", label: "Background drive (S, R)", min: -0.4, max: 0.8, step: 0.01, group: "wta" },
  { key: "g_I", label: "Global inhibition (S, R)", min: 0, max: 14, step: 0.1, group: "wta" },
  { key: "w_plus", label: "Within-cluster excitation", min: 0, max: 6, step: 0.05, group: "wta" },
  { key: "w_minus", label: "Cross-cluster suppression", min: -3, max: 0, step: 0.05, group: "wta" },
  { key: "w_SA", label: "Sensory to ring", min: 0, max: 3, step: 0.05, group: "coupling" },
  { key: "w_AR", label: "Ring to readout", min: 0, max: 3, step: 0.05, group: "coupling" },
  { key: "w_RA", label: "Readout to ring", min: 0, max: 3, step: 0.05, group: "coupling" },
  { key: "w_AS", label: "Ring to sensory", min: 0, max: 3, step: 0.05, group: "coupling" },
  { key: "tau_r", label: "Rate time constant (s)", min: 0.008, max: 0.08, step: 0.001, group: "coupling" },
];

const GROUPS = [
  { id: "chaos", title: "Self-perturbation and aperiodicity" },
  { id: "ring", title: "Ring attractor" },
  { id: "wta", title: "Winner-take-all modules" },
  { id: "coupling", title: "Coupling and integration" },
];

const REWIRE_KEYS = new Set(["w_plus", "w_minus", "ring_J0", "ring_J1", "w_SA", "w_AR", "w_RA", "w_AS", "g_rand"]);

export default function RingHybridNetwork() {
  const [params, setParams] = useState(BASE);
  const [running, setRunning] = useState(true);
  const [speed, setSpeed] = useState(8);
  const [diag, setDiag] = useState(null);
  const [now, setNow] = useState({ t: 0, theta: 0, coh: 0 });
  const [openGroup, setOpenGroup] = useState("chaos");

  const netRef = useRef(null);
  const pRef = useRef(params);
  const runRef = useRef(running);
  const spRef = useRef(speed);
  const ringRef = useRef(null);
  const thetaRef = useRef(null);
  const rasterRef = useRef(null);
  const colRef = useRef(0);
  const colT = useRef(0);

  pRef.current = params;
  runRef.current = running;
  spRef.current = speed;
  if (netRef.current === null) netRef.current = new Network(params);

  const reset = useCallback(() => {
    netRef.current = new Network(pRef.current, Math.floor(Math.random() * 99999));
    colRef.current = 0; colT.current = 0;
    setDiag(null);
    for (const ref of [rasterRef, thetaRef]) {
      const c = ref.current;
      if (c) { const x = c.getContext("2d"); x.fillStyle = PAPER; x.fillRect(0, 0, c.width, c.height); }
    }
  }, []);

  const setParam = (key, value) => {
    setParams((prev) => {
      const next = { ...prev, [key]: value };
      if (REWIRE_KEYS.has(key)) netRef.current.rewire(next);
      return next;
    });
  };

  const applyPreset = (k) => { const p = PRESETS[k].p; setParams(p); netRef.current.rewire(p); };

  useEffect(() => {
    let raf, frame = 0;
    const loop = () => {
      const net = netRef.current;
      if (runRef.current) for (let s = 0; s < spRef.current; s++) net.step(pRef.current);
      frame++;
      drawRing(net);
      drawTheta(net);
      drawRaster(net);
      if (frame % 10 === 0) {
        setNow({ t: net.t, theta: net.theta, coh: net.coh });
        const d = net.diagnostics();
        if (d) setDiag(d);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  function drawRing(net) {
    const c = ringRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    const w = c.width, h = c.height;
    ctx.fillStyle = PAPER; ctx.fillRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2;
    const r0 = Math.min(w, h) * 0.22, r1 = Math.min(w, h) * 0.44;
    ctx.strokeStyle = GRID; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, r0, 0, 2 * Math.PI); ctx.stroke();
    for (let i = 0; i < NR; i++) {
      const v = Math.min(1, net.r[OFF_A + i]);
      const ph = RING_PHI[i];
      const rr = r0 + v * (r1 - r0);
      ctx.strokeStyle = RING_INK;
      ctx.globalAlpha = 0.25 + 0.75 * v;
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      ctx.moveTo(cx + r0 * Math.cos(ph), cy - r0 * Math.sin(ph));
      ctx.lineTo(cx + rr * Math.cos(ph), cy - rr * Math.sin(ph));
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    for (let k = 0; k < K; k++) {
      const ph = CLU_PHI[k];
      const v = Math.min(1, net.S[k]);
      ctx.fillStyle = S_COLORS[k];
      ctx.globalAlpha = 0.3 + 0.7 * v;
      ctx.beginPath();
      ctx.arc(cx + (r1 + 14) * Math.cos(ph), cy - (r1 + 14) * Math.sin(ph), 3 + 5 * v, 0, 2 * Math.PI);
      ctx.fill();
      const vr = Math.min(1, net.Rr[k]);
      ctx.fillStyle = R_COLORS[k];
      ctx.globalAlpha = 0.3 + 0.7 * vr;
      ctx.beginPath();
      ctx.arc(cx + (r0 - 13) * Math.cos(ph), cy - (r0 - 13) * Math.sin(ph), 2.5 + 4 * vr, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    if (net.coh > 0.15) {
      ctx.strokeStyle = INK; ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + (r1 + 4) * Math.cos(net.theta), cy - (r1 + 4) * Math.sin(net.theta));
      ctx.stroke();
    }
  }

  function drawTheta(net) {
    const c = thetaRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    const w = c.width, h = c.height;
    const x = colT.current;
    ctx.fillStyle = PAPER; ctx.fillRect(x, 0, 4, h);
    const y = h * (1 - (net.theta + Math.PI) / (2 * Math.PI));
    ctx.globalAlpha = Math.min(1, 0.15 + net.coh);
    ctx.fillStyle = RING_INK;
    ctx.fillRect(x, y - 1.2, 2, 2.4);
    ctx.globalAlpha = 1;
    ctx.fillStyle = GRID; ctx.fillRect(x + 2, 0, 1, h);
    colT.current = (x + 2) % w;
  }

  function drawRaster(net) {
    const c = rasterRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    const w = c.width, h = c.height;
    const rows = 2 * K;
    const rh = h / rows;
    const x = colRef.current;
    ctx.fillStyle = PAPER; ctx.fillRect(x, 0, 4, h);
    for (let k = 0; k < K; k++) {
      const vs = Math.min(1, net.S[k]);
      if (vs > 0.04) {
        ctx.globalAlpha = Math.pow(vs, 0.7); ctx.fillStyle = S_COLORS[k];
        ctx.fillRect(x, k * rh, 2, rh - 0.5);
      }
      const vr = Math.min(1, net.Rr[k]);
      if (vr > 0.04) {
        ctx.globalAlpha = Math.pow(vr, 0.7); ctx.fillStyle = R_COLORS[k];
        ctx.fillRect(x, (K + k) * rh, 2, rh - 0.5);
      }
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = GRID; ctx.fillRect(x + 2, 0, 1, h);
    colRef.current = (x + 2) % w;
  }

  const v = verdict(diag);
  const fmt = (x, n = 2) => (x === undefined || x === null || Number.isNaN(x) ? "â" : x.toFixed(n));

  return (
    <div style={{ background: PAPER, color: INK, minHeight: "100%", padding: "18px 16px 40px", fontFamily: "'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif" }}>
      <style>{`
        .n { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-variant-numeric: tabular-nums; }
        .b { border: 1px solid ${INK}; background: transparent; color: ${INK}; font: inherit;
             padding: 7px 12px; border-radius: 2px; cursor: pointer; }
        .b[data-on="true"] { background: ${INK}; color: ${PAPER}; }
        .b:focus-visible, .c:focus-visible { outline: 2px solid ${FLAG}; outline-offset: 2px; }
        .c { border: 1px solid ${GRID}; background: transparent; color: ${MUTED}; font: inherit;
             font-size: 12px; padding: 6px 9px; border-radius: 2px; cursor: pointer; }
        input[type=range] { width: 100%; accent-color: ${INK}; }
        canvas { width: 100%; display: block; border: 1px solid ${GRID}; background: ${PAPER}; }
      `}</style>

      <div style={{ maxWidth: 620, margin: "0 auto" }}>
        <h1 style={{ fontSize: 21, fontWeight: 600, letterSpacing: "-0.01em", margin: "0 0 4px" }}>
          Step 1b â content without a winner
        </h1>
        <p style={{ margin: "0 0 16px", fontSize: 14, lineHeight: 1.55, color: MUTED, maxWidth: "62ch" }}>
          Two winner-take-all modules with discrete cluster codes, and between them a ring attractor whose
          content is a bump angle with no winner at all. Adaptation drags the bump; unstructured weights make the
          itinerary aperiodic rather than a loop.
        </p>

        <div style={{ border: `1px solid ${INK}`, padding: "12px", marginBottom: 12 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 10, gap: 10 }}>
            <span style={{ fontSize: 13, color: v.tone, fontWeight: 600 }}>{v.text}</span>
            <span className="n" style={{ fontSize: 12, color: MUTED }}>t {now.t.toFixed(1)} s</span>
          </div>
          <canvas ref={ringRef} width={560} height={300} style={{ height: 300 }} />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5, color: MUTED, marginTop: 6 }}>
            <span>outer marks sensory clusters Â· spokes are ring units Â· inner marks readout clusters</span>
            <span className="n">Î¸ {(now.theta).toFixed(2)} rad</span>
          </div>
        </div>

        <canvas ref={thetaRef} width={560} height={92} style={{ height: 92 }} />
        <div style={{ fontSize: 11.5, color: MUTED, margin: "5px 0 10px" }}>
          Bump angle over time, opacity showing how sharp the bump is. Continuous content, no state label anywhere in it.
        </div>

        <canvas ref={rasterRef} width={560} height={120} style={{ height: 120 }} />
        <div style={{ fontSize: 11.5, color: MUTED, margin: "5px 0 16px" }}>
          Sensory clusters above, readout clusters below.
        </div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
          <button className="b" data-on={running} onClick={() => setRunning((x) => !x)}>{running ? "Pause" : "Run"}</button>
          <button className="b" onClick={reset}>Reseed</button>
          <button className="b" onClick={() => netRef.current.stimulate(Math.floor(Math.random() * K))}>Stimulate a sensory cluster</button>
        </div>

        <label style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 12.5, color: MUTED, marginBottom: 16 }}>
          speed
          <input type="range" min={1} max={24} step={1} value={speed} onChange={(e) => setSpeed(+e.target.value)} />
        </label>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(94px, 1fr))", gap: 1, background: GRID, border: `1px solid ${GRID}`, marginBottom: 18 }}>
          {[
            ["transitions / s", diag ? fmt(diag.transRate) : "â"],
            ["mean dwell", diag ? `${(diag.dwell * 1000).toFixed(0)} ms` : "â"],
            ["bump sharpness", diag ? fmt(diag.coh) : "â"],
            ["sequence entropy", diag ? `${fmt(diag.H)} bits` : "â"],
            ["dimensionality", diag ? fmt(diag.pr) : "â"],
            ["states used", diag ? `${diag.states} / ${NSTATE}` : "â"],
          ].map(([k, val]) => (
            <div key={k} style={{ background: PAPER, padding: "9px 10px" }}>
              <div className="n" style={{ fontSize: 17 }}>{val}</div>
              <div style={{ fontSize: 10.5, color: MUTED, marginTop: 2 }}>{k}</div>
            </div>
          ))}
        </div>

        <div style={{ fontSize: 12.5, color: MUTED, marginBottom: 8 }}>Jump to a regime</div>
        <div style={{ display: "flex", gap: 7, flexWrap: "wrap", marginBottom: 20 }}>
          {Object.entries(PRESETS).map(([k, p]) => (
            <button key={k} className="c" onClick={() => applyPreset(k)}>{p.label}</button>
          ))}
        </div>

        {GROUPS.map((g) => (
          <div key={g.id} style={{ borderTop: `1px solid ${GRID}` }}>
            <button className="c" data-on={openGroup === g.id}
              style={{ border: "none", width: "100%", textAlign: "left", padding: "12px 0", fontSize: 13.5, color: openGroup === g.id ? INK : MUTED }}
              onClick={() => setOpenGroup(openGroup === g.id ? null : g.id)}>
              {g.title}
            </button>
            {openGroup === g.id && (
              <div style={{ paddingBottom: 14 }}>
                {SLIDERS.filter((s) => s.group === g.id).map((s) => (
                  <div key={s.key} style={{ marginBottom: 12 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5, marginBottom: 3 }}>
                      <span>{s.label}</span>
                      <span className="n" style={{ color: MUTED }}>{params[s.key].toFixed(s.step < 0.01 ? 3 : 2)}</span>
                    </div>
                    <input type="range" min={s.min} max={s.max} step={s.step} value={params[s.key]}
                      onChange={(e) => setParam(s.key, +e.target.value)} />
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}

        <p style={{ fontSize: 12.5, color: MUTED, lineHeight: 1.6, marginTop: 22, maxWidth: "64ch" }}>
          Set unstructured weight gain to zero and watch the sequence entropy fall: the network keeps moving but
          retraces the same itinerary. Set adaptation to zero and it stops moving at all. The two failures are
          different, and only one of them is the one the paper cares about.
        </p>
      </div>
    </div>
  );
}
