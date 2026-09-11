import numpy as np

MODULES = ("S", "A", "R")
K = 4
NPC = 15
NMOD = len(MODULES)
NG = NMOD * K
N = NMOD * K * NPC

MOD_OF = np.repeat(np.arange(NMOD), K * NPC)
CLU_OF = np.tile(np.repeat(np.arange(K), NPC), NMOD)
GID = MOD_OF * K + CLU_OF

CROSS = np.array([
    [0.00, 1.00, 0.15],
    [0.45, 0.00, 1.00],
    [0.10, 0.50, 0.00],
])

DEFAULTS = dict(
    tau_r=0.020,
    tau_a=0.500,
    beta=2.8,
    g_I=6.5,
    w_plus=3.2,
    w_minus=-1.0,
    w_cross=0.8,
    I_bg=0.30,
    sigma=0.06,
)

PRESETS = {
    "itinerant": dict(DEFAULTS),
    "frozen": dict(DEFAULTS, beta=0.0),
    "runaway": dict(DEFAULTS, g_I=0.5),
    "silent": dict(DEFAULTS, I_bg=-0.2),
    "hyper": dict(DEFAULTS, tau_a=0.05),
}


def build_W(w_plus, w_minus, w_cross, seed=0, jitter=0.1):
    rng = np.random.default_rng(seed)
    same_mod = MOD_OF[:, None] == MOD_OF[None, :]
    same_clu = GID[:, None] == GID[None, :]
    aligned = CLU_OF[:, None] == CLU_OF[None, :]
    cm = CROSS[MOD_OF[:, None], MOD_OF[None, :]]
    W = np.zeros((N, N))
    W += same_clu * (w_plus / NPC)
    W += (same_mod & ~same_clu) * (w_minus / (NPC * (K - 1)))
    W += (~same_mod) * aligned * cm * (w_cross / NPC)
    W *= 1.0 + jitter * rng.standard_normal((N, N))
    np.fill_diagonal(W, 0.0)
    return W


def phi(h, theta=0.35, k=0.08):
    return 1.0 / (1.0 + np.exp(-(h - theta) / k))


def simulate(params=None, T=20.0, dt=0.001, seed=1, sample_every=10, record_units=False):
    p = dict(DEFAULTS, **(params or {}))
    rng = np.random.default_rng(seed)
    W = build_W(p["w_plus"], p["w_minus"], p["w_cross"], seed)
    steps = int(T / dt)
    n_samp = steps // sample_every

    r = 0.05 + 0.02 * rng.random(N)
    a = np.zeros(N)
    proj = np.zeros((NG, N))
    for g in range(NG):
        proj[g] = (GID == g) / NPC

    clusters = np.zeros((n_samp, NG))
    units = np.zeros((n_samp, N)) if record_units else None
    noise_amp = p["sigma"] * 0.01 / np.sqrt(dt)

    for t in range(steps):
        h = W @ r - a - p["g_I"] * r.mean() + p["I_bg"]
        h += noise_amp * rng.standard_normal(N)
        r += dt / p["tau_r"] * (-r + phi(h))
        np.clip(r, 0.0, 1.0, out=r)
        a += dt / p["tau_a"] * (-a + p["beta"] * r)
        if t % sample_every == 0:
            i = t // sample_every
            if i < n_samp:
                clusters[i] = proj @ r
                if record_units:
                    units[i] = r

    return dict(clusters=clusters, units=units, params=p, dt_sample=dt * sample_every)


def winners(clusters, threshold=0.15):
    top = clusters.max(axis=1)
    return np.where(top > threshold, clusters.argmax(axis=1), -1)


def lz76(seq):
    s = "".join(chr(65 + int(x) + 1) for x in seq)
    n = len(s)
    if n < 3:
        return 0
    i, k, l, c, kmax = 0, 1, 1, 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            kmax = max(kmax, k)
            i += 1
            if i == l:
                c += 1
                l += kmax
                if l + 1 > n:
                    break
                i, k, kmax = 0, 1, 1
            else:
                k = 1
    return c


def diagnostics(result, burn_frac=0.1):
    cl = result["clusters"]
    burn = int(len(cl) * burn_frac)
    cl = cl[burn:]
    ds = result["dt_sample"]
    w = winners(cl)
    changes = w[1:] != w[:-1]
    n_trans = int(changes.sum())
    runs, run = [], 1
    for i in range(1, len(w)):
        if w[i] == w[i - 1]:
            run += 1
        else:
            runs.append(run)
            run = 1
    duration = len(cl) * ds
    C = np.cov(cl.T)
    pr = float(np.trace(C) ** 2 / np.sum(C * C)) if np.sum(C * C) > 0 else 0.0
    compressed = w[np.r_[True, changes]]
    return dict(
        transitions_per_s=n_trans / duration,
        mean_dwell_ms=float(np.mean(runs) * ds * 1000) if runs else duration * 1000,
        states_visited=len(set(w[w >= 0].tolist())),
        participation_ratio=pr,
        mean_rate=float(cl.mean()),
        frac_active=float((w >= 0).mean()),
        sequence_lz=lz76(compressed),
        window_s=duration,
    )


if __name__ == "__main__":
    for name, p in PRESETS.items():
        d = diagnostics(simulate(p, T=20.0))
        line = "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())
        print(f"{name:10s} {line}")