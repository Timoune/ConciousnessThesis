import numpy as np

K = 4
NPC = 15
NR = 64
N_S = K * NPC
N_A = NR
N_R = K * NPC
N = N_S + N_A + N_R

SL = slice(0, N_S)
AL = slice(N_S, N_S + N_A)
RL = slice(N_S + N_A, N)

CLU_S = np.repeat(np.arange(K), NPC)
CLU_R = np.repeat(np.arange(K), NPC)
RING_PHI = 2 * np.pi * np.arange(NR) / NR
CLU_PHI = 2 * np.pi * np.arange(K) / K

DEFAULTS = dict(
    tau_r=0.020,
    tau_a=0.500,
    beta=2.8,
    g_I=6.5,
    w_plus=3.2,
    w_minus=-1.0,
    I_bg=0.30,
    sigma=0.06,
    ring_J0=-3.4,
    ring_J1=5.2,
    ring_bg=0.34,
    ring_beta=2.2,
    w_SA=0.9,
    w_AR=0.9,
    w_RA=0.35,
    w_AS=0.25,
    g_rand=1.8,
)


def _tuning(phi_post, phi_pre, width=0.9):
    d = np.abs(np.angle(np.exp(1j * (phi_post[:, None] - phi_pre[None, :]))))
    return np.exp(-(d ** 2) / (2 * width ** 2))


def build_W(p, seed=0, jitter=0.08):
    rng = np.random.default_rng(seed)
    W = np.zeros((N, N))

    for lo, clu in ((SL, CLU_S), (RL, CLU_R)):
        same = clu[:, None] == clu[None, :]
        blk = same * (p["w_plus"] / NPC) + (~same) * (p["w_minus"] / (NPC * (K - 1)))
        W[lo, lo] = blk

    dphi = RING_PHI[:, None] - RING_PHI[None, :]
    W[AL, AL] = (p["ring_J0"] + p["ring_J1"] * np.cos(dphi)) / NR

    W[AL, SL] = _tuning(RING_PHI, CLU_PHI[CLU_S]) * (p["w_SA"] / N_S)
    W[RL, AL] = _tuning(CLU_PHI[CLU_R], RING_PHI) * (p["w_AR"] / NR)
    W[AL, RL] = _tuning(RING_PHI, CLU_PHI[CLU_R]) * (p["w_RA"] / N_R)
    W[SL, AL] = _tuning(CLU_PHI[CLU_S], RING_PHI) * (p["w_AS"] / NR)

    W *= 1.0 + jitter * rng.standard_normal((N, N))
    W += p["g_rand"] * rng.standard_normal((N, N)) / np.sqrt(N)
    np.fill_diagonal(W, 0.0)
    return W


def phi_act(h, theta=0.35, k=0.08):
    return 1.0 / (1.0 + np.exp(-(h - theta) / k))


def _drive(p, r, a, ext, W):
    mS = r[SL].mean()
    mR = r[RL].mean()
    inh = np.empty(N)
    inh[SL] = p["g_I"] * mS
    inh[AL] = 0.0
    inh[RL] = p["g_I"] * mR
    bg = np.empty(N)
    bg[SL] = p["I_bg"]
    bg[AL] = p["ring_bg"]
    bg[RL] = p["I_bg"]
    return W @ r - a - inh + bg + ext


def bump_angle(rA):
    z = (rA * np.exp(1j * RING_PHI)).sum()
    return np.angle(z), np.abs(z) / max(rA.sum(), 1e-9)


def simulate(params=None, T=20.0, dt=0.001, seed=1, sample_every=10,
             input_fn=None, r0=None, a0=None, record_units=False):
    p = dict(DEFAULTS, **(params or {}))
    rng = np.random.default_rng(seed)
    W = build_W(p, seed)
    steps = int(round(T / dt))
    n_samp = steps // sample_every

    r = 0.05 + 0.02 * rng.random(N) if r0 is None else r0.copy()
    a = np.zeros(N) if a0 is None else a0.copy()

    beta_vec = np.empty(N)
    beta_vec[SL] = p["beta"]
    beta_vec[AL] = p["ring_beta"]
    beta_vec[RL] = p["beta"]

    projS = np.array([(CLU_S == k) / NPC for k in range(K)])
    projR = np.array([(CLU_R == k) / NPC for k in range(K)])

    cs = np.zeros((n_samp, K))
    cr = np.zeros((n_samp, K))
    ring = np.zeros((n_samp, NR))
    theta = np.zeros(n_samp)
    coh = np.zeros(n_samp)
    units = np.zeros((n_samp, N)) if record_units else None
    adapt = np.zeros((n_samp, N)) if record_units else None

    noise_amp = p["sigma"] * 0.01 / np.sqrt(dt)
    ext = np.zeros(N)

    for t in range(steps):
        if input_fn is not None:
            ext = input_fn(t * dt)
        xi = noise_amp * rng.standard_normal(N)
        h1 = _drive(p, r, a, ext + xi, W)
        f1 = (-r + phi_act(h1)) / p["tau_r"]
        g1 = (-a + beta_vec * r) / p["tau_a"]
        rp = np.clip(r + dt * f1, 0.0, 1.0)
        ap = a + dt * g1
        h2 = _drive(p, rp, ap, ext + xi, W)
        f2 = (-rp + phi_act(h2)) / p["tau_r"]
        g2 = (-ap + beta_vec * rp) / p["tau_a"]
        r = np.clip(r + 0.5 * dt * (f1 + f2), 0.0, 1.0)
        a = a + 0.5 * dt * (g1 + g2)

        if t % sample_every == 0:
            i = t // sample_every
            if i < n_samp:
                cs[i] = projS @ r[SL]
                cr[i] = projR @ r[RL]
                ring[i] = r[AL]
                theta[i], coh[i] = bump_angle(r[AL])
                if record_units:
                    units[i] = r
                    adapt[i] = a

    return dict(S=cs, R=cr, ring=ring, theta=theta, coherence=coh,
                units=units, adapt=adapt, params=p, dt_sample=dt * sample_every,
                final=dict(r=r, a=a))