import numpy as np

try:
    from sklearn.cluster import KMeans
    _HAS_SK = True
except Exception:
    _HAS_SK = False


def features(res):
    th = res["theta"]
    co = res["coherence"]
    return np.column_stack([res["S"], res["R"], co * np.cos(th), co * np.sin(th)])


def _kmeans_init(X, k, seed):
    if _HAS_SK:
        km = KMeans(n_clusters=k, n_init=6, random_state=seed).fit(X)
        return km.cluster_centers_, km.labels_
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), k, replace=False)
    mu = X[idx]
    for _ in range(25):
        lab = np.argmin(((X[:, None, :] - mu[None]) ** 2).sum(-1), axis=1)
        for j in range(k):
            if (lab == j).any():
                mu[j] = X[lab == j].mean(0)
    return mu, lab


class GaussianHMM:
    def __init__(self, k, seed=0, iters=60, min_var=1e-5):
        self.k = k
        self.seed = seed
        self.iters = iters
        self.min_var = min_var

    def _emission_logp(self, X):
        d = X.shape[1]
        lp = np.zeros((len(X), self.k))
        for j in range(self.k):
            v = self.var[j]
            z = X - self.mu[j]
            lp[:, j] = -0.5 * (np.sum(z * z / v, axis=1) + np.sum(np.log(v)) + d * np.log(2 * np.pi))
        return lp

    def fit(self, X):
        n, d = X.shape
        mu, lab = _kmeans_init(X, self.k, self.seed)
        self.mu = mu
        self.var = np.stack([
            np.maximum(X[lab == j].var(0), self.min_var) if (lab == j).sum() > 1
            else np.maximum(X.var(0), self.min_var)
            for j in range(self.k)
        ])
        self.pi = np.full(self.k, 1.0 / self.k)
        self.A = np.full((self.k, self.k), 0.02 / max(self.k - 1, 1))
        np.fill_diagonal(self.A, 0.98)

        for _ in range(self.iters):
            lp = self._emission_logp(X)
            m = lp.max(1, keepdims=True)
            B = np.exp(lp - m)
            alpha = np.zeros((n, self.k))
            c = np.zeros(n)
            alpha[0] = self.pi * B[0]
            c[0] = alpha[0].sum() + 1e-300
            alpha[0] /= c[0]
            for t in range(1, n):
                alpha[t] = (alpha[t - 1] @ self.A) * B[t]
                c[t] = alpha[t].sum() + 1e-300
                alpha[t] /= c[t]
            beta = np.zeros((n, self.k))
            beta[-1] = 1.0
            for t in range(n - 2, -1, -1):
                beta[t] = (self.A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
            gamma = alpha * beta
            gamma /= gamma.sum(1, keepdims=True) + 1e-300
            xi = np.zeros((self.k, self.k))
            for t in range(n - 1):
                x = (alpha[t][:, None] * self.A) * (B[t + 1] * beta[t + 1])[None, :] / c[t + 1]
                xi += x
            self.pi = gamma[0] / gamma[0].sum()
            self.A = xi / (xi.sum(1, keepdims=True) + 1e-300)
            w = gamma.sum(0) + 1e-300
            self.mu = (gamma.T @ X) / w[:, None]
            for j in range(self.k):
                z = X - self.mu[j]
                self.var[j] = np.maximum((gamma[:, j][:, None] * z * z).sum(0) / w[j], self.min_var)
            self.loglik = float(np.sum(np.log(c) + m.ravel()))
        self.gamma = gamma
        return self

    def n_params(self, d):
        return self.k - 1 + self.k * (self.k - 1) + 2 * self.k * d

    def bic(self, X):
        n, d = X.shape
        return -2 * self.loglik + self.n_params(d) * np.log(n)

    def score(self, X):
        lp = self._emission_logp(X)
        m = lp.max(1, keepdims=True)
        B = np.exp(lp - m)
        alpha = self.pi * B[0]
        c0 = alpha.sum() + 1e-300
        alpha /= c0
        ll = np.log(c0) + m[0, 0]
        for t in range(1, len(X)):
            alpha = (alpha @ self.A) * B[t]
            c = alpha.sum() + 1e-300
            alpha /= c
            ll += np.log(c) + m[t, 0]
        return float(ll / len(X))

    def viterbi(self, X):
        lp = self._emission_logp(X)
        n = len(X)
        logA = np.log(self.A + 1e-300)
        delta = np.log(self.pi + 1e-300) + lp[0]
        psi = np.zeros((n, self.k), dtype=int)
        for t in range(1, n):
            m = delta[:, None] + logA
            psi[t] = m.argmax(0)
            delta = m.max(0) + lp[t]
        path = np.zeros(n, dtype=int)
        path[-1] = delta.argmax()
        for t in range(n - 2, -1, -1):
            path[t] = psi[t + 1][path[t + 1]]
        return path


def label_states(X, k_range=range(2, 11), seed=0, standardize=True):
    Z = X.copy()
    if standardize:
        s = Z.std(0)
        s[s < 1e-8] = 1.0
        Z = (Z - Z.mean(0)) / s
    cut = int(0.7 * len(Z))
    best, best_score = None, -np.inf
    curve = {}
    for k in k_range:
        try:
            m = GaussianHMM(k, seed=seed).fit(Z[:cut])
            s = m.score(Z[cut:])
        except Exception:
            continue
        curve[k] = float(s)
        if s > best_score:
            best, best_score = m, s
    ks = sorted(curve)
    vals = np.array([curve[k] for k in ks])
    gains = np.diff(vals)
    chosen = ks[-1]
    if len(gains) and gains.max() > 0:
        thresh = 0.2 * gains.max()
        for i, g in enumerate(gains):
            if g < thresh:
                chosen = ks[i]
                break
    best = GaussianHMM(chosen, seed=seed).fit(Z)
    return best, best.viterbi(Z), curve


def dwell_stats(path, dt_sample):
    ch = path[1:] != path[:-1]
    dur = len(path) * dt_sample
    runs, run = [], 1
    for i in range(1, len(path)):
        if path[i] == path[i - 1]:
            run += 1
        else:
            runs.append(run)
            run = 1
    runs.append(run)
    return dict(
        transitions_per_s=float(ch.sum() / dur),
        mean_dwell_s=float(np.mean(runs) * dt_sample),
        median_dwell_s=float(np.median(runs) * dt_sample),
        states_used=int(len(np.unique(path))),
    )


def participation_ratio(X, mean_rate, rate_floor=0.02, noise_var=None):
    if mean_rate < rate_floor:
        return float("nan")
    C = np.cov(X.T)
    if noise_var is not None:
        C = C - np.eye(len(C)) * noise_var
    tr = np.trace(C)
    ss = np.sum(C * C)
    if ss <= 0 or tr <= 0:
        return float("nan")
    return float(tr * tr / ss)


def entropy_rate(path):
    k = path.max() + 1
    T = np.zeros((k, k))
    for a, b in zip(path[:-1], path[1:]):
        T[a, b] += 1
    p = T.sum(1)
    p = p / p.sum()
    Tn = T / np.maximum(T.sum(1, keepdims=True), 1e-12)
    h = 0.0
    for i in range(k):
        row = Tn[i][Tn[i] > 0]
        h -= p[i] * np.sum(row * np.log2(row))
    return float(h)


def predictive_information(path, lag):
    a, b = path[:-lag], path[lag:]
    k = path.max() + 1
    J = np.zeros((k, k))
    for x, y in zip(a, b):
        J[x, y] += 1
    J /= J.sum()
    px = J.sum(1)
    py = J.sum(0)
    mask = J > 0
    return float(np.sum(J[mask] * np.log2(J[mask] / (px[:, None] * py[None, :])[mask])))


def order_entropy(path):
    seq = path[np.r_[True, path[1:] != path[:-1]]]
    if len(seq) < 6:
        return 0.0, 0.0
    k = path.max() + 1
    T = np.zeros((k, k))
    for a, b in zip(seq[:-1], seq[1:]):
        T[a, b] += 1
    rows = T[T.sum(1) > 0]
    rows = rows / rows.sum(1, keepdims=True)
    ent = float(np.mean([-np.sum(r[r > 0] * np.log2(r[r > 0])) for r in rows]))
    maxent = np.log2(max(k - 1, 2))
    return ent, ent / maxent


def cycle_score(path, max_period=400):
    seq = path[np.r_[True, path[1:] != path[:-1]]]
    n = len(seq)
    if n < 12:
        return 0.0, 0
    best, bp = 0.0, 0
    for p in range(1, min(max_period, n // 3)):
        m = float(np.mean(seq[p:] == seq[:-p]))
        if m > best:
            best, bp = m, p
    return best, bp


def lyapunov(sim_fn, params, T=12.0, dt=0.001, eps=1e-8, seed=3, n_restarts=40):
    p = dict(params, sigma=0.0)
    warm = sim_fn(p, T=2.0, dt=dt, seed=seed, sample_every=10000)
    r0 = warm["final"]["r"]
    a0 = warm["final"]["a"]
    rng = np.random.default_rng(seed + 5)
    d0 = rng.standard_normal(len(r0))
    d0 = eps * d0 / np.linalg.norm(d0)
    seg = T / n_restarts
    tot = 0.0
    rA, aA = r0.copy(), a0.copy()
    rB, aB = r0 + d0, a0.copy()
    for _ in range(n_restarts):
        oA = sim_fn(p, T=seg, dt=dt, seed=seed, sample_every=10000, r0=rA, a0=aA)
        oB = sim_fn(p, T=seg, dt=dt, seed=seed, sample_every=10000, r0=rB, a0=aB)
        rA, aA = oA["final"]["r"], oA["final"]["a"]
        rB, aB = oB["final"]["r"], oB["final"]["a"]
        d = np.linalg.norm(rB - rA)
        if d < 1e-14:
            d = 1e-14
        tot += np.log(d / eps)
        rB = rA + (rB - rA) * (eps / d)
        aB = aA + (aB - aA) * (eps / d)
    return float(tot / T)
