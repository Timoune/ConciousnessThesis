import numpy as np

import network
import analysis as an

FIXED_K = 8


def _labels(res, seed=0, burn=0.1):
    X = an.features(res)
    X = X[int(len(X) * burn):]
    Z = (X - X.mean(0)) / np.maximum(X.std(0), 1e-8)
    path = an.GaussianHMM(FIXED_K, seed=seed).fit(Z).viterbi(Z)
    return X, path


def probe_periodicity(T=40.0):
    rows = []
    for tag, ov in [("noise off", dict(sigma=0.0)), ("noise default", {}), ("noise x3", dict(sigma=0.18))]:
        res = network.simulate(ov, T=T)
        X, path = _labels(res)
        cs, cp = an.cycle_score(path)
        d = an.dwell_stats(path, res["dt_sample"])
        rows.append(dict(
            case=tag,
            transitions_per_s=d["transitions_per_s"],
            dwell_ms=d["mean_dwell_s"] * 1000,
            cycle_score=cs,
            cycle_period=cp,
            order_entropy=an.order_entropy(path)[1],
            bump_coherence=float(res["coherence"].mean()),
            pr=an.participation_ratio(X, float(res["S"].mean())),
        ))
    return rows


def probe_lyapunov(gains=(0.0, 0.9, 1.5, 1.8, 2.4), T=8.0):
    return [
        dict(g_rand=g, lambda_max=an.lyapunov(network.simulate, dict(network.DEFAULTS, g_rand=g), T=T))
        for g in gains
    ]


def probe_intensity(taus=(0.5, 0.25, 0.12, 0.06, 0.03, 0.015), seeds=4, T=40.0, horizon_ms=100):
    lag = int(horizon_ms / 1000 / network.DEFAULTS["tau_r"] * 0.2)
    lag = max(1, int(horizon_ms / 10))
    out = []
    for ta in taus:
        pi, co, tr = [], [], []
        for s in range(seeds):
            res = network.simulate(dict(tau_a=ta), T=T, seed=s + 1)
            _, path = _labels(res, seed=s)
            pi.append(an.predictive_information(path, lag))
            co.append(float(res["coherence"].mean()))
            tr.append(an.dwell_stats(path, res["dt_sample"])["transitions_per_s"])
        out.append(dict(tau_a=ta, pi_mean=float(np.mean(pi)), pi_sd=float(np.std(pi)),
                        bump_coherence=float(np.mean(co)), transitions_per_s=float(np.mean(tr))))
    return out


def _ridge_r2(X, Y, lag, ridge=1e-3, train=0.7):
    Xa = np.column_stack([X[:-lag], np.ones(len(X) - lag)])
    Ya = Y[lag:]
    c = int(train * len(Xa))
    A = Xa[:c].T @ Xa[:c] + ridge * np.eye(Xa.shape[1])
    w = np.linalg.solve(A, Xa[:c].T @ Ya[:c])
    p = Xa[c:] @ w
    ss = ((Ya[c:] - p) ** 2).sum()
    tot = ((Ya[c:] - Ya[c:].mean(0)) ** 2).sum()
    return float(1 - ss / tot)


def probe_content_baseline(T=60.0, horizons_ms=(100, 300, 500)):
    res = network.simulate(T=T, record_units=True)
    b = len(res["theta"]) // 10
    U, A, th = res["units"][b:], res["adapt"][b:], res["theta"][b:]
    Y = np.column_stack([np.cos(th), np.sin(th)])
    Scl = res["S"][b:]
    onehot = np.eye(network.K)[Scl.argmax(1)]
    pops = dict(
        winner_label=onehot,
        sensory_rates=Scl,
        ring_rates=U[:, network.AL],
        ring_adaptation=A[:, network.AL],
        readout_rates=U[:, network.RL],
    )
    return [
        dict(horizon_ms=h, **{k: _ridge_r2(v, Y, max(1, h // 10)) for k, v in pops.items()})
        for h in horizons_ms
    ]


def _table(title, rows):
    print(f"\n{title}")
    keys = list(rows[0].keys())
    print("  " + "  ".join(f"{k:>17s}" for k in keys))
    for r in rows:
        cells = []
        for k in keys:
            v = r[k]
            cells.append(f"{v:>17.3f}" if isinstance(v, float) else f"{str(v):>17s}")
        print("  " + "  ".join(cells))


if __name__ == "__main__":
    _table("Q1 â noiseless periodicity and noise ablation", probe_periodicity())
    _table("Q1 â largest Lyapunov exponent vs random-weight gain", probe_lyapunov())
    _table("Q5 â intensity proxy sweep over adaptation timescale", probe_intensity())
    _table("Q2 â future-content prediction, held-out R^2", probe_content_baseline())
