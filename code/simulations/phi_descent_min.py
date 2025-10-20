# --- PLOTTING HELPERS (many-body) -------------------------------------------
# --- MANY-BODY ORBIT (raw Metropolis, zoomed) -------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os

def _unwrap_segments(path: np.ndarray):
    segs = []
    for k in range(len(path)-1):
        a, b = path[k], path[k+1]
        d = b - a
        d -= np.round(d)
        segs.append(np.vstack([a, a+d]) % 1.0)
    return segs

def _closest_pair_indices(X: np.ndarray):
    """
    Return (i, j, dmin) where i<->j is the closest pair (torus metric).
    """
    N = X.shape[0]
    dmin = 1e9
    ii = jj = 0
    for i in range(N - 1):
        d = X[i] - X[i+1:]
        d -= np.round(d)                     # torus minimum-image
        r = np.sqrt((d**2).sum(axis=1))
        jrel = int(np.argmin(r))
        val = float(r[jrel])
        if val < dmin:
            dmin = val
            ii = i
            jj = i + 1 + jrel
    return ii, jj, dmin

def plot_orbit_many_body(outfile="paper/figures/orbit_many_body.png",
                         N_bg=20, pre_anneal=2000, steps=50000,
                         beta=2.8, step_single=0.005,
                         p_rotate=0.04, p_translate=0.94,
                         theta_std=0.0015, trans_std=0.028,
                         seed=42):
    """
    Run a small many-body Φ-descent and track a single bound pair in the crowd.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    rng = np.random.default_rng(seed)

    # use local implementations; define torus_delta here
    def torus_delta(a, b):
        d = b - a
        d -= np.round(d)
        return d

    # mst_phi and metropolis_step are expected to be defined in this module already.
    # (If you kept the safe fallbacks I provided earlier, they are present.)
    # Build initial state:
    X = rng.random((N_bg + 2, 2))

    # settle the background
    for _ in range(pre_anneal):
        metropolis_step(X, beta=beta, step=step_single, rng=rng)

    # choose one bound pair; keep tracking the same indices (no relock)
    i, j, _ = _closest_pair_indices(X)

    xi, xj = [], []
    for t in range(steps):
        u = rng.random()
        if u < p_rotate:
            # small rotation about midpoint
            mid = (X[i] + X[j]) * 0.5 % 1.0
            vi = torus_delta(mid, X[i]); vj = torus_delta(mid, X[j])
            th = rng.normal(0.0, theta_std)
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, -s], [s, c]])
            Xi_new = (mid + (R @ vi)) % 1.0
            Xj_new = (mid + (R @ vj)) % 1.0
            phi0 = mst_phi(X)
            oi, oj = X[i].copy(), X[j].copy()
            X[i], X[j] = Xi_new, Xj_new
            dphi = mst_phi(X) - phi0
            if dphi > 0 and rng.random() >= np.exp(-beta*dphi):
                X[i], X[j] = oi, oj
        elif u < p_rotate + p_translate:
            # coordinated translation (near-neutral)
            delta = rng.normal(0.0, trans_std, size=2)
            phi0 = mst_phi(X)
            oi, oj = X[i].copy(), X[j].copy()
            X[i] = (X[i] + delta) % 1.0
            X[j] = (X[j] + delta) % 1.0
            dphi = mst_phi(X) - phi0
            if dphi > 0 and rng.random() >= np.exp(-beta*dphi):
                X[i], X[j] = oi, oj
        else:
            # background jiggle
            metropolis_step(X, beta=beta, step=step_single, rng=rng)

        xi.append(X[i].copy()); xj.append(X[j].copy())

    xi = np.array(xi); xj = np.array(xj)
    segs_xi = _unwrap_segments(xi)
    segs_xj = _unwrap_segments(xj)

    # time colouring
    T = len(xi); tt = np.linspace(0, 1, T)

    plt.figure(figsize=(6.5, 6.0))
    # draw with gradual colour
    for k, seg in enumerate(segs_xi):
        plt.plot(seg[:,0], seg[:,1], lw=1.2, color=plt.cm.viridis(tt[min(k, T-1)]))
    for k, seg in enumerate(segs_xj):
        plt.plot(seg[:,0], seg[:,1], lw=1.2, color=plt.cm.plasma(tt[min(k, T-1)]))

    # start/end markers
    plt.scatter([xi[0,0], xj[0,0]], [xi[0,1], xj[0,1]], s=26, zorder=5)
    plt.scatter([xi[-1,0], xj[-1,0]], [xi[-1,1], xj[-1,1]], s=26, zorder=5)

    # auto-zoom around the path
    pts = np.vstack([xi, xj])
    xmin, ymin = pts.min(axis=0); xmax, ymax = pts.max(axis=0)
    span = max(xmax-xmin, ymax-ymin)
    margin = max(1e-3, 0.07 * span)
    plt.xlim(max(0.0, xmin - margin), min(1.0, xmax + margin))
    plt.ylim(max(0.0, ymin - margin), min(1.0, ymax + margin))

    plt.gca().set_aspect('equal', 'box')
    plt.title("Many-body $\Phi$-descent: tracked bound pair (raw Metropolis)")
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=240)
    plt.close()



def _ensure_rng(seed):
    return np.random.default_rng(seed if seed is not None else 0)

def _save(fig, outfile: str):
    import os
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def plot_clustering(sim, outfile="paper/figures/clustering.png", alpha=0.85):
    """
    Initial vs final projection (x-y); same axes, same aspect; light grid.
    Expects sim to expose sim.X0 (N,3) and sim.X (N,3) at final time.
    """
    X0 = getattr(sim, "X0", None)
    X  = getattr(sim, "X", None)
    if X0 is None or X is None:
        raise ValueError("Simulation object must carry X0 and X")

    fig, ax = plt.subplots(figsize=(6,6), constrained_layout=True)
    ax.scatter(X0[:,0], X0[:,1], s=14, alpha=0.35, label="initial")
    ax.scatter(X[:,0],  X[:,1],  s=22, alpha=alpha, label="final")
    ax.set_title("Clustering under $\Phi$-descent (projection)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, outfile)

def plot_mean_distance(sim, outfile="paper/figures/mean_distance.png"):
    """
    Show mean pairwise separation vs sample step; thin rolling median for smoothness.
    Expects sim.r_hist as 1D array of sampled mean distances.
    """
    r_hist = np.asarray(getattr(sim, "r_hist", []))
    if r_hist.size == 0:
        raise ValueError("Simulation object must carry r_hist (mean separation samples)")

    fig, ax = plt.subplots(figsize=(7.5,4.5))
    ax.plot(r_hist, lw=2)
    ax.set_title("Mean pairwise distance decreases under $\Phi$-descent")
    ax.set_xlabel("Sample step"); ax.set_ylabel(r"Mean separation $\bar r(t)$")
    ax.grid(True, linestyle="--", alpha=0.25)
    _save(fig, outfile)

def plot_inverse_square(sim, outfile="paper/figures/inverse_square.png", seed=1234):
    """
    Empirical scaling of |ΔΦ| when contracting a random close pair by ε vs separation r.
    Two curves:
      - exact two-point ΔΦ for φ = Σ (-1/r) (reference slope ~ r^{-2})
      - measured with random background (noisier)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    rng = np.random.default_rng(seed)

    # Pull a slice of state near the end for background estimate
    X = np.asarray(getattr(sim, "X", None))
    if X is None:
        raise ValueError("Simulation object must carry final positions X")
    N = len(X)

    # radii to test (log spaced), tiny radial contraction eps
    radii = np.geomspace(0.03, 0.35, 14)
    eps   = 1e-3

    # exact two-point ΔΦ ~ 1/(r-eps) - 1/r ≈ +eps / r^2  (since Φ uses -1/r on MST edges)
    exact = eps / (radii**2)

    # with background: choose random pairs at approx these separations, measure MST-based ΔΦ
    M = 24
    noisy = []
    for rt in radii:
        best = []
        tol = 0.02 + 0.15*rt
        trials = 0
        while len(best) < M and trials < 600:
            i, j = rng.integers(0, N, size=2)
            if i == j:
                continue
            dij = np.linalg.norm(X[i]-X[j])
            if abs(dij-rt) < tol:
                # measure ΔΦ by contracting i->i' slightly toward j
                Xi = X.copy()
                dir_ij = (X[j]-X[i]) / (np.linalg.norm(X[j]-X[i]) + 1e-12)
                Xi[i] = Xi[i] + (-eps) * dir_ij  # reduce r by eps
                dphi = mst_phi(Xi) - mst_phi(X)  # <-- use local mst_phi
                best.append(abs(dphi))
            trials += 1
        noisy.append(np.median(best) if best else np.nan)
    noisy = np.array(noisy)

    # plot
    fig, ax = plt.subplots(figsize=(8.5,5.0))
    ax.set_title(r"Inverse-square behaviour from $\Phi=\sum(-1/r)$ on MST")
    ax.loglog(radii, exact, label="two-point (exact $\\propto r^{-2}$)")
    ax.loglog(radii, noisy, "o", label="with background (noisy)")
    # visual reference slope r^{-2}
    ref = exact[0]*(radii/radii[0])**(-2)
    ax.loglog(radii, ref, "--", label=r"reference $r^{-2}$")
    ax.set_xlabel("Separation $r$")
    ax.set_ylabel(r"$|\Delta\Phi|$ upon contraction")
    ax.grid(True, which="both", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="lower left")

    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)

# ===== Shim: minimal driver + result container so external scripts can call us =====
from dataclasses import dataclass
import numpy as np

@dataclass
class SimResult:
    X0: np.ndarray          # initial positions (N,2)
    X:  np.ndarray          # final positions (N,2)
    r_hist: np.ndarray      # sampled mean separations (T,)
    meta: dict              # parameters for reproducibility

def _mean_pairwise_distance(X: np.ndarray) -> float:
    # O(N^2) mean pairwise distance on the torus
    N = X.shape[0]
    dsum = 0.0
    cnt  = 0
    for i in range(N-1):
        d = X[i] - X[i+1:]
        d -= np.round(d)              # torus minimum image
        dist = np.sqrt((d**2).sum(axis=1))
        dsum += float(dist.sum())
        cnt  += dist.size
    return dsum / max(1, cnt)

# ===== Minimal Φ machinery: safe fallbacks if missing =====
import numpy as np

def _torus_delta(a, b):
    """Minimum-image delta on [0,1)^2."""
    d = b - a
    d -= np.round(d)
    return d

def _pair_distance(a, b):
    d = _torus_delta(a, b)
    return float(np.sqrt((d**2).sum()))

# Fallback MST-based Φ if not already defined in this module
if 'mst_phi' not in globals():
    import networkx as nx
    def mst_phi(X: np.ndarray) -> float:
        """
        Φ estimator: sum of edge weights on MST with weights w = -1/d (toroidal).
        Lower Φ (more negative) favours shorter edges -> clustering.
        """
        N = X.shape[0]
        G = nx.Graph()
        for i in range(N):
            G.add_node(i)
        # complete graph weights (O(N^2), fine for N≈120)
        for i in range(N-1):
            for j in range(i+1, N):
                d = _pair_distance(X[i], X[j])
                w = -1.0 / (d + 1e-12)
                G.add_edge(i, j, weight=w)
        T = nx.minimum_spanning_tree(G, weight='weight')
        phi = 0.0
        for u, v, data in T.edges(data=True):
            phi += data['weight']
        return phi

# Fallback single-particle Metropolis step if not already defined
if 'metropolis_step' not in globals():
    def metropolis_step(X: np.ndarray, beta: float, step: float, rng: np.random.Generator):
        """
        Propose moving one random particle by N(0, step) on the torus; accept by Metropolis on ΔΦ.
        Returns (X_new, dphi, accepted:int)
        """
        i = int(rng.integers(0, X.shape[0]))
        X0 = X
        phi0 = mst_phi(X0)

        X1 = X0.copy()
        X1[i] = (X1[i] + rng.normal(0.0, step, size=2)) % 1.0

        phi1 = mst_phi(X1)
        dphi = phi1 - phi0
        if dphi <= 0.0 or rng.random() < np.exp(-beta * dphi):
            return X1, dphi, 1
        else:
            return X0, dphi, 0


def run_simulation(
    N: int = 120,
    steps: int = 8000,
    beta: float = 10.0,
    step: float = 0.02,
    seed: int = 42,
    sample_every: int = 40,
    progress: bool = False,
):
    """
    Minimal driver that uses the local Metropolis step and MST Φ.
    Returns SimResult with X0, X, r_hist, meta.
    Assumes the following functions already exist in this module:
        - metropolis_step(X, beta, step, rng)
        - mst_phi(X)
    """
    rng = np.random.default_rng(seed)
    X = rng.random((N, 2))
    X0 = X.copy()

    r_hist = []
    for t in range(steps):
        X, _, _ = metropolis_step(X, beta=beta, step=step, rng=rng)
        if (t % sample_every) == 0:
            r_hist.append(_mean_pairwise_distance(X))
        if progress and (t % max(1, steps // 20) == 0):
            print(f"[phi] t={t}/{steps}", flush=True)

    return SimResult(
        X0=X0,
        X=X,
        r_hist=np.asarray(r_hist),
        meta=dict(N=N, steps=steps, beta=beta, step=step, seed=seed, sample_every=sample_every),
    )
