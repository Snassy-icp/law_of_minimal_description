#!/usr/bin/env python3
"""
Predictive Compression Dynamics (PCD)
Phase I / Phase II / Baselines / Controls / Quantization Ablation
"""

import numpy as np
import gzip
import io
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.stats import pearsonr

np.random.seed(0)

# ================================================================
# Surrogate Φ_b
# ================================================================
def phi_b(x, a=0.05):
    """Compute surrogate Φ_b = Σ_{i<j} 1/sqrt(r_ij^2 + a^2)."""
    d = distance_matrix(x, x)
    iu = np.triu_indices(len(x), 1)
    r = d[iu]
    return np.sum(1.0 / np.sqrt(r * r + a * a))


def grad_phi_b(x, a=0.05):
    """Gradient of Φ_b (attractive flow direction is -∇Φ_b)."""
    N = len(x)
    grad = np.zeros_like(x)
    for i in range(N):
        diff = x[i] - x                # shape (N,3)
        r2 = np.sum(diff**2, axis=1) + a * a
        r3 = r2**1.5                   # (r^2 + a^2)^{3/2}
        grad[i] = np.sum(diff / r3[:, None], axis=0)
    # Return -∇Φ_b, i.e. the force-like direction that lowers Φ_b
    return -grad


# ================================================================
# Baseline geometric metrics
# ================================================================
def radius_of_gyration(x):
    """Root mean square distance from centroid."""
    c = np.mean(x, axis=0)
    return np.sqrt(np.mean(np.sum((x - c) ** 2, axis=1)))


def mean_nearest_neighbor(x):
    """Average nearest-neighbor distance."""
    d = distance_matrix(x, x)
    np.fill_diagonal(d, np.inf)
    return np.mean(np.min(d, axis=1))


def coordinate_variance(x):
    """Variance of coordinates (flattened)."""
    return np.var(x)


# ================================================================
# Compression utilities
# ================================================================
def gzip_bytes(b):
    with io.BytesIO() as buf:
        with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
            f.write(b)
        return len(buf.getvalue())


def phase1_histogram_encoder(x, bins=64):
    """
    Phase I encoder: pairwise distance histogram -> gzip.
    Does NOT depend on coordinate ordering.
    """
    d = distance_matrix(x, x)
    iu = np.triu_indices(len(x), 1)
    hist, _ = np.histogram(d[iu], bins=bins, range=(0, np.max(d)))
    return gzip_bytes(hist.tobytes())


def phase2_coordinate_encoder(x, dx=1e-2, shuffle=False):
    """
    Phase II encoder: quantized coordinates -> gzip.

    Phase IIa: shuffle=False (fixed particle order)
    Phase IIb: shuffle=True  (random permutation of particle order)

    Reviewer control: Phase IIb tests whether gzip gains are just
    ordering artifacts rather than true spatial regularity.
    """
    q = np.round(x / dx).astype(np.int32)
    if shuffle:
        q = q.copy()
        np.random.shuffle(q)  # permute particle order
    return gzip_bytes(q.tobytes())


# ================================================================
# Gradient descent with backtracking
# ================================================================
def run_dynamics(x0, steps=400, a=0.05, eta0=0.05):
    """
    Gradient descent with backtracking line search to enforce
    monotone nonincreasing Φ_b. Returns Φ_b trace and snapshots.
    """
    x = x0.copy()
    phis = [phi_b(x, a)]
    snapshots = [x.copy()]
    for t in range(steps):
        g = grad_phi_b(x, a)   # this is -∇Φ_b
        eta = eta0
        # backtracking line search
        while True:
            x_new = x - eta * g
            if phi_b(x_new, a) <= phis[-1]:
                break
            eta *= 0.5
            if eta < 1e-6:
                break
        x = x_new
        phis.append(phi_b(x, a))
        snapshots.append(x.copy())
    return np.array(phis), snapshots


# ================================================================
# Ensemble generators
# ================================================================
def make_uniform(N):
    """Uniform points in [0,1]^3."""
    return np.random.rand(N, 3)


def make_lattice(N):
    """
    Regular grid (roughly cubic), then add small Gaussian jitter.
    Works for arbitrary N by oversampling ceil(m^3) and truncating.
    """
    m = int(np.ceil(N ** (1.0 / 3.0)))
    xs = np.linspace(0, 1, m)
    ys = np.linspace(0, 1, m)
    zs = np.linspace(0, 1, m)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    pts = pts[:N]
    noise = 0.02 * np.random.randn(N, 3)
    return pts + noise


def make_blobs(N):
    """Two clusters: one near origin, one shifted along x."""
    half = N // 2
    a = 0.1 * np.random.randn(half, 3)
    b = 0.1 * np.random.randn(N - half, 3) + np.array([1.0, 0.0, 0.0])
    return np.vstack([a, b])


ENSEMBLES = {
    "uniform40":   lambda: make_uniform(40),
    "lattice40":   lambda: make_lattice(40),
    "blobs40":     lambda: make_blobs(40),
    "uniform400":  lambda: make_uniform(400),
}


# ================================================================
# Correlation helper
# ================================================================
def safe_corr(x, y):
    """
    Return (Pearson r, p). If either input has zero variance, return (0,1).
    We mostly care about r as an effect size.
    """
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 1.0
    return pearsonr(x, y)


# ================================================================
# Main experiment
# ================================================================
def main():
    results = []

    for name, gen in ENSEMBLES.items():
        x0 = gen()
        phis, snaps = run_dynamics(x0)

        n_snaps = len(snaps)
        # Subsample every 20th snapshot to reduce temporal autocorrelation
        idx = np.arange(0, n_snaps, 20)
        phis_sub = phis[idx]

        # Phase I encoder (pair-distance histogram, no dx dependence)
        c1 = [phase1_histogram_encoder(snaps[i]) for i in idx]

        # We'll sweep Δx to test quantization sensitivity (Phase II encoders)
        deltas = [1e-1, 1e-2, 1e-3]
        for dx in deltas:
            # Phase IIa = coordinate gzip, fixed order
            c2 = [phase2_coordinate_encoder(snaps[i], dx=dx, shuffle=False) for i in idx]
            # Phase IIb = coordinate gzip, shuffled order
            c2s = [phase2_coordinate_encoder(snaps[i], dx=dx, shuffle=True)  for i in idx]

            # Correlations of Φ_b with each compressed-size estimate
            r1, _  = safe_corr(phis_sub, c1)   # Φ_b vs Phase I
            r2, _  = safe_corr(phis_sub, c2)   # Φ_b vs Phase IIa
            r2s, _ = safe_corr(phis_sub, c2s)  # Φ_b vs Phase IIb

            # Baselines vs Phase IIa (the external compressor target)
            rg   = [radius_of_gyration(snaps[i])    for i in idx]
            nnd  = [mean_nearest_neighbor(snaps[i]) for i in idx]
            varc = [coordinate_variance(snaps[i])   for i in idx]

            rb_rg,  _  = safe_corr(rg,  c2)
            rb_nnd, _  = safe_corr(nnd, c2)
            rb_var, _  = safe_corr(varc, c2)

            results.append(
                (name,
                 len(snaps[0]),
                 dx,
                 len(idx),
                 r1,
                 r2,
                 r2s,
                 rb_rg,
                 rb_nnd,
                 rb_var)
            )

            # ---------- Plot base name ----------
            fig_base = f"figures/{name}_dx{dx:g}"

            # ---------- Plot 1: Φ_b vs iteration ----------
            # (Yes, we will regenerate/overwrite this per dx, which is harmless.
            #  This preserves filenames matching dx in the paper.)
            plt.figure()
            plt.plot(phis, marker='o', markersize=2, linewidth=1)
            plt.xlabel("Iteration")
            plt.ylabel("Φ_b")
            plt.title(f"{name}: Φ_b vs iteration")
            plt.tight_layout()
            plt.savefig(f"{fig_base}_phib_vs_iter.png", dpi=150)
            plt.close()

            # ---------- Plot 2: Φ_b vs Phase I compressed size ----------
            # Internal-consistency check: pair-distance histogram encoder
            plt.figure()
            plt.scatter(phis_sub, c1, c=idx, cmap="cividis", s=30)
            plt.xlabel("Φ_b")
            plt.ylabel("Phase I (pairwise histogram gzip) bytes")
            plt.title(f"{name}, Δx={dx:g}: Φ_b vs Phase I (r={r1:.3f})")
            cbar = plt.colorbar()
            cbar.set_label("Iteration index")
            plt.tight_layout()
            plt.savefig(f"{fig_base}_phib_vs_phase1.png", dpi=150)
            plt.close()

            # ---------- Plot 3: Φ_b vs Phase IIa compressed size ----------
            # External compressor, fixed particle order
            plt.figure()
            plt.scatter(phis_sub, c2, c=idx, cmap="viridis", s=30)
            plt.xlabel("Φ_b")
            plt.ylabel("Phase IIa compressed size (bytes)")
            plt.title(f"{name}, Δx={dx:g}: Φ_b vs Phase IIa (r={r2:.3f})")
            cbar = plt.colorbar()
            cbar.set_label("Iteration index")
            plt.tight_layout()
            plt.savefig(f"{fig_base}_phib_vs_phase2a.png", dpi=150)
            plt.close()

            # ---------- Plot 4: Φ_b vs Phase IIb compressed size ----------
            # External compressor, shuffled particle order (ordering control)
            plt.figure()
            plt.scatter(phis_sub, c2s, c=idx, cmap="plasma", s=30)
            plt.xlabel("Φ_b")
            plt.ylabel("Phase IIb shuffled compressed size (bytes)")
            plt.title(f"{name}, Δx={dx:g}: Φ_b vs Phase IIb (r={r2s:.3f})")
            cbar = plt.colorbar()
            cbar.set_label("Iteration index")
            plt.tight_layout()
            plt.savefig(f"{fig_base}_phib_vs_phase2b.png", dpi=150)
            plt.close()

        print(f"{name} done.")

    # ========== Summary table ==========
    print("\nSummary (decorrelated snapshots every 20th):")
    header = (
        "ensemble",
        "N",
        "Δx",
        "n_eff",
        "r_PhI",
        "r_PhIIa",
        "r_PhIIb",
        "r_Rg",
        "r_NND",
        "r_Var",
    )
    print(
        "{:<12s} {:>5s} {:>6s} {:>6s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}".format(
            *header
        )
    )

    for (
        ens,
        Nval,
        dx,
        neff,
        r1,
        r2,
        r2s,
        rb_rg,
        rb_nnd,
        rb_var,
    ) in results:
        print(
            "{:<12s} {:>5d} {:>6.0e} {:>6d} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f}".format(
                ens,
                Nval,
                dx,
                neff,
                r1,
                r2,
                r2s,
                rb_rg,
                rb_nnd,
                rb_var,
            )
        )

    print(
        "\nPhase I  = pair-distance histogram encoder (explicit pair structure)."
    )
    print(
        "Phase IIa = coordinate encoder (fixed order gzip on quantized coords)."
    )
    print(
        "Phase IIb = shuffled-coordinate encoder (random order gzip) as an ordering control."
    )
    print(
        "r_Rg / r_NND / r_Var are baseline correlations of simple geometric metrics "
        "with Phase IIa compressed size."
    )
    print("Figures saved under ./figures/")


if __name__ == "__main__":
    main()
