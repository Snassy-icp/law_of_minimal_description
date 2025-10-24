#!/usr/bin/env python3
"""
Predictive Compression Dynamics (PCD)
Phase I / Phase II / Baselines / Controls / Quantization Ablation
Multi-seed replication and result aggregation.
"""

import numpy as np
import gzip, io, time, os
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.stats import pearsonr
import pandas as pd

# ================================================================
# Surrogate Φ_b
# ================================================================
def phi_b(x, a=0.05):
    d = distance_matrix(x, x)
    iu = np.triu_indices(len(x), 1)
    r = d[iu]
    return np.sum(1.0 / np.sqrt(r * r + a * a))

def grad_phi_b(x, a=0.05):
    N = len(x)
    grad = np.zeros_like(x)
    for i in range(N):
        diff = x[i] - x
        r2 = np.sum(diff**2, axis=1) + a * a
        r3 = r2**1.5
        grad[i] = np.sum(diff / r3[:, None], axis=0)
    return -grad  # descent direction

# ================================================================
# Baseline geometric metrics
# ================================================================
def radius_of_gyration(x):
    c = np.mean(x, axis=0)
    return np.sqrt(np.mean(np.sum((x - c) ** 2, axis=1)))

def mean_nearest_neighbor(x):
    d = distance_matrix(x, x)
    np.fill_diagonal(d, np.inf)
    return np.mean(np.min(d, axis=1))

def coordinate_variance(x):
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
    d = distance_matrix(x, x)
    iu = np.triu_indices(len(x), 1)
    hist, _ = np.histogram(d[iu], bins=bins, range=(0, np.max(d)))
    return gzip_bytes(hist.tobytes())

def phase2_coordinate_encoder(x, dx=1e-2, shuffle=False):
    q = np.round(x / dx).astype(np.int32)
    if shuffle:
        q = q.copy()
        np.random.shuffle(q)
    return gzip_bytes(q.tobytes())

# ================================================================
# Gradient descent with backtracking
# ================================================================
def run_dynamics(x0, steps=400, a=0.05, eta0=0.05):
    x = x0.copy()
    phis = [phi_b(x, a)]
    snapshots = [x.copy()]
    for _ in range(steps):
        g = grad_phi_b(x, a)
        eta = eta0
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
def make_uniform(N): return np.random.rand(N, 3)

def make_lattice(N):
    m = int(np.ceil(N ** (1.0 / 3.0)))
    xs = np.linspace(0, 1, m)
    ys = np.linspace(0, 1, m)
    zs = np.linspace(0, 1, m)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)[:N]
    return pts + 0.02 * np.random.randn(N, 3)

def make_blobs(N):
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
# Helper: safe correlation
# ================================================================
def safe_corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 1.0
    return pearsonr(x, y)

# ================================================================
# Single-seed experiment
# ================================================================
def run_all(seed=0):
    np.random.seed(seed)
    results = []
    os.makedirs("figures", exist_ok=True)

    for name, gen in ENSEMBLES.items():
        x0 = gen()
        phis, snaps = run_dynamics(x0)
        n_snaps = len(snaps)
        idx = np.arange(0, n_snaps, 20)
        phis_sub = phis[idx]
        c1 = [phase1_histogram_encoder(snaps[i]) for i in idx]

        for dx in [1e-1, 1e-2, 1e-3]:
            c2  = [phase2_coordinate_encoder(snaps[i], dx, False) for i in idx]
            c2s = [phase2_coordinate_encoder(snaps[i], dx, True)  for i in idx]

            r1, _  = safe_corr(phis_sub, c1)
            r2, _  = safe_corr(phis_sub, c2)
            r2s,_  = safe_corr(phis_sub, c2s)

            rg   = [radius_of_gyration(snaps[i]) for i in idx]
            nnd  = [mean_nearest_neighbor(snaps[i]) for i in idx]
            varc = [coordinate_variance(snaps[i]) for i in idx]

            rb_rg,  _  = safe_corr(rg,  c2)
            rb_nnd, _  = safe_corr(nnd, c2)
            rb_var, _  = safe_corr(varc, c2)

            results.append({
                "seed": seed, "ensemble": name, "N": len(snaps[0]), "Δx": dx, "n_eff": len(idx),
                "r_PhI": r1, "r_PhIIa": r2, "r_PhIIb": r2s,
                "r_Rg": rb_rg, "r_NND": rb_nnd, "r_Var": rb_var
            })

            # --- plotting (identical to original) ---
            fig_base = f"figures/{name}_dx{dx:g}"
            plt.figure(); plt.plot(phis, marker='o', ms=2, lw=1)
            plt.xlabel("Iteration"); plt.ylabel("Φ_b"); plt.title(f"{name}: Φ_b vs iteration")
            plt.tight_layout(); plt.savefig(f"{fig_base}_phib_vs_iter.png", dpi=150); plt.close()

            for phase, data, cmap, lab, r in [
                ("phase2a", c2, "viridis", "Phase IIa", r2),
                ("phase2b", c2s, "plasma", "Phase IIb", r2s),
                ("phase1", c1, "cividis", "Phase I", r1),
            ]:
                plt.figure()
                plt.scatter(phis_sub, data, c=idx, cmap=cmap, s=30)
                plt.xlabel("Φ_b"); plt.ylabel(f"{lab} compressed size (bytes)")
                plt.title(f"{name}, Δx={dx:g}: Φ_b vs {lab} (r={r:.3f})")
                cbar = plt.colorbar(); cbar.set_label("Iteration index")
                plt.tight_layout(); plt.savefig(f"{fig_base}_phib_vs_{phase}.png", dpi=150)
                plt.close()
        print(f"{name} done (seed={seed}).")
    return pd.DataFrame(results)

# ================================================================
# Multi-seed replication driver
# ================================================================
def main():
    seeds = list(range(50))  # adjust as desired
    all_results = []
    t0 = time.time()

    for s in seeds:
        print(f"\n=== Running seed {s} ===")
        start = time.time()
        df = run_all(seed=s)
        df["runtime_sec"] = time.time() - start
        all_results.append(df)
        print(f"Seed {s} done in {df['runtime_sec'].iloc[0]:.1f} s")

    total = time.time() - t0
    print(f"\nAll {len(seeds)} seeds completed in {total/60:.1f} minutes total.")

    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv("results_multiseed.csv", index=False)

    summary = (all_df.groupby(["ensemble", "Δx"])
               .agg(["mean", "std"])
               .round(3))
    summary.to_csv("results_summary.csv")
    print("\nSaved results_multiseed.csv and results_summary.csv")

    print("\nAggregate mean±std correlations:")
    print(summary[["r_PhI", "r_PhIIa", "r_PhIIb", "r_Rg", "r_NND", "r_Var"]])

if __name__ == "__main__":
    main()
