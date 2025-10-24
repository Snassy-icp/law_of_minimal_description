#!/usr/bin/env python3
"""
Predictive Compression Dynamics – Multi-seed, Multi-compressor experiment
Fully parallel, reproducible, writes master CSV and summary.
"""

import numpy as np, gzip, io, bz2, lzma, time, os, multiprocessing as mp, pandas as pd
from scipy.spatial import distance_matrix
from scipy.stats import pearsonr
import zstandard as zstd
import matplotlib.pyplot as plt

# ================================================================
# Surrogate Φ_b and gradient
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
    return -grad

# ================================================================
# Geometric baselines
# ================================================================
def radius_of_gyration(x):
    c = np.mean(x, axis=0)
    return np.sqrt(np.mean(np.sum((x - c)**2, axis=1)))

def mean_nearest_neighbor(x):
    d = distance_matrix(x, x)
    np.fill_diagonal(d, np.inf)
    return np.mean(np.min(d, axis=1))

def coordinate_variance(x):
    return np.var(x)

# ================================================================
# Compression backends
# ================================================================
def compress_bytes(b, method="gzip"):
    if method == "gzip":
        with io.BytesIO() as buf:
            with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
                f.write(b)
            return len(buf.getvalue())
    elif method == "bz2":
        return len(bz2.compress(b, compresslevel=9))
    elif method == "lzma":
        return len(lzma.compress(b, preset=6))
    elif method == "zstd":
        cctx = zstd.ZstdCompressor(level=6)
        return len(cctx.compress(b))
    elif method == "none":
        return len(b)
    else:
        raise ValueError(f"Unknown compressor: {method}")

# ================================================================
# Encoders
# ================================================================
def phase1_histogram_encoder(x, bins=64, method="gzip"):
    d = distance_matrix(x, x)
    iu = np.triu_indices(len(x), 1)
    hist, _ = np.histogram(d[iu], bins=bins, range=(0, np.max(d)))
    return compress_bytes(hist.tobytes(), method=method)

def phase2_coordinate_encoder(x, dx=1e-2, shuffle=False, method="gzip"):
    q = np.round(x / dx).astype(np.int32)
    if shuffle:
        q = q.copy()
        np.random.shuffle(q)
    return compress_bytes(q.tobytes(), method=method)

# ================================================================
# Dynamics
# ================================================================
def run_dynamics(x0, steps=400, a=0.05, eta0=0.05):
    x = x0.copy()
    phis = [phi_b(x, a)]
    snaps = [x.copy()]
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
        snaps.append(x.copy())
    return np.array(phis), snaps

# ================================================================
# Ensembles
# ================================================================
def make_uniform(N): return np.random.rand(N,3)
def make_lattice(N):
    m = int(np.ceil(N ** (1.0/3.0)))
    xs = np.linspace(0,1,m); ys = np.linspace(0,1,m); zs = np.linspace(0,1,m)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.stack([gx, gy, gz], axis=-1).reshape(-1,3)[:N]
    return pts + 0.02*np.random.randn(N,3)
def make_blobs(N):
    half = N//2
    a = 0.1*np.random.randn(half,3)
    b = 0.1*np.random.randn(N-half,3)+np.array([1.0,0.0,0.0])
    return np.vstack([a,b])

ENSEMBLES = {
    "uniform40":   lambda: make_uniform(40),
    "lattice40":   lambda: make_lattice(40),
    "blobs40":     lambda: make_blobs(40),
    "uniform400":  lambda: make_uniform(400),
}

# ================================================================
# Helpers
# ================================================================
def safe_corr(x,y):
    if np.std(x)==0 or np.std(y)==0:
        return 0.0,1.0
    return pearsonr(x,y)

# ================================================================
# One seed task
# ================================================================
def run_seed(seed):
    np.random.seed(seed)
    results=[]
    for name, gen in ENSEMBLES.items():
        x0=gen()
        phis, snaps = run_dynamics(x0)
        idx=np.arange(0,len(snaps),20)
        phis_sub=phis[idx]
        for method in ["gzip","bz2","lzma","zstd","none"]:
            c1=[phase1_histogram_encoder(snaps[i],method=method) for i in idx]
            for dx in [1e-1,1e-2,1e-3]:
                c2=[phase2_coordinate_encoder(snaps[i],dx,False,method=method) for i in idx]
                c2s=[phase2_coordinate_encoder(snaps[i],dx,True,method=method) for i in idx]
                r1,_=safe_corr(phis_sub,c1)
                r2,_=safe_corr(phis_sub,c2)
                r2s,_=safe_corr(phis_sub,c2s)
                rg=[radius_of_gyration(snaps[i]) for i in idx]
                nnd=[mean_nearest_neighbor(snaps[i]) for i in idx]
                varc=[coordinate_variance(snaps[i]) for i in idx]
                rb_rg,_=safe_corr(rg,c2)
                rb_nnd,_=safe_corr(nnd,c2)
                rb_var,_=safe_corr(varc,c2)
                results.append({
                    "seed":seed,"ensemble":name,"Δx":dx,"compressor":method,"n_eff":len(idx),
                    "r_PhI":r1,"r_PhIIa":r2,"r_PhIIb":r2s,"r_Rg":rb_rg,"r_NND":rb_nnd,"r_Var":rb_var
                })
        print(f"Seed {seed}, {name} done.")
    return results

# ================================================================
# Driver
# ================================================================
def main():
    N_PROCS=8
    seeds=list(range(50))
    t0=time.time()
    with mp.Pool(N_PROCS) as pool:
        all_runs=pool.map(run_seed,seeds)
    all_flat=[r for sub in all_runs for r in sub]
    df=pd.DataFrame(all_flat)
    df.to_csv("results_multiseed_full.csv",index=False)
    summary=(df.groupby(["ensemble","Δx","compressor"])
             .agg(["mean","std"])
             .round(3))
    summary.to_csv("results_summary_full.csv")
    print(f"\nAll {len(seeds)} seeds finished in {(time.time()-t0)/60:.1f} min total.")
    print(summary[["r_PhIIa","r_PhIIb","r_Rg","r_NND"]])

if __name__=="__main__":
    main()
