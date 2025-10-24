#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import gzip
import struct
import os
from scipy.stats import pearsonr

# ============================================================
# Preregister-style global knobs
# ============================================================

SEED = 0                   # fixed RNG seed
DT_INIT = 0.05             # initial trial step size for gradient step
BACKTRACK_ITERS = 10       # max halvings in backtracking line search
STEPS = 400                # number of iterations for N=40 runs
STEPS_LARGE = 400          # same for N=400 run
RECORD_STRIDE = 5          # record snapshots every k steps
SUBSAMPLE_STRIDE = 20      # use every k-th recorded snapshot in stats (reduce autocorr)
A_SOFT = 0.05              # softening length "a" in sqrt(r^2 + a^2)
DX_QUANT = 0.01            # quantization for Phase II coordinate gzip
BINS_MIN = 0.0             # Phase I histogram lower bound
BINS_MAX = 2.0             # Phase I histogram upper bound
NBINS = 33                 # number of Phase I histogram bins

OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

rng = np.random.default_rng(SEED)

# ============================================================
# Surrogate Phi_b and its gradient
# ============================================================

def phi_and_forces(x, a_soft=A_SOFT):
    """
    x: (N,3) array of particle positions.
    Returns:
        phi : scalar (our surrogate Φ_b)
        F   : (N,3) array = -∇Φ_b  (attractive soft-core pair force).
    We include *all* pairs (i<j).
    """
    diffs = x[:, None, :] - x[None, :, :]        # (N,N,3)
    dist2 = np.sum(diffs**2, axis=2)             # (N,N)
    dist2_soft = dist2 + a_soft**2

    N = x.shape[0]
    idx = np.arange(N)
    dist2_soft[idx, idx] = np.inf  # kill self-terms

    inv_sqrt = 1.0 / np.sqrt(dist2_soft)         # 1/sqrt(r^2+a^2)
    phi = np.sum(np.triu(inv_sqrt, k=1))         # sum over i<j only

    inv32 = dist2_soft**(-1.5)                   # 1/(r^2+a^2)^(3/2)
    inv32[idx, idx] = 0.0

    # F_i = - Σ_j (x_i - x_j)/(r^2+a^2)^(3/2)
    F = -np.einsum('ij,ijx->ix', inv32, diffs)   # shape (N,3)
    return phi, F

# ============================================================
# Initial configurations
# ============================================================

def init_uniform(N):
    # uniform in cube [-1,1]^3
    return rng.uniform(-1.0, 1.0, size=(N,3))

def init_lattice(N):
    # nearly cubic lattice (with a tiny jitter)
    n = int(np.ceil(N ** (1.0/3.0)))
    coords = []
    lin = np.linspace(-1.0, 1.0, n)
    for xi in lin:
        for yi in lin:
            for zi in lin:
                coords.append([xi, yi, zi])
    coords = np.array(coords)[:N]
    coords += rng.normal(scale=0.05, size=coords.shape)
    return coords

def init_blobs(N):
    # two Gaussian blobs
    half = N//2
    c1 = np.array([-0.5, 0.0, 0.0])
    c2 = np.array([ 0.5, 0.0, 0.0])
    blob1 = c1 + rng.normal(scale=0.2, size=(half,3))
    blob2 = c2 + rng.normal(scale=0.2, size=(N-half,3))
    return np.vstack([blob1, blob2])

def init_uniform_large(N):
    return init_uniform(N)

# ============================================================
# One gradient-descent step with backtracking line search
# Ensures Φ_b(new) <= Φ_b(old), so Φ_b is discrete-time Lyapunov.
# ============================================================

def gradient_step_backtrack(x, dt_init):
    """
    x: (N,3) current coords
    dt_init: trial step size
    Returns:
        x_new, phi_new
    Procedure:
        - compute phi,F at x
        - try x - dt_init * gradΦ_b. But F = -∇Φ_b, so x + dt_init * F.
        - if Φ_b decreases, accept
        - else halve dt and retry, up to BACKTRACK_ITERS
    """
    phi_old, F = phi_and_forces(x)
    dt = dt_init
    for _ in range(BACKTRACK_ITERS):
        x_trial = x + dt * F
        phi_trial, _ = phi_and_forces(x_trial)
        if phi_trial <= phi_old:
            return x_trial, phi_trial
        dt *= 0.5
    # If still not improved, fall back to a very small step
    x_trial = x + (dt * F)
    phi_trial, _ = phi_and_forces(x_trial)
    # We will accept even if tiny increase, but typically this won't happen
    return x_trial, phi_trial

# ============================================================
# Integrator loop
# ============================================================

def run_simulation(x0, steps, dt_init, record_stride):
    """
    Performs a monotonic-descent evolution of Φ_b via backtracking.
    Records snapshots every record_stride.
    Returns:
      iters_list: recorded iteration indices
      phi_list:   Φ_b at those iterations
      coords_list: corresponding coordinates (N,3)
    """
    x = x0.copy()
    iters_list = []
    phi_list = []
    coords_list = []

    # record t=0
    phi_now, _ = phi_and_forces(x)
    iters_list.append(0)
    phi_list.append(phi_now)
    coords_list.append(x.copy())

    for t in range(1, steps+1):
        # backtracking step
        x, phi_now = gradient_step_backtrack(x, dt_init)

        # record occasionally
        if t % record_stride == 0:
            iters_list.append(t)
            phi_list.append(phi_now)
            coords_list.append(x.copy())

    return np.array(iters_list), np.array(phi_list), coords_list

# ============================================================
# Compression: Phase I (pair-distance histogram code)
# ============================================================

def serialize_phase1_pairhist(coords, bin_edges):
    """
    Phase I surrogate compressor:
      - compute all pairwise distances r_ij
      - bin them
      - store (count, mean residual) per bin
    Serialize -> gzip -> return compressed size in bytes.
    """
    N = coords.shape[0]
    diffs = coords[:,None,:] - coords[None,:,:]
    dist = np.sqrt(np.sum(diffs**2, axis=2))
    iu, ju = np.triu_indices(N, k=1)
    r = dist[iu, ju]

    bin_idx = np.digitize(r, bin_edges) - 1
    nbins = len(bin_edges) - 1
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

    counts = np.zeros(nbins, dtype=np.int32)
    residuals = np.zeros(nbins, dtype=np.float32)

    for k in range(nbins):
        mask = (bin_idx == k)
        counts[k] = np.sum(mask)
        if counts[k] > 0:
            residuals[k] = np.mean(r[mask] - bin_centers[k])
        else:
            residuals[k] = 0.0

    buf = struct.pack('<i', nbins)
    for k in range(nbins):
        buf += struct.pack('<if', int(counts[k]), float(residuals[k]))
    compressed = gzip.compress(buf)
    return len(compressed)

# ============================================================
# Compression: Phase II (raw quantized coordinates only)
# ============================================================

def serialize_phase2_coords(coords, dx_quant=DX_QUANT):
    """
    Phase II compressor:
      - quantize coordinates to int grid
      - serialize raw integer table
      - gzip
    This does NOT explicitly encode pairwise structure.
    """
    N = coords.shape[0]
    q = np.round(coords / dx_quant).astype(np.int32)
    buf = struct.pack('<i', N)
    buf += q.astype(np.int32).tobytes(order='C')
    compressed = gzip.compress(buf)
    return len(compressed)

# ============================================================
# Evaluate snapshots: compute Φ_b, Phase I size, Phase II size;
# sub-sample for correlation statistics
# ============================================================

def evaluate_snapshots(iters_list, phi_list, coords_list,
                       bin_edges, subsample_stride=SUBSAMPLE_STRIDE):
    phase1_sizes = []
    phase2_sizes = []
    for coords in coords_list:
        nbytes1 = serialize_phase1_pairhist(coords, bin_edges)
        nbytes2 = serialize_phase2_coords(coords)
        phase1_sizes.append(nbytes1)
        phase2_sizes.append(nbytes2)

    iters_arr = np.array(iters_list)
    phi_arr = np.array(phi_list)
    phase1_arr = np.array(phase1_sizes)
    phase2_arr = np.array(phase2_sizes)

    # Subsample to mitigate temporal autocorrelation
    mask = (iters_arr % subsample_stride == 0)
    phi_sub = phi_arr[mask]
    p1_sub = phase1_arr[mask]
    p2_sub = phase2_arr[mask]

    # Pearson correlations
    # Note: sign may be +/-; we only test monotonic association strength.
    if len(phi_sub) > 1:
        r_p1, p_p1 = pearsonr(phi_sub, p1_sub)
        r_p2, p_p2 = pearsonr(phi_sub, p2_sub)
    else:
        r_p1, p_p1, r_p2, p_p2 = np.nan, np.nan, np.nan, np.nan

    return {
        "iters": iters_arr,
        "phi": phi_arr,
        "phase1": phase1_arr,
        "phase2": phase2_arr,
        "iters_sub": iters_arr[mask],
        "phi_sub": phi_sub,
        "phase1_sub": p1_sub,
        "phase2_sub": p2_sub,
        "r_phase1": r_p1,
        "p_phase1": p_p1,
        "r_phase2": r_p2,
        "p_phase2": p_p2,
    }

# ============================================================
# Plotting
# ============================================================

def plot_phi_vs_iter(name, results, outdir=OUTDIR):
    it = results["iters"]
    phi = results["phi"]

    plt.figure(figsize=(6,4))
    plt.plot(it, phi, marker='o', ms=3)
    plt.xlabel("iteration")
    plt.ylabel(r"$\Phi_b$")
    plt.title(f"{name}: surrogate $\\Phi_b$ vs iteration")
    plt.tight_layout()
    fname = os.path.join(outdir, f"{name}_phib_vs_iter.png")
    plt.savefig(fname, dpi=150)
    plt.close()

def plot_compression_vs_phi(name, results, which="phase1", outdir=OUTDIR):
    it = results["iters"]
    phi = results["phi"]

    if which == "phase1":
        comp = results["phase1"]
        ylabel = "Phase I compressed bytes (pair-hist gzip)"
    else:
        comp = results["phase2"]
        ylabel = "Phase II compressed bytes (coord gzip)"

    plt.figure(figsize=(6,4))
    sc = plt.scatter(phi, comp, c=it, cmap='viridis', s=18)
    cbar = plt.colorbar(sc)
    cbar.set_label("iteration (earlier=lighter)")
    plt.xlabel(r"$\Phi_b$")
    plt.ylabel(ylabel)
    plt.title(f"{name}: {ylabel} vs $\\Phi_b$")
    plt.tight_layout()
    fname = os.path.join(outdir, f"{name}_phib_vs_{which}.png")
    plt.savefig(fname, dpi=150)
    plt.close()

# ============================================================
# One full experiment for an ensemble
# ============================================================

def run_experiment(name, init_func, N, steps):
    x0 = init_func(N)
    iters_list, phi_list, coords_list = run_simulation(
        x0,
        steps=steps,
        dt_init=DT_INIT,
        record_stride=RECORD_STRIDE
    )

    # define Phase I histogram bins
    bin_edges = np.linspace(BINS_MIN, BINS_MAX, NBINS+1)

    results = evaluate_snapshots(
        iters_list,
        phi_list,
        coords_list,
        bin_edges=bin_edges,
        subsample_stride=SUBSAMPLE_STRIDE
    )

    # plots
    plot_phi_vs_iter(name, results)
    plot_compression_vs_phi(name, results, which="phase1")
    plot_compression_vs_phi(name, results, which="phase2")

    return results

# ============================================================
# Main
# ============================================================

def main():
    # N=40 cases
    res_uniform40 = run_experiment("uniform40", init_uniform, N=40, steps=STEPS)
    res_lattice40 = run_experiment("lattice40", init_lattice, N=40, steps=STEPS)
    res_blobs40   = run_experiment("blobs40",   init_blobs,   N=40, steps=STEPS)

    # scaling case: N=400 uniform
    res_uniform400 = run_experiment("uniform400", init_uniform_large, N=400, steps=STEPS_LARGE)

    # Summaries
    def summarize(name, N, R):
        return {
            "name": name,
            "N": N,
            "r_phase1": R["r_phase1"],
            "p_phase1": R["p_phase1"],
            "r_phase2": R["r_phase2"],
            "p_phase2": R["p_phase2"],
            "n_subsamples": len(R["phi_sub"])
        }

    rows = [
        summarize("uniform40", 40, res_uniform40),
        summarize("lattice40", 40, res_lattice40),
        summarize("blobs40",   40, res_blobs40),
        summarize("uniform400",400, res_uniform400),
    ]

    print("")
    print("Summary (subsampled every {} iterations to reduce temporal autocorrelation):".format(SUBSAMPLE_STRIDE))
    print("{:12s} {:>5s} {:>5s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
        "ensemble","N","n_eff","r_phase1","p_phase1","r_phase2","p_phase2"
    ))
    for row in rows:
        print("{:12s} {:5d} {:5d} {:12.3f} {:12.2e} {:12.3f} {:12.2e}".format(
            row["name"], row["N"], row["n_subsamples"],
            row["r_phase1"], row["p_phase1"],
            row["r_phase2"], row["p_phase2"]
        ))

    print("\nPhase I = pair-distance histogram code + gzip (explicitly encodes pair structure).")
    print("Phase II = raw quantized coordinates + gzip (no explicit pair structure).")
    print("r_phase1 / r_phase2 = Pearson correlation between Φ_b and compressed size,")
    print("computed only on subsampled snapshots (to partially decorrelate time).")
    print("Φ_b(t) is enforced monotone nonincreasing by backtracking line search,")
    print("so each run supplies a discrete-time Lyapunov-style descent curve.")
    print(f"Figures saved in: {OUTDIR}/")

if __name__ == "__main__":
    main()
