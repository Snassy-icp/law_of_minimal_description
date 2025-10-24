import numpy as np
import itertools
import zlib
import struct
import matplotlib.pyplot as plt

# ============================================================
# Helper: generate datasets
# ============================================================

def gen_uniform(N, box=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(low=-box, high=box, size=(N, 3))
    return x

def gen_blobs(N, offset=0.5, spread=0.1, seed=1):
    rng = np.random.default_rng(seed)
    half = N // 2
    c1 = rng.normal(loc=-offset, scale=spread, size=(half, 3))
    c2 = rng.normal(loc=+offset, scale=spread, size=(N - half, 3))
    x = np.vstack([c1, c2])
    return x

def gen_lattice(N, spacing=0.3, jitter=0.01, seed=2):
    # Build roughly cubic lattice near origin
    rng = np.random.default_rng(seed)
    # pick k such that k^3 ~ N
    k = int(round(N ** (1/3)))
    if k**3 < N:
        k += 1
    coords = []
    for ix in range(k):
        for iy in range(k):
            for iz in range(k):
                coords.append([ (ix - k/2)*spacing,
                                (iy - k/2)*spacing,
                                (iz - k/2)*spacing ])
    coords = np.array(coords[:N])
    x = coords + rng.normal(scale=jitter, size=coords.shape)
    return x

# ============================================================
# Surrogate functional Phi_b and its gradient
# Phi_b = sum_{i<j} m_i m_j / sqrt(||x_i - x_j||^2 + a^2)
# ============================================================

def compute_phi_and_grad(x, m, a):
    """
    x: (N,3)
    m: (N,)
    a: scalar softening
    returns:
      phi: scalar
      grad: (N,3), i-th row = d/dx_i Phi_b
    NOTE: grad here is +dPhi/dx, so the force-like descent direction
    is -grad.
    """
    N = x.shape[0]
    grad = np.zeros_like(x)
    phi = 0.0
    # double loop O(N^2); OK for small N
    for i in range(N):
        for j in range(i+1, N):
            rij = x[i] - x[j]
            r2 = np.dot(rij, rij)
            denom = np.sqrt(r2 + a*a)
            # contribution to phi
            phi += m[i]*m[j] / denom
            # d/dx_i of 1/sqrt(r^2+a^2) = -(rij)/(r^2+a^2)^(3/2)
            # so grad_i Phi_b += m_i m_j * [ -(rij)/(...^(3/2)) ]
            factor = m[i]*m[j] * (1.0 / ( (r2+a*a)**(1.5) ))
            grad_i = - factor * rij
            # and grad_j = -grad_i
            grad[i] += grad_i
            grad[j] -= grad_i
    return phi, grad

# ============================================================
# Quantization + compression proxy
# ============================================================

def quantize_positions(x, grid_dx):
    """
    Snap positions to a finite grid to reflect finite precision b bits.
    """
    return np.round(x / grid_dx) * grid_dx

def pairwise_distances(x):
    """
    Return array of all pairwise distances r_ij for i<j
    """
    N = x.shape[0]
    dists = []
    for i in range(N):
        for j in range(i+1, N):
            rij = x[i] - x[j]
            r = np.linalg.norm(rij)
            dists.append(r)
    return np.array(dists, dtype=np.float64)

def serialize_histogram_for_compression(dists, bin_edges):
    """
    This is our 'reference coding scheme for evaluation':
    - We bin all pairwise distances into fixed, preregistered bins.
    - We record histogram counts.
    - We also record mean residual within each bin (coarse).
    - We serialize these into a byte buffer deterministically.
    Then we call zlib.compress as a stand-in for LZ77-like coding.
    """
    # fixed bins
    hist_counts, _ = np.histogram(dists, bins=bin_edges)
    # residuals: difference between each distance and bin center, averaged per bin
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    residual_sums = np.zeros_like(bin_centers)
    residual_nums = np.zeros_like(bin_centers)
    # assign each dist to bin
    inds = np.digitize(dists, bin_edges) - 1
    for r, b in zip(dists, inds):
        if 0 <= b < len(bin_centers):
            residual_sums[b] += (r - bin_centers[b])
            residual_nums[b] += 1
    # average residual per bin (0 if empty)
    with np.errstate(divide='ignore', invalid='ignore'):
        residual_avgs = np.where(residual_nums>0,
                                 residual_sums/np.maximum(residual_nums,1),
                                 0.0)

    # serialize deterministically:
    # [num_bins][(count_0,resid_0),(count_1,resid_1),...]
    # We'll pack as 64-bit ints for counts, 64-bit floats for residuals.
    buf = bytearray()
    nbins = len(hist_counts)
    buf.extend(struct.pack("<I", nbins))  # 4-byte unsigned int
    for c, ravg in zip(hist_counts, residual_avgs):
        buf.extend(struct.pack("<Q", int(c)))      # 8-byte unsigned long long
        buf.extend(struct.pack("<d", float(ravg))) # 8-byte double

    # compress via zlib as a rough universal code proxy
    compressed = zlib.compress(bytes(buf), level=9)
    return len(compressed)

# ============================================================
# Gradient descent runner for PCD
# ============================================================

def run_pcd_flow(
        x0,
        m,
        a=0.05,
        grid_dx=0.01,
        dt=0.01,
        steps=200,
        bin_edges=None,
        record_every=5,
        ):
    """
    Run explicit Euler on dot x = -grad Phi_b, where grad Phi_b
    is from compute_phi_and_grad.
    Track over time:
      - Phi_b
      - compressed size (bytes)
    Returns dict with trajectories, times, phis, comp_sizes.
    """
    x = x0.copy()
    N = x.shape[0]

    times = []
    phis = []
    comp_sizes = []
    snapshots = []  # store a few x's if we want

    for t in range(steps):
        # compute phi and grad at current x
        phi, grad = compute_phi_and_grad(x, m, a=a)

        # explicit Euler step: x <- x - dt * grad
        x = x - dt * grad

        # record occasionally
        if t % record_every == 0 or t == steps-1:
            xq = quantize_positions(x, grid_dx=grid_dx)
            dists = pairwise_distances(xq)
            # If no bin_edges given, create once based on initial scale
            if bin_edges is None:
                raise ValueError("Please supply preregistered bin_edges.")

            comp_len = serialize_histogram_for_compression(dists, bin_edges)

            times.append(t)
            phis.append(phi)
            comp_sizes.append(comp_len)
            snapshots.append(x.copy())

    return {
        "times": np.array(times),
        "phis": np.array(phis),
        "comp": np.array(comp_sizes, dtype=np.float64),
        "snaps": snapshots,
    }

# ============================================================
# Plot helpers
# ============================================================

def make_plots(tag, results):
    """
    Make:
    1) phi vs iteration
    2) phi vs compressed size (scatter)
    Saves PNG files with prefix `tag`.
    """
    times = results["times"]
    phis  = results["phis"]
    comp  = results["comp"]

    # Plot 1: phi over time
    plt.figure()
    plt.plot(times, phis, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel(r"$\Phi_b$")
    plt.title(f"{tag}: surrogate objective over time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{tag}_phib_vs_iter.png", dpi=200)
    plt.close()

    # Plot 2: phi vs compressed size
    plt.figure()
    plt.scatter(phis, comp)
    plt.xlabel(r"$\Phi_b$")
    plt.ylabel("compressed size (bytes)")
    plt.title(f"{tag}: surrogate vs gzip size")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{tag}_phib_vs_compressed.png", dpi=200)
    plt.close()

# ============================================================
# Main experiment
# ============================================================

if __name__ == "__main__":
    # Experiment parameters (this is effectively our preregistration block)
    N = 40              # number of particles
    a = 0.05            # softening radius
    grid_dx = 0.01      # quantization step for coordinates
    dt = 0.01           # Euler step size
    steps = 400         # number of iterations
    record_every = 5    # record cadence
    m = np.ones(N)      # all masses 1 for now

    # preregistered histogram bins for distances:
    # we assume distances up to ~2.0 units is enough for these synthetic sets
    max_r = 2.0
    nbins = 32
    bin_edges = np.linspace(0.0, max_r, nbins+1)

    datasets = {
        "uniform": gen_uniform(N, box=1.0, seed=0),
        "blobs":   gen_blobs(N, offset=0.5, spread=0.1, seed=1),
        "lattice": gen_lattice(N, spacing=0.3, jitter=0.01, seed=2),
    }

    for tag, x0 in datasets.items():
        results = run_pcd_flow(
            x0=x0,
            m=m,
            a=a,
            grid_dx=grid_dx,
            dt=dt,
            steps=steps,
            bin_edges=bin_edges,
            record_every=record_every,
        )
        make_plots(tag, results)

        # For convenience, print rough correlation between Phi_b and compressed size
        from scipy.stats import pearsonr
        r, p = pearsonr(results["phis"], results["comp"])
        print(f"{tag}: correlation(Phi_b, compressed_size) = {r:.3f} (p={p:.2e})")
