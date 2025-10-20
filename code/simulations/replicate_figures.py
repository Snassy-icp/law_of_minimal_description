#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, time, traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless

BASE = Path(__file__).parent.resolve()
OUT  = BASE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

def stamp(msg): print(f"[replicate] {msg}", flush=True)

# import modules by path, no packages needed
def load_module(path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def ensure_saved(p: Path):
    if not p.exists() or p.stat().st_size == 0:
        raise RuntimeError(f"Expected figure not found: {p}")

def many_body(fast: bool):
    phi = load_module(BASE / "phi_descent_min.py")

    # resolve runner
    if hasattr(phi, "run_simulation"):
        run_simulation = phi.run_simulation
    elif hasattr(phi, "run"):
        run_simulation = phi.run
    else:
        raise ImportError("phi_descent_min.py must define run_simulation() or run()")

    plot_clustering     = getattr(phi, "plot_clustering")
    plot_mean_distance  = getattr(phi, "plot_mean_distance")
    plot_inverse_square = getattr(phi, "plot_inverse_square")
    plot_orbit_many_body= getattr(phi, "plot_orbit_many_body", None)

    stamp("Many-body: running")
    steps = 2500 if fast else 8000
    sim = run_simulation(N=120, steps=steps, beta=10.0, step=0.02,
                         seed=42, sample_every=40, progress=True)

    p = OUT / "clustering.png"
    plot_clustering(sim, outfile=str(p)); ensure_saved(p); stamp(f"Saved {p}")

    p = OUT / "mean_distance.png"
    plot_mean_distance(sim, outfile=str(p)); ensure_saved(p); stamp(f"Saved {p}")

    p = OUT / "inverse_square.png"
    plot_inverse_square(sim, outfile=str(p)); ensure_saved(p); stamp(f"Saved {p}")

    if plot_orbit_many_body:
        p = OUT / "orbit_many_body.png"
        plot_orbit_many_body(outfile=str(p),
                             N_bg=20, pre_anneal=2000, steps=50000,
                             beta=2.8, step_single=0.005,
                             p_rotate=0.04, p_translate=0.94,
                             theta_std=0.0015, trans_std=0.028,
                             seed=42)
        ensure_saved(p); stamp(f"Saved {p}")

def two_body(fast: bool):
    tb = load_module(BASE / "two_body_phi_orbit.py")

    if hasattr(tb, "run_two_body_orbit"):
        run_two_body_orbit = tb.run_two_body_orbit
    elif hasattr(tb, "simulate_two_body"):
        run_two_body_orbit = tb.simulate_two_body
    else:
        raise ImportError("two_body_phi_orbit.py must define run_two_body_orbit() or simulate_two_body()")

    plot_two_body_tracks = getattr(tb, "plot_two_body_tracks")
    plot_radius          = getattr(tb, "plot_radius")

    stamp("Two-body: running")
    steps = 6000 if fast else 15000
    res = run_two_body_orbit(steps=steps, r0=0.28, beta=8.0, theta_std=0.012,
                             p_radial=0.02, radial_scale=0.002, seed=11)

    p = OUT / "orbit_two_body.png"
    plot_two_body_tracks(res["xi"], res["xj"], r_hist=res.get("r_hist"),
                         outfile=str(p), auto_zoom=True, pad=0.06,
                         arrows_every=30, arrow_scale=0.55, show_inset=True)
    ensure_saved(p); stamp(f"Saved {p}")

    p = OUT / "two_body_r_vs_t.png"
    plot_radius(res["r_hist"], outfile=str(p))
    ensure_saved(p); stamp(f"Saved {p}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    stamp(f"Output dir: {OUT}")
    t0 = time.time()
    try:
        many_body(args.fast)
        two_body(args.fast)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
    (OUT / "build_metadata.txt").write_text(f"built: {time.strftime('%Y-%m-%d %H:%M:%S')}\nfast={args.fast}\n")
    stamp(f"Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
