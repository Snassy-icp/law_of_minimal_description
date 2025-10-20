#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build all paper figures deterministically from code files by PATH (no packages needed).

Usage:
  python -m figures.make_figures
  # or
  python figures/make_figures.py
"""

from __future__ import annotations
import argparse, json, time, subprocess, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless

ROOT = Path(__file__).resolve().parents[1]          # repo root
SIMDIR = ROOT / "code" / "simulations"              # path to .py sims (no package needed)
FIGDIR = ROOT / "paper" / "figures"                 # write ONLY the paper figs here
FIGDIR.mkdir(parents=True, exist_ok=True)

def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"

def load_module(path: Path):
    """Import a .py module by filesystem path (no packages required)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, f"Cannot load module from {path}"
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def ensure_saved(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Expected figure not found: {path}")

def build_many_body(args):
    # load phi_descent_min.py by path
    phi = load_module(SIMDIR / "phi_descent_min.py")

    # Find a simulation runner that exists
    if hasattr(phi, "run_simulation"):
        run_simulation = phi.run_simulation
    elif hasattr(phi, "run"):  # fallback if your file used a different name
        run_simulation = phi.run
    else:
        raise ImportError("Neither run_simulation nor run() found in phi_descent_min.py")

    # pick plotting helpers (raise if missing)
    plot_clustering      = getattr(phi, "plot_clustering")
    plot_mean_distance   = getattr(phi, "plot_mean_distance")
    plot_inverse_square  = getattr(phi, "plot_inverse_square")

    # run sim
    steps = 3000 if args.fast else 12000
    sim = run_simulation(N=120, steps=steps, beta=10.0, step=0.02,
                         seed=42, sample_every=40, progress=True)

    # make figs to paper/figures (final paper assets)
    out = FIGDIR / "clustering.png"
    plot_clustering(sim, outfile=str(out)); ensure_saved(out)

    out = FIGDIR / "mean_distance.png"
    plot_mean_distance(sim, outfile=str(out)); ensure_saved(out)

    out = FIGDIR / "inverse_square.png"
    plot_inverse_square(sim, outfile=str(out)); ensure_saved(out)

def build_two_body(args):
    tb = load_module(SIMDIR / "two_body_phi_orbit.py")

    # name is run_two_body_orbit in our provided file; fall back if renamed
    if hasattr(tb, "run_two_body_orbit"):
        run_two_body_orbit = tb.run_two_body_orbit
    elif hasattr(tb, "simulate_two_body"):
        run_two_body_orbit = tb.simulate_two_body
    else:
        raise ImportError("two_body_phi_orbit.py must define run_two_body_orbit() or simulate_two_body()")

    plot_two_body_tracks = getattr(tb, "plot_two_body_tracks")
    plot_radius          = getattr(tb, "plot_radius")

    res = run_two_body_orbit(
        steps = 8000 if args.fast else 15000,
        r0=0.28, beta=8.0, theta_std=0.012,
        p_radial=0.02, radial_scale=0.002, seed=18
    )

    out = FIGDIR / "orbit_two_body.png"
    plot_two_body_tracks(res["xi"], res["xj"], r_hist=res.get("r_hist"),
                         outfile=str(out), auto_zoom=True, pad=0.06,
                         arrows_every=30, arrow_scale=0.55, show_inset=True)
    ensure_saved(out)

    out = FIGDIR / "two_body_r_vs_t.png"
    plot_radius(res["r_hist"], outfile=str(out)); ensure_saved(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    build_many_body(args)
    build_two_body(args)
    (FIGDIR / "build_metadata.txt").write_text(
        json.dumps({
            "git": git_hash(),
            "fast": args.fast,
            "unix_time": int(time.time()),
        }, indent=2)
    )
    print(f"[OK] Figures built into {FIGDIR} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
