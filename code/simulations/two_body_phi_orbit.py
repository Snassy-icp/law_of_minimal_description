#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
two_body_phi_orbit.py
Clean, honest 2-body Φ system that produces a recognisable orbit and r(t) plot.

- State: two points on [0,1)^2 given by midpoint (cx,cy), radius r, angle θ.
- Φ = -1/r  (MST for two points is just the single edge; lower is better)
- Proposals:
    * Rotation: θ ← θ + N(0, theta_std).   ΔΦ = 0 → always accepted
    * Radial:   r ← r*(1+δ) with small δ.  ΔΦ = -1/r' + 1/r; inward favoured by Metropolis

Outputs:
  orbit_two_body.png       (trajectory with time colour, arrows, inset r(t))
  two_body_r_vs_t.png      (radius over time)
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ---------------- core mechanics ----------------

def phi_two_body(r: float) -> float:
    return -1.0 / (r + 1e-12)

def run_two_body_orbit(
    steps: int = 15000,
    center = (0.5, 0.5),
    r0: float = 0.28,
    beta: float = 8.0,
    theta_std: float = 0.012,
    p_radial: float = 0.02,
    radial_scale: float = 0.002,
    seed: int = 11,
):
    """
    Drive the 2-body Φ process and return dict with:
      xi, xj   : (T,2) arrays of positions
      r_hist   : (T,) radius over time
      accept_stats : dict
      params   : dict
    """
    rng = np.random.default_rng(seed)
    cx, cy = center
    r = float(r0)
    theta = 0.0

    def pos_from_state(r, theta):
        # Two opposite points on a circle, wrapped to unit torus for visual consistency
        xi = np.array([cx + r*np.cos(theta),           cy + r*np.sin(theta)]) % 1.0
        xj = np.array([cx + r*np.cos(theta + np.pi),   cy + r*np.sin(theta + np.pi)]) % 1.0
        return xi, xj

    xi_list, xj_list, r_hist = [], [], []
    radial_accepts = 0
    radial_attempts = 0

    for _ in range(steps):
        if rng.random() < p_radial:
            # propose radius tweak (slight inward bias via mean < 0)
            radial_attempts += 1
            delta = rng.normal(loc=-radial_scale, scale=radial_scale)
            r_new = max(1e-3, min(0.45, r * (1.0 + delta)))
            dphi = phi_two_body(r_new) - phi_two_body(r)
            # Metropolis accept
            if dphi <= 0.0 or rng.random() < np.exp(-beta * dphi):
                r = r_new
                radial_accepts += 1
        else:
            # pure rotation (ΔΦ = 0)
            theta += rng.normal(0.0, theta_std)

        xi, xj = pos_from_state(r, theta)
        xi_list.append(xi)
        xj_list.append(xj)
        r_hist.append(r)

    return dict(
        xi=np.array(xi_list),
        xj=np.array(xj_list),
        r_hist=np.array(r_hist),
        accept_stats=dict(radial_accepts=radial_accepts, radial_attempts=radial_attempts),
        params=dict(steps=steps, center=center, r0=r0, beta=beta, theta_std=theta_std,
                    p_radial=p_radial, radial_scale=radial_scale, seed=seed),
    )

# --------------- plotting helpers ----------------

def _save(fig, outfile: str):
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)

def plot_two_body_tracks(
    xi, xj, r_hist=None,
    title="Two-body $\\Phi$ system: quasi-orbit with slow inspiral",
    outfile="paper/figures/orbit_two_body.png",
    auto_zoom=True, pad=0.08,
    arrows_every=25, arrow_scale=0.65,
    show_inset=True
):
    xi = np.asarray(xi)[:, :2]
    xj = np.asarray(xj)[:, :2]
    T  = len(xi)
    t  = np.linspace(0, 1, T)

    fig, ax = plt.subplots(figsize=(6.6, 6.2))

    # time-coloured polylines
    for k in range(T-1):
        ax.plot(xi[k:k+2,0], xi[k:k+2,1], color=plt.cm.viridis(t[k]), lw=1.8)
        ax.plot(xj[k:k+2,0], xj[k:k+2,1], color=plt.cm.plasma(t[k]),  lw=1.8)

    # start/end markers
    ax.scatter([xi[0,0]],[xi[0,1]], s=26, color="tab:blue")
    ax.scatter([xi[-1,0]],[xi[-1,1]], s=26, color="tab:orange")
    ax.scatter([xj[0,0]],[xj[0,1]], s=26, color="tab:blue")
    ax.scatter([xj[-1,0]],[xj[-1,1]], s=26, color="tab:orange")

    # arrows every k steps (visual only)
    if arrows_every and arrows_every > 0 and T > 1:
        idx = np.arange(0, T-1, arrows_every)
        # xi
        U = xi[idx+1,0] - xi[idx,0]; V = xi[idx+1,1] - xi[idx,1]
        ax.quiver(xi[idx,0], xi[idx,1], U, V, angles="xy", scale_units="xy",
                  scale=1.0/arrow_scale, width=0.002, alpha=0.5, color="tab:green")
        # xj
        U2 = xj[idx+1,0] - xj[idx,0]; V2 = xj[idx+1,1] - xj[idx,1]
        ax.quiver(xj[idx,0], xj[idx,1], U2, V2, angles="xy", scale_units="xy",
                  scale=1.0/arrow_scale, width=0.002, alpha=0.5, color="tab:red")

    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.25)

    if auto_zoom:
        XY = np.vstack([xi, xj])
        xmin, ymin = XY.min(0); xmax, ymax = XY.max(0)
        dx, dy = xmax-xmin, ymax-ymin
        padxy = pad * max(dx, dy, 1e-6)
        ax.set_xlim(xmin - padxy, xmax + padxy)
        ax.set_ylim(ymin - padxy, ymax + padxy)
    else:
        ax.set_xlim(0,1); ax.set_ylim(0,1)

    if show_inset and r_hist is not None and len(r_hist) > 1:
        axins = inset_axes(ax, width="38%", height="32%", loc="upper right", borderpad=1.2)
        axins.plot(np.asarray(r_hist), lw=1.6)
        axins.set_xticks([]); axins.set_yticks([])
        axins.set_title("$r(t)$", fontsize=10, pad=2)
        axins.grid(True, linestyle="--", alpha=0.2)

    _save(fig, outfile)

def plot_radius(r_hist, outfile="paper/figures/two_body_r_vs_t.png"):
    r_hist = np.asarray(r_hist)
    fig, ax = plt.subplots(figsize=(8.0,4.3))
    ax.plot(r_hist, lw=2)
    ax.set_title("Two-body radius over time (rare radial descent)")
    ax.set_xlabel("Step"); ax.set_ylabel(r"Radius $r(t)$")
    ax.grid(True, linestyle="--", alpha=0.25)
    _save(fig, outfile)

# --------------- CLI (optional) ----------------
if __name__ == "__main__":
    # Standalone test run (writes next to this file in ./figures/)
    outdir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(outdir, exist_ok=True)
    res = run_two_body_orbit()
    plot_two_body_tracks(res["xi"], res["xj"], r_hist=res["r_hist"],
                         outfile=os.path.join(outdir, "orbit_two_body.png"),
                         auto_zoom=True, pad=0.06, arrows_every=30, arrow_scale=0.55, show_inset=True)
    plot_radius(res["r_hist"], outfile=os.path.join(outdir, "two_body_r_vs_t.png"))
    print("Saved two-body figures into:", outdir)
