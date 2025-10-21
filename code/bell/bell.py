#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import random
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import matplotlib
matplotlib.use("Agg")  # use non-interactive backend

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # plotting optional


LOG10 = math.log10(2.0)  # 0.30102999566...


# ----------------------------- Utilities -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def h2(p: float) -> float:
    """Binary entropy (bits)."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    q = 1.0 - p
    return -(p * math.log2(p) + q * math.log2(q))


def ideal_bernoulli_codelen_bits(k: int, n: int) -> float:
    """
    Ideal (oracle) code length in bits for n Bernoulli trials with k ones:
    n * h2(k/n). Perfect for diagnostics like no-signalling.
    """
    if n <= 0:
        return 0.0
    return n * h2(k / n)


def kt_codelength_bits_bernoulli(bits: np.ndarray) -> float:
    """
    Krichevsky–Trofimov prequential codelength (bits) for a 0/1 sequence.
    """
    a = b = 0.5  # KT prior
    ones = 0
    L = 0.0
    for t, s in enumerate(bits, 1):
        p_next1 = (ones + a) / (t - 1 + a + b)
        p = p_next1 if s else (1.0 - p_next1)
        L += -math.log2(p)
        if s:
            ones += 1
    return L


def nosig_savings_bits_ideal(B: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
    """
    No-signalling diagnostic via IDEAL codelengths (oracle on the test data).
    Savings = L(B|Y) - L(B|X,Y). Expect ~0 for true no-signalling.
    """
    L_By = 0.0
    for y in (0, 1):
        mask_y = (Y == y)
        n_y = int(mask_y.sum())
        if n_y == 0:
            continue
        k_y = int(B[mask_y].sum())
        L_By += ideal_bernoulli_codelen_bits(k_y, n_y)

    L_Bxy = 0.0
    for x in (0, 1):
        for y in (0, 1):
            mask_xy = (X == x) & (Y == y)
            n_xy = int(mask_xy.sum())
            if n_xy == 0:
                continue
            k_xy = int(B[mask_xy].sum())
            L_Bxy += ideal_bernoulli_codelen_bits(k_xy, n_xy)

    return L_By - L_Bxy


def code_length_fixed_p(bits: np.ndarray, p_one: float) -> float:
    """
    Ideal fixed-p codelength for a Bernoulli sequence using true p (in bits).
    """
    # sum -log2( p if bit==1 else (1-p) )
    p1 = max(1e-12, min(1 - 1e-12, p_one))
    q1 = 1.0 - p1
    ones = int(bits.sum())
    zeros = bits.size - ones
    return -ones * math.log2(p1) - zeros * math.log2(q1)


# --------------------------- Data generation -------------------------

def balanced_settings(N: int) -> Tuple[np.ndarray, np.ndarray]:
    per = N // 4
    XY = [(0, 0)] * per + [(0, 1)] * per + [(1, 0)] * per + [(1, 1)] * per
    while len(XY) < N:
        XY.append((random.randint(0, 1), random.randint(0, 1)))
    random.shuffle(XY)
    X = np.fromiter((xy[0] for xy in XY), dtype=np.int8, count=N)
    Y = np.fromiter((xy[1] for xy in XY), dtype=np.int8, count=N)
    return X, Y


def random_settings(N: int) -> Tuple[np.ndarray, np.ndarray]:
    X = np.random.randint(0, 2, size=N, dtype=np.int8)
    Y = np.random.randint(0, 2, size=N, dtype=np.int8)
    return X, Y


@dataclass
class Dataset:
    X: np.ndarray  # settings
    Y: np.ndarray
    A: np.ndarray  # outcomes (0/1)
    B: np.ndarray
    s: np.ndarray  # score bit: A xor B xor (X&Y), success when s=0


def dataset_LHV(N: int, X: np.ndarray, Y: np.ndarray) -> Dataset:
    # Deterministic local strategy: A=0, B=0 -> wins unless X=Y=1
    A = np.zeros(N, dtype=np.int8)
    B = np.zeros(N, dtype=np.int8)
    s = A ^ B ^ (X & Y)
    return Dataset(X, Y, A, B, s)


def dataset_QUANTUM(N: int, X: np.ndarray, Y: np.ndarray, omega: float = math.cos(math.pi / 8) ** 2) -> Dataset:
    # Win (s=0) with probability omega, else s=1. Ensure unbiased marginals.
    win = np.random.random(size=N) < omega
    parity = (X & Y) ^ (~win).astype(np.int8)  # when lose, flip parity
    A = np.random.randint(0, 2, size=N, dtype=np.int8)
    B = A ^ parity
    s = A ^ B ^ (X & Y)
    return Dataset(X, Y, A, B, s)


def dataset_PR(N: int, X: np.ndarray, Y: np.ndarray) -> Dataset:
    # Perfect nonlocal box: A random; B = A xor (X&Y); s=0 always
    A = np.random.randint(0, 2, size=N, dtype=np.int8)
    B = A ^ (X & Y)
    s = A ^ B ^ (X & Y)
    return Dataset(X, Y, A, B, s)


# ----------------------------- Reporting -----------------------------

def mdL_summary(bits_s: np.ndarray, label: str, N: int) -> dict:
    """Compute savings for different MDL schemes on the score bits s."""
    # Train/test split
    n_train = N // 2
    s_train = bits_s[:n_train]
    s_test = bits_s[n_train:]

    p_hat = float(s_train.mean()) if n_train > 0 else 0.5
    L_test_model = code_length_fixed_p(s_test, p_hat)
    L_test_fair = len(s_test) * 1.0
    sav_train_test = L_test_fair - L_test_model  # bits

    # KT universal
    L_kt = kt_codelength_bits_bernoulli(bits_s)
    L_fair = N * 1.0
    sav_kt = L_fair - L_kt

    # Fixed-p theory baselines
    p_lhv = 0.25  # q_fail at CHSH LHV bound
    p_q = 1.0 - (math.cos(math.pi / 8) ** 2)  # Tsirelson failure prob
    p_pr = 0.0

    L_fix_lhv = code_length_fixed_p(bits_s, p_lhv)
    L_fix_q = code_length_fixed_p(bits_s, p_q)
    L_fix_pr = code_length_fixed_p(bits_s, p_pr)

    return {
        "p_hat": p_hat,
        "sav_train_test": sav_train_test,
        "sav_kt": sav_kt,
        "L_fix_lhv": L_fix_lhv,
        "L_fix_q": L_fix_q,
        "L_fix_pr": L_fix_pr,
        "L_fair": L_fair,
        "N": N,
    }


def evidence_print(name1: str, L1: float, name2: str, L2: float):
    delta = L1 - L2  # bits
    log10_odds = -delta * LOG10
    print(f"  MDL evidence ({name1} vs {name2}): ΔL={delta:+,} bits; log10(odds)={log10_odds:.2f}")


def report_mode(mode: str, X: np.ndarray, Y: np.ndarray) -> Tuple[Dataset, dict]:
    N = len(X)

    if mode == "LHV":
        D = dataset_LHV(N, X, Y)
    elif mode == "QUANTUM":
        D = dataset_QUANTUM(N, X, Y)
    elif mode == "PR":
        D = dataset_PR(N, X, Y)
    else:
        raise ValueError("Unknown mode")

    s = D.s.astype(np.int8)
    omega = 1.0 - float(s.mean())
    S = 8.0 * omega - 4.0  # standard CHSH S
    q_fail = 1.0 - omega
    S_std = 8.0*omega - 4.0

    res = mdL_summary(s, mode, N)

    # Print header
    print("-" * 90)
    print(f"{mode:<8}  ω={omega:.6f}  S={S_std:.3f}  q_fail(S)={1.0-omega:.6f}")
    print(f"  Train/Test MDL:    savings={res['sav_train_test']:10.0f} bits  (p_one(train)={res['p_hat']:.6f})")
    print(f"  KT (universal):    savings={res['sav_kt']:10.0f} bits")
    print(f"  Fixed-p (LHV):     savings={res['L_fair']-res['L_fix_lhv']:10.0f} bits  [p_one=0.25]")
    print(f"  Fixed-p (Quantum): savings={res['L_fair']-res['L_fix_q']:10.0f} bits  [p_one≈0.146447]")
    print(f"  Fixed-p (PR):      savings={res['L_fair']-res['L_fix_pr']:10.0f} bits  [p_one=0.0]")

    # Evidence comparisons using fixed-p baselines (clean, closed-form)
    evidence_print("Quantum", res["L_fix_q"], "LHV", res["L_fix_lhv"])
    evidence_print("PR", res["L_fix_pr"], "LHV", res["L_fix_lhv"])
    evidence_print("PR", res["L_fix_pr"], "Quantum", res["L_fix_q"])

    # No-signalling diagnostics (ideal code lengths)
    savA = nosig_savings_bits_ideal(D.A, D.X, D.Y)
    savB = nosig_savings_bits_ideal(D.B, D.X, D.Y)
    print(f"  No-signalling (A): savings≈{savA:8.0f} bits  ({savA/N:.2e} bits/trial)  ~0 expected")
    print(f"  No-signalling (B): savings≈{savB:8.0f} bits  ({savB/N:.2e} bits/trial)  ~0 expected")

    return D, res


# ------------------------------- Plots --------------------------------

def _safe_savefig(title: str):
    import re
    fname = re.sub(r'[^A-Za-z0-9_.-]+', '_', title)
    plt.savefig(f"{fname}.png", dpi=200)
    plt.close()


def plot_savings_bar(results: List[Tuple[str, dict]], title: str = "Bell/CHSH via MDL — Savings per Mode"):
    if plt is None:
        return
    modes = [name for name, _ in results]
    train = [res["sav_train_test"] for _, res in results]
    kt = [res["sav_kt"] for _, res in results]
    best = []
    for _, res in results:
        best.append(max(res["L_fair"] - res["L_fix_lhv"],
                        res["L_fair"] - res["L_fix_q"],
                        res["L_fair"] - res["L_fix_pr"]))

    x = np.arange(len(modes))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w, train, width=w, label="Train/Test MDL")
    ax.bar(x, kt, width=w, label="KT MDL")
    ax.bar(x + w, best, width=w, label="Best Fixed-p MDL")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Savings (bits)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _safe_savefig(title)

def plot_convergence(prefix_savings_per_mode: List[Tuple[str, np.ndarray]]):
    if plt is None:
        return
    title = "KT savings per trial vs N"
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, cum_sav in prefix_savings_per_mode:
        n = np.arange(1, len(cum_sav) + 1)
        ax.plot(n, cum_sav / n, label=name)  # bits per trial
    ax.set_xlabel("Trials")
    ax.set_ylabel("KT savings per trial (bits)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _safe_savefig(title)


def kt_savings_prefix(bits_s: np.ndarray) -> np.ndarray:
    """Return cumulative KT savings array for s (in bits)."""
    a = b = 0.5
    ones = 0
    L_kt = 0.0
    L_fair = 0.0
    n = len(bits_s)
    out = np.zeros(n, dtype=float)
    for t, s in enumerate(bits_s, 1):
        p_next1 = (ones + a) / (t - 1 + a + b)
        L_kt += -math.log2(p_next1 if s else (1.0 - p_next1))
        L_fair += 1.0
        out[t - 1] = L_fair - L_kt
        if s:
            ones += 1
    return out


def plot_nosig_residuals(modes_savA: List[Tuple[str, float]], modes_savB: List[Tuple[str, float]], N: int):
    if plt is None:
        return
    title = "No-signalling residuals (expect ~0)"
    labels = [name for name, _ in modes_savA]
    valsA = [sav / N for _, sav in modes_savA]
    valsB = [sav / N for _, sav in modes_savB]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, valsA, width=w, label="A")
    ax.bar(x + w / 2, valsB, width=w, label="B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("No-signalling residual (bits/trial)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _safe_savefig(title)



# -------------------------------- Main --------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bell/CHSH via MDL")
    parser.add_argument("--N", type=int, default=200_000, help="Number of trials")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--plot", action="store_true", help="Plot savings bar chart")
    parser.add_argument("--plot_convergence", action="store_true", help="Plot KT savings per trial vs N")
    parser.add_argument("--plot_residuals", action="store_true", help="Plot no-signalling residuals (bits/trial)")
    parser.add_argument("--balance_settings", action="store_true", help="Force exactly N/4 per (X,Y) pair")
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Settings: N={args.N:,}  seed={args.seed}  "
          f"plot={args.plot}  plot_convergence={args.plot_convergence}  "
          f"plot_residuals={args.plot_residuals}  balance_settings={args.balance_settings}")

    # Settings
    if args.balance_settings:
        X, Y = balanced_settings(args.N)
    else:
        X, Y = random_settings(args.N)

    results = []
    datasets = {}

    for mode in ["LHV", "QUANTUM", "PR"]:
        D, res = report_mode(mode, X, Y)
        results.append((mode, res))
        datasets[mode] = D

    if args.plot:
        plot_savings_bar(results)

    if args.plot_convergence and plt is not None:
        pref = []
        for mode in ["LHV", "QUANTUM", "PR"]:
            s = datasets[mode].s.astype(np.int8)
            pref.append((mode, kt_savings_prefix(s)))
        plot_convergence(pref)

    if args.plot_residuals and plt is not None:
        modesA, modesB = [], []
        for mode in ["LHV", "QUANTUM", "PR"]:
            D = datasets[mode]
            modesA.append((mode, nosig_savings_bits_ideal(D.A, D.X, D.Y)))
            modesB.append((mode, nosig_savings_bits_ideal(D.B, D.X, D.Y)))
        plot_nosig_residuals(modesA, modesB, args.N)


if __name__ == "__main__":
    main()
