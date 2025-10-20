# The Law of Minimal Description

**A simple informational bias — toward shorter descriptions — reproduces gravity, orbits, inverse-square law behavior, and clustering without any force postulates.**

This project proposes a single physical principle:

> **The universe evolves toward states of shorter description.**  
> **ΔΦ ≤ 0** — *The Law of Minimal Description*

From this alone, using **description length minimization** (Kolmogorov complexity / MDL style), we show:

✅ Gravity emerges from **spatial compression gradients**  
✅ Newton’s inverse-square law arises from **informational flux conservation**  
✅ Binary orbits appear **without assuming force or momentum**  
✅ Many-body gravitational clustering occurs naturally  
✅ Quantum probability (Born rule) follows from **compression weighting**  
✅ Time and causality emerge from **temporal compression**  
✅ General Relativity follows from **second variations of Φ**

---

### 📄 Paper (PDF)

👉 **Full paper:**  
`paper/pdf/law_of_minimal_description.pdf`  
https://github.com/Snassy-icp/law_of_minimal_description/blob/main/paper/pdf/law_of_minimal_description.pdf

---

### 🧪 Reproducible Simulations

All simulations use **Φ-descent** via a Metropolis process minimizing **total description length** approximated by a **Minimum Spanning Tree (MST)** encoding cost.

Run figures (quick mode):

```bash
make figures-fast

## Reproducing paper figures

```bash
# 1) install
make setup

# 2) regenerate all figures deterministically
make figures        # or: make figures-fast

# Outputs in code/simulation/paper/figures/
# build_metadata.txt records the git hash and seed used.
