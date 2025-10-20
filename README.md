## Reproducing paper figures

```bash
# 1) install
make setup

# 2) regenerate all figures deterministically
make figures        # or: make figures-fast

# Outputs in paper/figures/
# build_metadata.txt records the git hash and seed used.
