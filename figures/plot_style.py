# Unified plot style: consistent fonts, sizes, and defaults
import matplotlib as mpl

def use_paper_style():
    mpl.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
    })
