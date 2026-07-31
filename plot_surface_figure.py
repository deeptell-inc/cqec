"""Generate fig_surface.pdf for the manuscript: surface code vs LearnCI."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "blind_cqec_pkg"))

from run_surface_stim import (
    surface_pL_codecap, surface_pL_circuit, per_qubit_depolarizing, learnci_fidelity,
)
from algorithms_faithful import make_regev_faithful
from blind_cqec import fidelity
from blind_cqec.nqubit_learning import pauli_matrix

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "Lnl", "figures")
plt.rcParams.update({"font.size": 10})


def main():
    rho, _ = make_regev_faithful(); n = 6
    Ps = np.stack([pauli_matrix(b, n) for b in range(4 ** n)])

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    for ax, title, pL_fn, ps in [
        (axes[0], "code-capacity (single-shot, threshold ~11%)", surface_pL_codecap,
         [0.02, 0.05, 0.08, 0.11, 0.15]),
        (axes[1], "circuit-level (FT, d rounds, threshold ~0.6%)", surface_pL_circuit,
         [0.002, 0.005, 0.01, 0.02, 0.05]),
    ]:
        learn = [np.mean([learnci_fidelity(rho, n, p, 16000, Ps, np.random.default_rng(k))
                          for k in range(5)]) for p in ps]
        for dd, mk in [(3, "o-"), (5, "s-"), (7, "^-")]:
            fs = [fidelity(per_qubit_depolarizing(rho, n, pL_fn(dd, p)), rho) for p in ps]
            ax.plot(ps, fs, mk, ms=4, label=f"surface d={dd}")
        ax.plot(ps, learn, "k*--", ms=8, label="LearnCI")
        ax.set_xscale("log"); ax.set_xlabel("physical error rate $p$")
        ax.set_ylabel("logical fidelity"); ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
        # explicit ticks at the data points; suppress crowded log minor labels
        ax.xaxis.set_major_locator(FixedLocator(ps))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    fig.suptitle("Surface code baseline vs LearnCI (Regev state, $n=6$)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_surface.pdf")
    fig.savefig(out); print("wrote", out)


if __name__ == "__main__":
    main()
