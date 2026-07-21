"""Figures for each experiment, sharing one Matplotlib style.

Every function takes a result object (from the experiment's ``run()``) and
returns a Matplotlib ``Figure``, so the same code renders inline in a notebook
and is saved to PNG for the HTML summary.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from uncertainties import unumpy as unp

from .display import PALETTE, apply_style, errorbar_scatter, split_uarray
from . import millikan as _millikan


def _line_from_fit(ax, x_nominal, slope, intercept, label):
    """Draw a fitted straight line across the current x-range."""
    xs = np.linspace(x_nominal.min(), x_nominal.max(), 100)
    ax.plot(xs, slope.n * xs + intercept.n, "-", lw=2,
            color=PALETTE["fit"], label=label)


def millikan_charge_quantization(result: _millikan.MillikanResult):
    """Sorted droplet charges with the five plateaus and integer-multiple lines."""
    apply_style()
    fig, ax = plt.subplots()

    charge = result.table["charge_C"].to_numpy()
    order = np.argsort(unp.nominal_values(charge))
    ordered = charge[order]
    idx = np.arange(1, len(ordered) + 1)
    errorbar_scatter(ax, unp.uarray(idx, 0), ordered, label="Droplet charge")

    e = result.elementary_charge.n
    for n in range(1, 5):
        ax.axhline(n * e, ls="--", lw=1.2, color=PALETTE["reference"],
                   label="n·e (estimated)" if n == 1 else None)
        ax.annotate(f"{n}e", (len(ordered) + 0.1, n * e), va="center",
                    fontsize=9, color=PALETTE["reference"])

    ax.set_xlabel("Droplet (sorted by charge)")
    ax.set_ylabel("Electric charge (C)")
    ax.set_title("Millikan: charge quantization")
    ax.legend(loc="upper left")
    ax.set_xlim(0.5, len(ordered) + 1)
    fig.tight_layout()
    return fig


def millikan_charge_vs_radius(result: _millikan.MillikanResult):
    """Charge against Stokes radius (the raw physical scatter)."""
    apply_style()
    fig, ax = plt.subplots()
    errorbar_scatter(ax, result.table["radius_m"].to_numpy(),
                     result.table["charge_C"].to_numpy(), label="Droplets")
    ax.set_xlabel("Droplet radius (m)")
    ax.set_ylabel("Electric charge (C)")
    ax.set_title("Millikan: charge vs. radius")
    fig.tight_layout()
    return fig


def stefan_boltzmann_fit(result):
    """log(U_thermopile) vs log(T) with the fitted Stefan-Boltzmann exponent."""
    apply_style()
    fig, ax = plt.subplots()

    log_t = result.table["log_T"].to_numpy()
    log_u = result.table["log_U_term"].to_numpy()
    errorbar_scatter(ax, log_t, log_u, label="Measurements")

    tn, _ = split_uarray(log_t)
    _line_from_fit(ax, tn, result.exponent, result.intercept,
                   f"Fit: slope = {result.exponent:.2u}")

    ax.set_xlabel("log T  (T in K)")
    ax.set_ylabel("log U$_{thermopile}$  (V)")
    ax.set_title("Stefan-Boltzmann: radiated power vs. temperature")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def photoelectric_fit(result):
    """Stopping voltage vs frequency with the fitted h/e slope."""
    apply_style()
    fig, ax = plt.subplots()

    nu = result.table["frequency_Hz"].to_numpy()
    u0 = result.table["stopping_voltage_V"].to_numpy()
    errorbar_scatter(ax, nu, u0, label="Measurements")

    nn, _ = split_uarray(nu)
    _line_from_fit(ax, nn, result.slope, -result.work_function_eV,
                   f"Fit: h/e = {result.slope:.2u} V·s")

    ax.set_xlabel("Frequency ν (Hz)")
    ax.set_ylabel("Stopping voltage U$_0$ (V)")
    ax.set_title("Photoelectric effect: stopping voltage vs. frequency")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig
