"""Presentation helpers shared by every experiment.

Keeps two concerns out of the physics modules and the notebooks:

* turning ``uncertainties``/``unumpy`` values into human-readable tables;
* a single, consistent Matplotlib look for every figure.
"""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from uncertainties import unumpy as unp

# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def format_uarray(values: Iterable, fmt: str = ".2u") -> list[str]:
    """Render an iterable of ``ufloat`` values as ``"nominal+/-std"`` strings.

    ``fmt`` follows the ``uncertainties`` mini-language; ``".2u"`` keeps two
    significant digits on the uncertainty (the physics convention).
    """
    return [f"{value:{fmt}}" for value in values]


def uframe(columns: dict[str, Iterable], fmt: str = ".2u", **fixed) -> pd.DataFrame:
    """Build a display ``DataFrame`` from named columns of ``ufloat`` arrays.

    Columns whose values are plain scalars (via ``fixed``) are added verbatim,
    which is handy for identifiers such as a droplet index or an applied voltage.
    """
    data = {name: format_uarray(values, fmt) for name, values in columns.items()}
    data.update(fixed)
    return pd.DataFrame(data)


def split_uarray(values) -> tuple:
    """Return the ``(nominal_values, std_devs)`` pair of a ``unumpy`` array."""
    return unp.nominal_values(values), unp.std_devs(values)


def uabs(values):
    """Absolute value of a ``unumpy`` array without the deprecated ``__abs__``.

    Flips the sign of negative-nominal entries; the (symmetric) uncertainty is
    unchanged, so multiplying by ``+/-1`` reproduces ``abs`` exactly.
    """
    import numpy as np

    return values * np.sign(unp.nominal_values(values))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

PALETTE = {
    "data": "#2563EB",       # blue  -- measured / derived points
    "fit": "#DC2626",        # red   -- fitted models
    "reference": "#059669",  # green -- literature / reference values
    "grid": "#D1D5DB",
    "text": "#1F2937",
}


def apply_style() -> None:
    """Apply the shared figure style (replaces the old ``seaborn-whitegrid``)."""
    plt.rcParams.update(
        {
            "figure.figsize": (7.5, 5.0),
            "figure.dpi": 110,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.edgecolor": PALETTE["text"],
            "axes.labelcolor": PALETTE["text"],
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.8,
            "text.color": PALETTE["text"],
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "font.size": 11,
        }
    )


def errorbar_scatter(ax, x, y, *, label=None, color=None):
    """Scatter ``ufloat`` arrays with error bars taken from their uncertainties."""
    xn, xs = split_uarray(x)
    yn, ys = split_uarray(y)
    ax.errorbar(
        xn, yn, xerr=xs, yerr=ys,
        fmt="o", markersize=5, capsize=3, elinewidth=1,
        color=color or PALETTE["data"], ecolor=PALETTE["grid"],
        markeredgecolor="white", markeredgewidth=0.6, label=label,
    )
    return ax
