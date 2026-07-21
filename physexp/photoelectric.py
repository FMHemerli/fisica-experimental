"""Photoelectric effect: Planck's constant from the stopping-voltage slope.

A grating selects a wavelength from the diffraction angle; each wavelength has a
frequency and a measured stopping voltage. Einstein's relation
``e * U_0 = h * nu - phi`` makes the ``U_0`` vs. ``nu`` slope equal to ``h/e``,
so a straight-line fit yields Planck's constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from uncertainties import ufloat
from uncertainties import unumpy as unp

# --- Constants ------------------------------------------------------------- #
GRATING_SPACING = 1 / 600000        # diffraction grating line spacing (m)
SPEED_OF_LIGHT = 299792458          # m/s
ELEMENTARY_CHARGE = 1.602e-19       # C (value used in the experiment)
PLANCK_REFERENCE = 6.62607015e-34   # J.s, literature value for comparison

ANGLE_UNCERTAINTY = 0.25            # goniometer resolution (deg)
VOLTAGE_UNCERTAINTY = 0.01          # stopping-voltage resolution (V)

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "photoelectric.csv"


@dataclass
class PhotoelectricResult:
    table: pd.DataFrame
    slope: object            # U_0 vs nu slope = h/e (ufloat)
    work_function_eV: object  # phi from the intercept (ufloat)
    planck_constant: object  # h = slope * e (ufloat)


def load_measurements(path: Path | str = DATA_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def wavelength(angle_deg):
    """First-order diffraction wavelength from the goniometer angle."""
    return GRATING_SPACING * unp.sin(unp.radians(angle_deg))


def frequency(wavelength_m):
    return SPEED_OF_LIGHT / wavelength_m


def linear_fit(x, y) -> tuple:
    """Uncertainty-weighted straight-line fit, returning ``(slope, intercept)``.

    ``cov="unscaled"`` treats the measurement uncertainties as absolute (they are
    not rescaled by the reduced chi-square of the fit), matching the convention of
    the original coursework analysis.
    """
    xn = unp.nominal_values(x)
    yn, ys = unp.nominal_values(y), unp.std_devs(y)
    coeffs, cov = np.polyfit(xn, yn, 1, w=1 / ys, cov="unscaled")
    slope = ufloat(coeffs[0], np.sqrt(cov[0, 0]))
    intercept = ufloat(coeffs[1], np.sqrt(cov[1, 1]))
    return slope, intercept


def run(path: Path | str = DATA_FILE) -> PhotoelectricResult:
    df = load_measurements(path)

    angle = unp.uarray(df["diffraction_angle_deg"].to_numpy(), ANGLE_UNCERTAINTY)
    stopping_voltage = unp.uarray(
        df["stopping_voltage_V"].to_numpy(), VOLTAGE_UNCERTAINTY
    )

    wl = wavelength(angle)
    nu = frequency(wl)
    slope, intercept = linear_fit(nu, stopping_voltage)

    planck = slope * ELEMENTARY_CHARGE
    work_function = -intercept  # phi/e in volts == work function in eV

    table = pd.DataFrame(
        {
            "angle_deg": df["diffraction_angle_deg"],
            "wavelength_m": wl,
            "frequency_Hz": nu,
            "stopping_voltage_V": stopping_voltage,
        }
    )
    return PhotoelectricResult(table=table, slope=slope,
                               work_function_eV=work_function, planck_constant=planck)
