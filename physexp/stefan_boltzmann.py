"""Stefan-Boltzmann experiment: radiated power vs. filament temperature.

A tungsten lamp filament is driven at increasing currents; its resistance gives
its temperature, and a thermopile measures the emitted radiation. Fitting
``log(U_thermopile)`` against ``log(T)`` recovers the Stefan-Boltzmann exponent,
expected to be 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from uncertainties import ufloat
from uncertainties import unumpy as unp

# --- Lamp calibration constants -------------------------------------------- #
ROOM_RESISTANCE = ufloat(0.1773, 0.0009)  # filament resistance at room temp (ohm)
ALPHA = 4.82e-3                            # linear resistivity coefficient (1/K)
BETA = 6.76e-7                             # quadratic resistivity coefficient (1/K^2)
ROOM_TEMPERATURE_C = 24                    # thermocouple room temperature (C)

CURRENT_UNCERTAINTY = 0.01   # ammeter resolution (A)
VOLTAGE_UNCERTAINTY = 0.01   # lamp voltmeter resolution (V)

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "stefan_boltzmann.csv"


@dataclass
class StefanBoltzmannResult:
    table: pd.DataFrame
    nominal_resistance: object  # R_0 (ufloat)
    exponent: object            # fitted Stefan-Boltzmann exponent (ufloat)
    intercept: object           # fit intercept (ufloat)


def load_measurements(path: Path | str = DATA_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def nominal_resistance() -> object:
    """Filament resistance extrapolated to 0 C (R_0)."""
    t = ROOM_TEMPERATURE_C
    return ROOM_RESISTANCE / (1 + ALPHA * t + BETA * t**2)


def filament_temperature(resistance, r0):
    """Invert the quadratic resistance-temperature law to get temperature (K)."""
    return 273 + 1 / (2 * BETA) * (
        unp.sqrt(ALPHA**2 + 4 * BETA * (resistance / r0 - 1)) - ALPHA
    )


def linear_fit(x, y) -> tuple:
    """Uncertainty-weighted least-squares fit ``y = slope*x + intercept``.

    Weights are ``1/sigma_y``; ``cov="unscaled"`` keeps the measurement
    uncertainties absolute (no rescaling by the reduced chi-square), the same
    convention used in the original coursework analysis.
    """
    xn = unp.nominal_values(x)
    yn, ys = unp.nominal_values(y), unp.std_devs(y)
    coeffs, cov = np.polyfit(xn, yn, 1, w=1 / ys, cov="unscaled")
    slope = ufloat(coeffs[0], np.sqrt(cov[0, 0]))
    intercept = ufloat(coeffs[1], np.sqrt(cov[1, 1]))
    return slope, intercept


def run(path: Path | str = DATA_FILE) -> StefanBoltzmannResult:
    df = load_measurements(path)

    current = unp.uarray(df["current_A"].to_numpy(), CURRENT_UNCERTAINTY)
    lamp_voltage = unp.uarray(df["lamp_voltage_V"].to_numpy(), VOLTAGE_UNCERTAINTY)
    thermopile = unp.uarray(df["thermopile_voltage_V"].to_numpy(), VOLTAGE_UNCERTAINTY)

    r0 = nominal_resistance()
    resistance = lamp_voltage / current
    temperature = filament_temperature(resistance, r0)

    log_t = unp.log(temperature)
    log_u = unp.log(thermopile)
    exponent, intercept = linear_fit(log_t, log_u)

    table = pd.DataFrame(
        {
            "current_A": df["current_A"],
            "resistance_ohm": resistance,
            "temperature_K": temperature,
            "thermopile_voltage_V": thermopile,
            "log_T": log_t,
            "log_U_term": log_u,
        }
    )
    return StefanBoltzmannResult(table=table, nominal_resistance=r0,
                                 exponent=exponent, intercept=intercept)
