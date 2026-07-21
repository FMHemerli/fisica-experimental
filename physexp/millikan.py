"""Millikan oil-drop experiment: elementary charge from droplet motion.

Reproduces the original notebook's physics exactly, but operates on whole
``unumpy`` arrays instead of ``for i in range(0, 20, 1)`` loops over lists of
scalar ``ufloat`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from uncertainties import ufloat
from uncertainties import unumpy as unp

from .display import uabs

# --- Physical constants ---------------------------------------------------- #
OIL_DENSITY = 1.03e3      # silicone oil density (kg/m^3)
AIR_DENSITY = 1.293       # air density at room temperature (kg/m^3)
AIR_VISCOSITY = 1.82e-5   # air dynamic viscosity (kg/m.s)
GRAVITY = 9.79            # local gravitational acceleration (m/s^2)

PLATE_DISTANCE = ufloat(2.5e-3, 0.01e-3)   # capacitor plate separation (m)
SCALE_HEIGHT = ufloat(8.9e-4, 0.1e-4)      # eyepiece scale height (m)
TIME_UNCERTAINTY = 0.001                   # timing resolution (s)
VOLTAGE_UNCERTAINTY = 5.0                  # applied-voltage uncertainty (V)

# Literature value, used only as a reference line in plots/summaries.
ELEMENTARY_CHARGE = 1.602176634e-19        # C

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "millikan_droplets.csv"


@dataclass
class MillikanResult:
    """Everything a notebook needs to present the experiment."""

    table: pd.DataFrame          # per-droplet nominal + uncertainty quantities
    charge_levels: pd.DataFrame  # grouped charge plateaus L1..L5
    elementary_charge: object    # estimated e (ufloat)


def load_measurements(path: Path | str = DATA_FILE) -> pd.DataFrame:
    """Load the raw droplet timings/voltages."""
    return pd.read_csv(path)


def droplet_speeds(df: pd.DataFrame) -> tuple:
    """Rise and fall speeds for every droplet (vectorized).

    A droplet crosses ``grid_divisions/30`` scale heights; dividing that travel
    distance by the timed durations gives the two speeds.
    """
    travel = (df["grid_divisions"].to_numpy() / 30.0) * SCALE_HEIGHT
    rise_time = unp.uarray(df["rise_time_s"].to_numpy(), TIME_UNCERTAINTY)
    fall_time = unp.uarray(df["descent_time_s"].to_numpy(), TIME_UNCERTAINTY)
    return travel / rise_time, travel / fall_time


def droplet_radius(rise_speed, fall_speed):
    """Stokes radius of each droplet from the speed difference."""
    speed_gap = uabs(rise_speed - fall_speed)
    coeff = 1.5 * np.sqrt(AIR_VISCOSITY / (GRAVITY * (OIL_DENSITY - AIR_DENSITY)))
    return coeff * unp.sqrt(speed_gap)


def droplet_charge(rise_speed, fall_speed, voltage):
    """Electric charge of each droplet from its balanced/free motion."""
    speed_gap = uabs(rise_speed - fall_speed)
    coeff = (9 * np.pi * PLATE_DISTANCE / 2) * unp.sqrt(
        AIR_VISCOSITY**3 / (GRAVITY * (OIL_DENSITY - AIR_DENSITY))
    )
    return coeff * ((rise_speed + fall_speed) / voltage) * unp.sqrt(speed_gap)


# Charge plateaus: the original grouping of the sorted charges into five levels
# assumed to be consecutive integer multiples of the elementary charge.
_LEVEL_SLICES = [slice(0, 4), slice(4, 8), slice(8, 11), slice(11, 18), slice(18, 20)]


def charge_levels(charge) -> tuple:
    """Average the sorted charges into the five plateaus and return ``(levels, e)``.

    ``e`` is the mean spacing between consecutive plateaus, i.e. the estimated
    elementary charge under the single-multiple-per-level assumption.
    """
    order = np.argsort(unp.nominal_values(charge))
    ordered = charge[order]
    levels = np.array([ordered[s].mean() for s in _LEVEL_SLICES], dtype=object)
    spacings = np.diff(np.concatenate([[0.0], levels]))
    return levels, spacings.mean()


def run(path: Path | str = DATA_FILE) -> MillikanResult:
    """Full pipeline: raw CSV -> per-droplet table + charge levels + e estimate."""
    df = load_measurements(path)
    rise_speed, fall_speed = droplet_speeds(df)
    radius = droplet_radius(rise_speed, fall_speed)
    charge = droplet_charge(rise_speed, fall_speed,
                            unp.uarray(df["voltage_V"].to_numpy(), VOLTAGE_UNCERTAINTY))

    table = pd.DataFrame(
        {
            "droplet": df["droplet"],
            "voltage_V": df["voltage_V"],
            "rise_speed_m_s": rise_speed,
            "fall_speed_m_s": fall_speed,
            "radius_m": radius,
            "charge_C": charge,
        }
    )

    levels, e_estimate = charge_levels(charge)
    levels_table = pd.DataFrame(
        {"level": [f"L{i + 1}" for i in range(len(levels))], "charge_C": levels}
    )
    return MillikanResult(table=table, charge_levels=levels_table,
                          elementary_charge=e_estimate)
