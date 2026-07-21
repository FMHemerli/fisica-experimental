"""Computational routines for the Modern Physics laboratory experiments.

Each submodule turns a raw-measurements ``DataFrame`` (loaded from ``data/*.csv``)
into derived quantities carrying full error propagation via ``uncertainties``.
The heavy lifting is vectorized with ``uncertainties.unumpy`` instead of the
element-by-element Python loops used in the original notebooks.
"""

from . import display, millikan, photoelectric, stefan_boltzmann

__all__ = ["display", "millikan", "photoelectric", "stefan_boltzmann"]
