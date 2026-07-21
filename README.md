# Modern Physics Laboratory Analysis

![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white)
![uncertainties](https://img.shields.io/badge/uncertainties-error_propagation-8A2BE2)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)

Computational analysis for Modern Physics laboratory experiments (CCENS-UFES).
Each experiment turns raw measurements into physical constants with full error
propagation via [`uncertainties`](https://pythonhosted.org/uncertainties/).

## Experiments

| Experiment | Result | Notebook |
|---|---|---|
| Millikan oil drop | elementary charge `e` | `notebooks/millikan.ipynb` |
| Stefan-Boltzmann | radiation-law exponent (≈4) | `notebooks/stefan_boltzmann.ipynb` |
| Photoelectric effect | Planck's constant `h` | `notebooks/photoelectric.ipynb` |

## Layout

```
data/        raw measurements as CSV (one file per experiment)
physexp/     reusable calculation + plotting package
notebooks/   thin presentation notebooks (load data, call physexp, show tables/plots)
figures/     exported figures
```

The physics lives in `physexp/*.py` as vectorized, uncertainty-aware functions;
the notebooks only load data, call `run()`, and present tables and figures. Raw
data is kept in `data/*.csv`, separate from the code.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Open any notebook under `notebooks/`, or drive the package directly:

```python
from physexp import millikan
result = millikan.run()
print(result.elementary_charge)   # e with propagated uncertainty
```

## License

Licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** — see [`LICENSE`](LICENSE).

Copyright (C) 2026 Flávio Manoel Santos Hemerli

You may use, modify and redistribute this code, including commercially, but any derivative work — including software you run as a networked service — must be released as open source under the same AGPL-3.0 terms.
