# oc_py — O‑C (Observed – Calculated) Fitting for Astronomers

`oc_py` is a Python package for performing *O‑C (Observed minus Calculated)* analysis commonly used in observational astronomy.
O‑C analysis helps compare observed event times (e.g., eclipse minima, transit mid‑times, pulsation timings) with predicted values from a model to study period changes and systematic deviations.

This package provides tools to load timing data, compute predicted ephemerides, perform O‑C calculations, fit models, and visualize O‑C diagrams — making it easier to analyze timing residuals and detect trends.

> Designed for researchers and students working with time‑series events in astronomy. ([GitHub][1])

---

## Features

* Compute observed minus calculated (O‑C) residuals
* Fit timing models (linear, polynomial, custom ephemerides)
* Load timing datasets (CSV, plain text, or custom formats)
* Plot O‑C diagrams with uncertainties
* Tools for simulation and example datasets

---

## 💡 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/mshemuni/oc_py.git
cd oc_py
```

### 2. Create a Python virtual environment (optional, recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install in editable mode (for development):

```bash
pip install -e .
```

---

## Usage Examples

### 1. Basic O‑C calculation

Suppose you have a file `timings.csv` with observed event times:

```python
from oc_py import OCFit

# Load observed timings
oc = OCFit.from_csv("timings.csv")

# Define your ephemeris (e.g., period and epoch)
oc.set_ephemeris(epoch=2450000.5, period=1.23456)

# Compute O‑C residuals
oc.compute_residuals()

# Print summary
print(oc.summary())
```

### 2. Fit a model to residuals

```python
# Fit a linear trend to the O‑C residuals
results = oc.fit_trend(degree=1)
print(results)
```

### 3. Plotting the O‑C diagram

```python
oc.plot(residuals=True, model=True)
```

(The plotting API automatically labels axes and displays uncertainties if available.)

---

## Directory Overview

```
oc_py/
├── src/oc_py/           # Python package source
├── tests/               # Unit tests
├── docs/                # Documentation and examples
├── requirements.txt     # Runtime dependencies
├── setup.py / pyproject.toml  # Packaging
└── README.md            # This file
```

---

## Development

If you want to contribute:

1. Fork the repository.
2. Create a new feature branch:

   ```bash
   git checkout -b feature/my‑awesome‑feature
   ```
3. Add tests for new functionality.
4. Submit a pull request with a clear description.

---

## Citation / Acknowledgement

If you use this tool in your research, you can cite the repository or link to it directly in your methods section.

---

## License

**oc_py** is released under the **GPL‑3.0 License** — see the `LICENSE` file for details. ([GitHub][1])

---

## Feedback & Support

Found a bug or need help? Open an issue here on GitHub and include:

* a clear description of the problem
* a minimal reproducible example
