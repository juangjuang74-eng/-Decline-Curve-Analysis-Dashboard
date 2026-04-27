# Decline Curve Analysis Dashboard

> A production-grade petroleum engineering tool for Arps decline curve analysis, 
> built as a portfolio project targeting E&P companies.

## Features

- Fit **Exponential, Hyperbolic, and Harmonic** Arps decline models
- Automatic parameter estimation via non-linear least squares (scipy)
- Interactive **rate vs time** charts (linear + log scale)
- **Cumulative production** forecast with EUR estimation
- **Revenue projection** based on oil price input
- Upload your own production CSV or use the built-in sample generator
- Export forecast data to CSV

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Free)

1. Push to GitHub
2. Go to share.streamlit.io
3. Connect your repo → Deploy

## Methods

| Model | Equation | Use Case |
|-------|----------|----------|
| Exponential | q = qi·e^(-di·t) | Tight/shale, b≈0 |
| Hyperbolic | q = qi·(1+b·di·t)^(-1/b) | Most conventional wells, 0<b<1 |
| Harmonic | q = qi·(1+di·t)^(-1) | Gravity drainage, b=1 |

**References:**
- Arps, J.J. (1945). Analysis of Decline Curves. *Trans. AIME*, 160, 228-247.
- Standing, M.B. (1977). Volumetric and Phase Behavior of Oil Field Hydrocarbon Systems.

## Tech Stack

`Python` · `Streamlit` · `Plotly` · `SciPy` · `petropt` · `pandas` · `numpy`
