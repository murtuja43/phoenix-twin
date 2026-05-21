# Phoenix Twin — Rehabilitation Simulation Module

A "what-if" simulator for rehabilitation prescription policies. Predicts how a patient's recovery trajectory (ROM, MQS, HRV) responds to changes in exercise intensity, volume, target range of motion, and rest days over a 1–8 week horizon, with bootstrap uncertainty bands.

Designed as an add-on module for the Phoenix Rehabilitation Platform (aiphoenix.kz). Runs standalone as a Streamlit app; integrates as the Simulation tab under Sensors → Wearables.

## Features

- **Per-patient forecasting model** — gradient-boosted regression ensemble, retrained per patient on their session history
- **Honest uncertainty** — 30-model bootstrap ensemble, 5th–95th percentile bands on every forecast
- **Risk flags** — automatic warnings for under-recovery, overload prescription, ROM stagnation, low-confidence forecasts
- **Policy comparison** — overlay two prescription policies on the same patient to support clinical decisions
- **Synthetic patient cohort** — 200 virtual patients with realistic recovery dynamics for development and demo
- **Tab shell** — Live / Replay / AI Analysis / Simulation / Compare, with the other four shown as roadmap placeholders

## Architecture
app.py                      # Streamlit shell, tab bar, page styling
src/
patient_generator.py      # Synthetic cohort generation (200 virtual patients)
forecaster.py             # Per-patient GBR ensemble + bootstrap rollout
risk_flags.py             # Clinical heuristic warnings on forecasts
ui_simulation.py          # Simulation tab UI: sidebar, sliders, charts
data/
patients.csv              # Generated session log (24 sessions × 200 patients)
patient_traits.csv        # Hidden recovery parameters per patient


## Run locally

```bash
conda create -n phoenix-twin python=3.12 -y
conda activate phoenix-twin
pip install streamlit pandas numpy scikit-learn plotly matplotlib

python -m src.patient_generator   # generates data/patients.csv
streamlit run app.py
```

Open http://localhost:8501.

## Technical notes

- Forecaster: per-patient bootstrap ensemble of `GradientBoostingRegressor` (n_estimators=80, max_depth=3). One model per outcome (ROM, MQS, HRV). 30 bootstrap samples at 85% of history each.
- Features: session index, week, intensity %, volume reps, prescribed target ROM, executed load, cumulative load, lagged previous-session values for each outcome.
- Iterative rollout: forecast horizon is rolled forward one session at a time; each step's median prediction becomes the lag feature for the next step.

## Limitations and honest caveats

- Synthetic data only. Forecasts have not been validated against real patient outcomes.
- Bootstrap intervals are approximate uncertainty estimates, not formally calibrated prediction intervals.
- The model cannot extrapolate well past the prescription regimes it has observed in training. The "Overload prescription" risk flag exists to warn users when policies push into extrapolation territory.
- Not a substitute for clinical judgement. Designed as a decision-support tool only.

## Roadmap

- Real-data validation on KIMORE or PAMAP2 subset
- Bayesian policy optimizer over the prescription space
- Direct integration into Phoenix Sensors → Wearables tab
- Upgrade forecaster to Gaussian Process or neural ODE for principled uncertainty