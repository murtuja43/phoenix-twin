# Phoenix Twin — Rehabilitation Simulation Module

A "what-if" simulator for rehabilitation prescription policies. Predicts how a patient's recovery trajectory (ROM, MQS, HRV) responds to changes in exercise intensity, volume, and target range of motion over a 4-week horizon, with uncertainty bands.

Designed as an add-on module for the Phoenix Rehabilitation Platform (aiphoenix.kz). Runs standalone as a Streamlit app; integrates into Phoenix as the Simulation tab under Sensors → Wearables.

## Stack
- Python 3.12, Streamlit, scikit-learn, plotly
- Regression + bootstrap forecaster (per-patient)
- Synthetic patient cohort (200 virtual patients, 8–12 week trajectories)

## Run locally
```bash
conda activate phoenix-twin
streamlit run app.py
```

## Status
Work in progress.