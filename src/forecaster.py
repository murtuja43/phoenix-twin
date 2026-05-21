"""
Per-patient forecaster for Phoenix Twin.

Given a patient's session history and a candidate prescription policy,
predict the next N sessions of (ROM, MQS, HRV) with uncertainty bands.

Approach:
  - Three independent gradient-boosted regressors (ROM, MQS, HRV).
  - Bootstrap ensemble: retrain on resampled history, aggregate predictions.
  - Iterative rollout: each forecasted session feeds the next as new "history".

This is a deliberately simple, fast, defensible baseline. Upgradable later
to Gaussian Processes or neural ODEs without changing the API surface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.ensemble import GradientBoostingRegressor
from pathlib import Path


# ---------- configuration ----------

OUTCOMES = ["rom_deg", "mqs", "hrv_ms"]
SESSIONS_PER_WEEK = 3
N_BOOTSTRAP = 30          # ensemble size for uncertainty
BOOTSTRAP_FRAC = 0.85     # fraction of history each bootstrap model trains on
MODEL_PARAMS = dict(
    n_estimators=80,
    max_depth=3,
    learning_rate=0.08,
    random_state=0,        # base seed; bootstraps perturb it
)


# ---------- public API ----------

@dataclass
class Policy:
    """A 'what-if' prescription policy applied over the forecast horizon."""
    intensity_pct: float        # 50–100
    volume_reps: int            # 10–50
    target_rom: float           # degrees, the prescribed goal
    rest_days_per_week: int = 1 # 0–3; reduces sessions per week if > 1

    def sessions_per_week(self) -> int:
        """Active days per week given prescribed rest."""
        return max(1, SESSIONS_PER_WEEK + 1 - self.rest_days_per_week)


@dataclass
class Forecast:
    """Forecast output for one patient under one policy."""
    sessions: pd.DataFrame  # columns: session_index, week, outcome, p05, p50, p95


# ---------- feature engineering ----------

def _build_features(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn a session log into (X, y) for training.

    Features per session (the 'state' the model sees):
      session_index, week, intensity_pct, volume_reps, prescribed_target_rom,
      executed_load, cum_load, lag1_rom, lag1_mqs, lag1_hrv
    """
    h = history.sort_values("session_index").reset_index(drop=True).copy()

    # Lag features — what the patient's last session looked like
    for col in OUTCOMES:
        h[f"lag1_{col}"] = h[col].shift(1)

    # Cumulative load up to (but not including) this session
    h["cum_load"] = h["executed_load"].shift(1).fillna(0).cumsum()

    # First session has no lags — fill with the patient's own first observed values
    for col in OUTCOMES:
        h[f"lag1_{col}"] = h[f"lag1_{col}"].fillna(h[col].iloc[0])

    feature_cols = [
        "session_index", "week",
        "intensity_pct", "volume_reps", "prescribed_target_rom",
        "executed_load", "cum_load",
        "lag1_rom_deg", "lag1_mqs", "lag1_hrv_ms",
    ]
    X = h[feature_cols].copy()
    y = h[OUTCOMES].copy()
    return X, y


# ---------- model training ----------

@dataclass
class PatientModel:
    """Trained bootstrap ensemble for one patient."""
    patient_id: int
    ensembles: dict = field(default_factory=dict)  # outcome -> list of fitted regressors
    feature_cols: list = field(default_factory=list)
    last_history_row: pd.Series | None = None
    last_session_index: int = 0
    last_week: int = 0
    cum_load_at_end: float = 0.0


def train_patient_model(history: pd.DataFrame) -> PatientModel:
    """Fit a bootstrap ensemble of GBR models for one patient."""
    if history.empty:
        raise ValueError("Empty history passed to train_patient_model")

    patient_id = int(history["patient_id"].iloc[0])
    X, y = _build_features(history)
    feature_cols = X.columns.tolist()

    ensembles = {outcome: [] for outcome in OUTCOMES}
    n = len(X)
    sample_n = max(8, int(np.ceil(n * BOOTSTRAP_FRAC)))

    rng = np.random.default_rng(patient_id)  # deterministic per patient

    for b in range(N_BOOTSTRAP):
        idx = rng.choice(n, size=sample_n, replace=True)
        Xb = X.iloc[idx]
        for outcome in OUTCOMES:
            yb = y[outcome].iloc[idx]
            params = dict(MODEL_PARAMS)
            params["random_state"] = b
            model = GradientBoostingRegressor(**params)
            model.fit(Xb, yb)
            ensembles[outcome].append(model)

    # Snapshot end-of-history state for rollout
    last_row = history.sort_values("session_index").iloc[-1]
    cum_load = X["cum_load"].iloc[-1] + last_row["executed_load"]

    return PatientModel(
        patient_id=patient_id,
        ensembles=ensembles,
        feature_cols=feature_cols,
        last_history_row=last_row,
        last_session_index=int(last_row["session_index"]),
        last_week=int(last_row["week"]),
        cum_load_at_end=float(cum_load),
    )


# ---------- forecasting (iterative rollout) ----------

def forecast(
    model: PatientModel,
    policy: Policy,
    horizon_weeks: int = 4,
) -> Forecast:
    """Roll the patient forward under a fixed policy for horizon_weeks weeks.

    Returns a tidy long dataframe: one row per (session, outcome) with p05/p50/p95.
    """
    sessions_pw = policy.sessions_per_week()
    n_steps = horizon_weeks * sessions_pw

    # Executed load under the new policy (assume adherence already baked into training)
    executed_load = (policy.intensity_pct / 100.0) * policy.volume_reps

    # State we evolve forward, one session at a time
    prev = {col: float(model.last_history_row[col]) for col in OUTCOMES}
    cum_load = model.cum_load_at_end
    session_idx = model.last_session_index
    week = model.last_week

    rows = []
    for step in range(1, n_steps + 1):
        session_idx += 1
        # Advance week marker every `sessions_pw` sessions
        if (step - 1) % sessions_pw == 0 and step > 1:
            week += 1
        elif step == 1:
            week += 1  # forecast starts in the next week

        feat = pd.DataFrame([{
            "session_index": session_idx,
            "week": week,
            "intensity_pct": policy.intensity_pct,
            "volume_reps": policy.volume_reps,
            "prescribed_target_rom": policy.target_rom,
            "executed_load": executed_load,
            "cum_load": cum_load,
            "lag1_rom_deg": prev["rom_deg"],
            "lag1_mqs": prev["mqs"],
            "lag1_hrv_ms": prev["hrv_ms"],
        }])[model.feature_cols]

        # Collect predictions from all bootstrap models, per outcome
        preds_this_step = {}
        for outcome in OUTCOMES:
            preds = np.array([m.predict(feat)[0] for m in model.ensembles[outcome]])
            preds_this_step[outcome] = preds
            rows.append({
                "session_index": session_idx,
                "week": week,
                "outcome": outcome,
                "p05": float(np.percentile(preds, 5)),
                "p50": float(np.percentile(preds, 50)),
                "p95": float(np.percentile(preds, 95)),
            })

        # Roll the state forward using the median prediction
        for outcome in OUTCOMES:
            prev[outcome] = float(np.percentile(preds_this_step[outcome], 50))
        cum_load += executed_load

    return Forecast(sessions=pd.DataFrame(rows))


# ---------- convenience: load + train in one call ----------

def load_history(patient_id: int, csv_path: str | Path = "data/patients.csv") -> pd.DataFrame:
    """Load a single patient's history from the cohort CSV."""
    df = pd.read_csv(csv_path)
    sub = df[df.patient_id == patient_id].copy()
    if sub.empty:
        raise ValueError(f"No history for patient {patient_id}")
    return sub


def train_and_forecast(
    patient_id: int,
    policy: Policy,
    horizon_weeks: int = 4,
    csv_path: str | Path = "data/patients.csv",
) -> tuple[PatientModel, Forecast, pd.DataFrame]:
    """High-level convenience: load history, train, forecast. Returns all three."""
    history = load_history(patient_id, csv_path)
    model = train_patient_model(history)
    fc = forecast(model, policy, horizon_weeks=horizon_weeks)
    return model, fc, history


# ---------- smoke test ----------

def main():
    """Quick smoke test: train one patient, forecast under a default policy."""
    policy = Policy(intensity_pct=80, volume_reps=25, target_rom=120, rest_days_per_week=1)

    print(f"Training patient 1 with policy: {policy}")
    model, fc, history = train_and_forecast(patient_id=1, policy=policy, horizon_weeks=4)

    print(f"\nHistory: {len(history)} sessions (weeks {history.week.min()}-{history.week.max()})")
    print(f"Last observed ROM: {history.sort_values('session_index').rom_deg.iloc[-1]:.1f}°")
    print(f"\nForecast ({len(fc.sessions)//3} sessions × 3 outcomes):")
    pivot = fc.sessions.pivot(index="session_index", columns="outcome", values="p50")
    print(pivot.round(1).to_string())

    print(f"\nForecast uncertainty bands (final session):")
    last_sess = fc.sessions.session_index.max()
    last = fc.sessions[fc.sessions.session_index == last_sess]
    for _, r in last.iterrows():
        print(f"  {r.outcome:8s}  p05={r.p05:6.1f}  p50={r.p50:6.1f}  p95={r.p95:6.1f}")


if __name__ == "__main__":
    main()