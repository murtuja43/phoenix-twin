"""
Risk flag generator for Phoenix Twin forecasts.

Inspects a forecast and the policy that produced it, returning
human-readable warnings about clinically meaningful risk patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from src.forecaster import Policy


@dataclass
class RiskFlag:
    severity: str   # 'info' | 'warning' | 'danger'
    title: str
    detail: str


def evaluate(
    forecast_long: pd.DataFrame,
    history: pd.DataFrame,
    policy: Policy,
    patient_baseline_hrv: float,
) -> list[RiskFlag]:
    """Return a list of risk flags for the given forecast.

    Heuristics chosen to be conservative and clinically defensible.
    """
    flags: list[RiskFlag] = []

    # ---- HRV drift ----
    hrv_fc = forecast_long[forecast_long.outcome == "hrv_ms"].sort_values("session_index")
    if not hrv_fc.empty:
        min_hrv = hrv_fc.p50.min()
        end_hrv = hrv_fc.p50.iloc[-1]
        drop_from_baseline = patient_baseline_hrv - min_hrv

        if drop_from_baseline > 8:
            flags.append(RiskFlag(
                severity="danger",
                title="High under-recovery risk",
                detail=f"Forecast HRV drops {drop_from_baseline:.1f} ms below baseline "
                       f"({patient_baseline_hrv:.1f} → {min_hrv:.1f} ms). "
                       "Consider lowering intensity or adding rest days."
            ))
        elif drop_from_baseline > 4:
            flags.append(RiskFlag(
                severity="warning",
                title="Moderate fatigue accumulation",
                detail=f"HRV trends down by {drop_from_baseline:.1f} ms over the horizon. "
                       "Patient may be entering an under-recovery zone."
            ))

    # ---- Overload from policy ----
    executed_load = (policy.intensity_pct / 100.0) * policy.volume_reps
    if executed_load > 40:
        flags.append(RiskFlag(
            severity="danger",
            title="Overload prescription",
            detail=f"Executed load ({executed_load:.0f} units) exceeds the model's training range. "
                   "Forecasts in this region rely on extrapolation and should be treated cautiously."
        ))
    elif executed_load > 32:
        flags.append(RiskFlag(
            severity="warning",
            title="High prescribed load",
            detail=f"Executed load ({executed_load:.0f} units) is near the top of the typical range. "
                   "Expect higher fatigue accumulation."
        ))

    # ---- ROM stagnation ----
    rom_fc = forecast_long[forecast_long.outcome == "rom_deg"].sort_values("session_index")
    if not rom_fc.empty and len(history) > 0:
        last_obs_rom = history.sort_values("session_index").rom_deg.iloc[-1]
        end_rom = rom_fc.p50.iloc[-1]
        gain = end_rom - last_obs_rom
        if gain < 1.0:
            flags.append(RiskFlag(
                severity="warning",
                title="Limited ROM progression",
                detail=f"Forecast predicts only {gain:+.1f}° ROM gain over {len(rom_fc)} sessions. "
                       "Consider adjusting prescription or reviewing patient adherence."
            ))
        elif gain > 3:
            flags.append(RiskFlag(
                severity="info",
                title="Strong ROM progression",
                detail=f"Forecast predicts +{gain:.1f}° ROM gain — patient is responding well to this policy."
            ))

    # ---- Wide uncertainty (low model confidence) ----
    rom_band = (rom_fc.p95 - rom_fc.p05).iloc[-1] if not rom_fc.empty else 0
    if rom_band > 12:
        flags.append(RiskFlag(
            severity="warning",
            title="Low forecast confidence",
            detail=f"ROM uncertainty band is wide ({rom_band:.1f}°). "
                   "Model is uncertain — collect more sessions or treat forecast as exploratory."
        ))

    # ---- Rest day pattern ----
    if policy.rest_days_per_week == 0:
        flags.append(RiskFlag(
            severity="info",
            title="No prescribed rest days",
            detail="Schedule has no rest days. Monitor HRV closely for early signs of overtraining."
        ))

    return flags