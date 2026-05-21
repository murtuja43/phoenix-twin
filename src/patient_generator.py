"""
Synthetic patient generator for Phoenix Twin.

Generates a cohort of virtual rehabilitation patients with multi-week
session histories. Each patient has hidden recovery traits that determine
how they respond to prescribed exercise load. The forecasting model in
Step 3 only sees the observable session log and must infer the response.

Outputs a CSV at data/patients.csv with one row per (patient, session).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

# Reproducibility — fix the seed so everyone generating data gets the same cohort
RNG_SEED = 42

# Cohort and timeline configuration
N_PATIENTS = 200
N_WEEKS = 8
SESSIONS_PER_WEEK = 3  # typical rehab schedule: Mon/Wed/Fri


@dataclass
class PatientTraits:
    """Hidden parameters that govern a single patient's recovery dynamics.
    The forecaster never sees these — it must infer behaviour from sessions."""

    patient_id: int
    baseline_rom: float          # starting ROM in degrees (post-injury)
    target_rom: float            # asymptotic recoverable ROM
    recovery_rate: float         # how fast ROM improves per week with good load
    fatigue_susceptibility: float  # how much overload damages MQS / HRV
    adherence: float             # 0..1, fraction of prescribed load actually done
    baseline_hrv: float          # personal HRV baseline (RMSSD ms)
    noise_level: float           # per-session measurement noise


def sample_traits(rng: np.random.Generator, patient_id: int) -> PatientTraits:
    """Draw one virtual patient from realistic clinical distributions."""
    return PatientTraits(
        patient_id=patient_id,
        baseline_rom=rng.uniform(55, 75),      # degrees after surgery
        target_rom=rng.uniform(125, 140),      # healthy knee flexion target
        recovery_rate=rng.uniform(0.08, 0.18), # per-week response to load
        fatigue_susceptibility=rng.uniform(0.5, 1.5),
        adherence=rng.uniform(0.7, 1.0),
        baseline_hrv=rng.uniform(35, 70),      # RMSSD in ms
        noise_level=rng.uniform(1.5, 4.0),
    )


def sample_prescription(rng: np.random.Generator, week: int, traits: PatientTraits) -> dict:
    """Clinicians ramp up load over weeks. We simulate a plausible prescription
    that gets harder as the patient progresses, with some noise."""
    progress = week / N_WEEKS
    intensity = np.clip(60 + 30 * progress + rng.normal(0, 5), 50, 100)
    volume = int(np.clip(15 + 25 * progress + rng.normal(0, 3), 10, 50))
    # Prescribed target ROM creeps up as patient improves
    target = np.clip(traits.baseline_rom + (traits.target_rom - traits.baseline_rom) * progress + rng.normal(0, 3),
                     traits.baseline_rom, traits.target_rom)
    return {
        "intensity_pct": float(intensity),
        "volume_reps": int(volume),
        "prescribed_target_rom": float(target),
    }


def simulate_session(
    rng: np.random.Generator,
    traits: PatientTraits,
    week: int,
    session_in_week: int,
    prev_rom: float,
    prev_mqs: float,
    cumulative_load: float,
) -> dict:
    """Simulate one rehab session for one patient.

    Recovery dynamics (simplified but realistic):
      - ROM improves with executed load, asymptoting toward target_rom
      - MQS improves with practice but drops when load × susceptibility is high
      - HRV drops when recent load is high (under-recovery signal)
    """
    rx = sample_prescription(rng, week, traits)
    executed_load = (rx["intensity_pct"] / 100.0) * rx["volume_reps"] * traits.adherence

    # ROM update: exponential approach to target, scaled by executed load
    rom_gain = traits.recovery_rate * (executed_load / 30.0) * (traits.target_rom - prev_rom) / traits.target_rom
    rom = prev_rom + rom_gain + rng.normal(0, traits.noise_level)
    rom = float(np.clip(rom, traits.baseline_rom - 5, traits.target_rom + 5))

    # MQS update: learns toward a ceiling, penalized by overload
    overload_penalty = max(0, executed_load - 35) * traits.fatigue_susceptibility
    mqs_base = 50 + 35 * (rom - traits.baseline_rom) / max(1.0, traits.target_rom - traits.baseline_rom)
    mqs = mqs_base - overload_penalty * 0.4 + rng.normal(0, 3)
    mqs = float(np.clip(mqs, 0, 100))

    # HRV: baseline minus under-recovery, plus noise
    recent_load_factor = cumulative_load / max(1.0, week * SESSIONS_PER_WEEK)
    hrv_drop = traits.fatigue_susceptibility * recent_load_factor * 0.25
    hrv = traits.baseline_hrv - hrv_drop + rng.normal(0, 2.5)
    hrv = float(np.clip(hrv, 15, 100))

    return {
        "patient_id": traits.patient_id,
        "week": week,
        "session_in_week": session_in_week,
        "session_index": (week - 1) * SESSIONS_PER_WEEK + session_in_week,
        "intensity_pct": rx["intensity_pct"],
        "volume_reps": rx["volume_reps"],
        "prescribed_target_rom": rx["prescribed_target_rom"],
        "executed_load": float(executed_load),
        "rom_deg": rom,
        "mqs": mqs,
        "hrv_ms": hrv,
    }


def generate_patient_history(rng: np.random.Generator, traits: PatientTraits) -> list[dict]:
    """Run the per-session simulation for one patient over N_WEEKS."""
    sessions = []
    prev_rom = traits.baseline_rom
    prev_mqs = 50.0
    cumulative_load = 0.0

    for week in range(1, N_WEEKS + 1):
        for s in range(1, SESSIONS_PER_WEEK + 1):
            session = simulate_session(rng, traits, week, s, prev_rom, prev_mqs, cumulative_load)
            sessions.append(session)
            prev_rom = session["rom_deg"]
            prev_mqs = session["mqs"]
            cumulative_load += session["executed_load"]

    return sessions


def generate_cohort(n_patients: int = N_PATIENTS, seed: int = RNG_SEED) -> pd.DataFrame:
    """Build the full cohort dataframe."""
    rng = np.random.default_rng(seed)
    all_rows = []
    traits_rows = []

    for pid in range(1, n_patients + 1):
        traits = sample_traits(rng, pid)
        traits_rows.append({
            "patient_id": traits.patient_id,
            "baseline_rom": traits.baseline_rom,
            "target_rom": traits.target_rom,
            "recovery_rate": traits.recovery_rate,
            "fatigue_susceptibility": traits.fatigue_susceptibility,
            "adherence": traits.adherence,
            "baseline_hrv": traits.baseline_hrv,
        })
        all_rows.extend(generate_patient_history(rng, traits))

    sessions_df = pd.DataFrame(all_rows)
    traits_df = pd.DataFrame(traits_rows)
    return sessions_df, traits_df


def main():
    """Generate cohort and write CSVs to data/."""
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)

    sessions_df, traits_df = generate_cohort()
    sessions_path = out_dir / "patients.csv"
    traits_path = out_dir / "patient_traits.csv"

    sessions_df.to_csv(sessions_path, index=False)
    traits_df.to_csv(traits_path, index=False)

    print(f"Generated {len(sessions_df)} sessions across {sessions_df['patient_id'].nunique()} patients")
    print(f"  → {sessions_path}")
    print(f"  → {traits_path}")
    print("\nSample rows:")
    print(sessions_df.head(6).to_string(index=False))
    print("\nPer-patient session count:")
    print(sessions_df.groupby('patient_id').size().describe())


if __name__ == "__main__":
    main()