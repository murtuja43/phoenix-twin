"""
Simulation tab UI for Phoenix Twin.

Layout:
  - Sidebar: patient selector + patient context card
  - Main:
      1. Observed history (last 8 weeks, 3 outcomes)
      2. Policy controls (sliders + run button)
      3. Forecast (observed + 4-week forecast with uncertainty bands)
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.forecaster import Policy, train_and_forecast, load_history
from src.risk_flags import evaluate as evaluate_risks, RiskFlag


# ---------- caching wrappers ----------

@st.cache_data(show_spinner=False)
def _load_cohort() -> pd.DataFrame:
    return pd.read_csv("data/patients.csv")


@st.cache_data(show_spinner=False)
def _load_traits() -> pd.DataFrame:
    return pd.read_csv("data/patient_traits.csv")


@st.cache_data(show_spinner="Running forecast...")
def _run_forecast(patient_id: int, intensity: float, volume: int,
                  target_rom: float, rest_days: int, horizon_weeks: int):
    """Cache forecasts keyed on patient + policy so slider tweaks are instant after first run."""
    policy = Policy(
        intensity_pct=intensity,
        volume_reps=volume,
        target_rom=target_rom,
        rest_days_per_week=rest_days,
    )
    _, fc, history = train_and_forecast(patient_id, policy, horizon_weeks=horizon_weeks)
    return fc.sessions, history


# ---------- plotting ----------

OUTCOME_META = {
    "rom_deg":  {"label": "Range of Motion",        "unit": "°",   "color": "#22d3ee"},
    "mqs":      {"label": "Movement Quality Score", "unit": "",    "color": "#a78bfa"},
    "hrv_ms":   {"label": "HRV (RMSSD)",            "unit": " ms", "color": "#f472b6"},
}


def _history_chart(history: pd.DataFrame, outcome: str) -> go.Figure:
    """Small chart showing just the patient's observed sessions for one outcome."""
    meta = OUTCOME_META[outcome]
    h = history.sort_values("session_index")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=h["session_index"], y=h[outcome],
        mode="lines+markers",
        line=dict(color=meta["color"], width=2),
        marker=dict(size=5),
        name="Observed",
        hovertemplate=f"Session %{{x}}<br>{meta['label']}: %{{y:.1f}}{meta['unit']}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{meta['label']} — observed", font=dict(size=13)),
        height=220,
        margin=dict(l=10, r=10, t=40, b=30),
        xaxis_title="Session",
        yaxis_title=f"{meta['label']}{meta['unit']}",
        template="plotly_dark",
        showlegend=False,
    )
    return fig


def _forecast_chart(history: pd.DataFrame, forecast_long: pd.DataFrame, outcome: str) -> go.Figure:
    """Combined chart: observed history + forecast median + p05–p95 band."""
    meta = OUTCOME_META[outcome]
    h = history.sort_values("session_index")
    f = forecast_long[forecast_long.outcome == outcome].sort_values("session_index")

    # Bridge: connect last observed point to first forecast point
    last_obs_x = h["session_index"].iloc[-1]
    last_obs_y = h[outcome].iloc[-1]
    bridge_x = [last_obs_x] + f["session_index"].tolist()
    bridge_p50 = [last_obs_y] + f["p50"].tolist()
    bridge_p05 = [last_obs_y] + f["p05"].tolist()
    bridge_p95 = [last_obs_y] + f["p95"].tolist()

    fig = go.Figure()

    # Observed history
    fig.add_trace(go.Scatter(
        x=h["session_index"], y=h[outcome],
        mode="lines+markers",
        line=dict(color="#94a3b8", width=2),
        marker=dict(size=5),
        name="Observed",
        hovertemplate=f"Session %{{x}}<br>Observed: %{{y:.1f}}{meta['unit']}<extra></extra>",
    ))

    # Uncertainty band (p05–p95), drawn as filled area
    fig.add_trace(go.Scatter(
        x=bridge_x + bridge_x[::-1],
        y=bridge_p95 + bridge_p05[::-1],
        fill="toself",
        fillcolor=_hex_to_rgba(meta["color"], 0.18),
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name="90% interval",
        showlegend=True,
    ))

    # Forecast median
    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_p50,
        mode="lines+markers",
        line=dict(color=meta["color"], width=2, dash="dot"),
        marker=dict(size=5),
        name="Forecast (median)",
        hovertemplate=f"Session %{{x}}<br>Forecast: %{{y:.1f}}{meta['unit']}<extra></extra>",
    ))

    # Vertical separator at forecast start
    fig.add_vline(x=last_obs_x + 0.5, line_dash="dash", line_color="#475569",
                  annotation_text="forecast →", annotation_position="top right",
                  annotation_font_color="#94a3b8", annotation_font_size=11)

    fig.update_layout(
        title=dict(text=f"{meta['label']}", font=dict(size=14)),
        height=280,
        margin=dict(l=10, r=10, t=40, b=30),
        xaxis_title="Session",
        yaxis_title=f"{meta['label']}{meta['unit']}",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

def _compare_chart(
    history: pd.DataFrame,
    forecast_a: pd.DataFrame,
    forecast_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    outcome: str,
) -> go.Figure:
    """Compare two policy forecasts on the same outcome."""
    meta = OUTCOME_META[outcome]
    h = history.sort_values("session_index")
    fa = forecast_a[forecast_a.outcome == outcome].sort_values("session_index")
    fb = forecast_b[forecast_b.outcome == outcome].sort_values("session_index")

    last_obs_x = h["session_index"].iloc[-1]
    last_obs_y = h[outcome].iloc[-1]

    def bridge(fc):
        return ([last_obs_x] + fc["session_index"].tolist(),
                [last_obs_y] + fc["p50"].tolist(),
                [last_obs_y] + fc["p05"].tolist(),
                [last_obs_y] + fc["p95"].tolist())

    xa, ya50, ya05, ya95 = bridge(fa)
    xb, yb50, yb05, yb95 = bridge(fb)

    fig = go.Figure()
    # observed
    fig.add_trace(go.Scatter(x=h["session_index"], y=h[outcome], mode="lines+markers",
                             line=dict(color="#94a3b8", width=2), marker=dict(size=4),
                             name="Observed"))
    # Policy A
    fig.add_trace(go.Scatter(x=xa + xa[::-1], y=ya95 + ya05[::-1], fill="toself",
                             fillcolor=_hex_to_rgba("#22d3ee", 0.15),
                             line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
                             name="A · 90% band", showlegend=True))
    fig.add_trace(go.Scatter(x=xa, y=ya50, mode="lines+markers",
                             line=dict(color="#22d3ee", width=2, dash="dot"),
                             marker=dict(size=5),
                             name=label_a))
    # Policy B
    fig.add_trace(go.Scatter(x=xb + xb[::-1], y=yb95 + yb05[::-1], fill="toself",
                             fillcolor=_hex_to_rgba("#f59e0b", 0.15),
                             line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
                             name="B · 90% band", showlegend=True))
    fig.add_trace(go.Scatter(x=xb, y=yb50, mode="lines+markers",
                             line=dict(color="#f59e0b", width=2, dash="dot"),
                             marker=dict(size=5),
                             name=label_b))

    fig.add_vline(x=last_obs_x + 0.5, line_dash="dash", line_color="#475569")

    fig.update_layout(
        title=dict(text=f"{meta['label']} — Policy A vs Policy B", font=dict(size=14)),
        height=340, margin=dict(l=10, r=10, t=40, b=30),
        xaxis_title="Session", yaxis_title=f"{meta['label']}{meta['unit']}",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #rrggbb to rgba string for translucent fills."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


# ---------- main render function ----------

def render_simulation_tab():
    cohort = _load_cohort()
    traits = _load_traits()
    patient_ids = sorted(cohort["patient_id"].unique().tolist())

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.markdown("### 👤 Patient")
        patient_id = st.selectbox(
            "Select patient",
            options=patient_ids,
            index=0,
            format_func=lambda pid: f"Patient {pid:03d}",
        )

        # Patient context card
        t = traits[traits.patient_id == patient_id].iloc[0]
        h = cohort[cohort.patient_id == patient_id].sort_values("session_index")
        last_rom = h.rom_deg.iloc[-1]
        last_mqs = h.mqs.iloc[-1]
        last_hrv = h.hrv_ms.iloc[-1]

        st.markdown(
            f"""
            <div style="background: rgba(30,41,59,0.4); padding: 14px; border-radius: 10px;
                        border-left: 3px solid #22d3ee; margin-top: 10px;">
                <div style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em;">
                    Patient Profile
                </div>
                <div style="margin-top:8px; font-size:0.9rem; line-height:1.6;">
                    <b>Baseline ROM:</b> {t.baseline_rom:.1f}°<br>
                    <b>Recovery target:</b> {t.target_rom:.1f}°<br>
                    <b>Adherence:</b> {t.adherence*100:.0f}%<br>
                    <b>Baseline HRV:</b> {t.baseline_hrv:.1f} ms
                </div>
                <hr style="border-color:#334155; margin: 10px 0;">
                <div style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em;">
                    Latest session
                </div>
                <div style="margin-top:8px; font-size:0.9rem; line-height:1.6;">
                    <b>ROM:</b> {last_rom:.1f}°<br>
                    <b>MQS:</b> {last_mqs:.1f}<br>
                    <b>HRV:</b> {last_hrv:.1f} ms
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.caption(
            "⚠️ This is a decision-support tool. Forecasts use a per-patient "
            "regression model with bootstrap uncertainty. Not a substitute for clinical judgement."
        )

    # ---------- MAIN AREA ----------
    st.subheader("🎯 What-if Simulation")
    st.caption(
        f"Forecasting recovery for **Patient {patient_id:03d}** under a candidate prescription policy. "
        f"Bands show 90% prediction interval from 30-model bootstrap ensemble."
    )

    # --- Section 1: Observed history ---
    with st.expander("📊 Observed history (last 8 weeks)", expanded=False):
        history = load_history(patient_id)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.plotly_chart(_history_chart(history, "rom_deg"), use_container_width=True)
        with c2:
            st.plotly_chart(_history_chart(history, "mqs"), use_container_width=True)
        with c3:
            st.plotly_chart(_history_chart(history, "hrv_ms"), use_container_width=True)

    # --- Section 2: Policy controls ---
    st.markdown("### 🎛 Prescription Policy")
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        intensity = st.slider("Intensity (%)", 50, 100, 80, 5,
                              help="Effort level. Higher = harder sets.")
    with pcol2:
        volume = st.slider("Volume (reps/session)", 10, 50, 25, 5,
                           help="Total reps per session.")
    with pcol3:
        target_rom = st.slider("Target ROM (°)", 80, 140, 120, 5,
                               help="Goal angle for the session.")
    with pcol4:
        rest_days = st.slider("Rest days / week", 0, 3, 1,
                              help="More rest = fewer sessions per week.")

    horizon = st.slider("Forecast horizon (weeks)", 1, 8, 4,
                        help="How far ahead to predict.")

    run = st.button("▶️ Run Simulation", type="primary", use_container_width=True)

    # --- Section 3: Forecast output ---
    if run or st.session_state.get("auto_run_done", False):
        st.session_state["auto_run_done"] = True

        forecast_long, history = _run_forecast(
            patient_id=patient_id,
            intensity=float(intensity),
            volume=int(volume),
            target_rom=float(target_rom),
            rest_days=int(rest_days),
            horizon_weeks=int(horizon),
        )

        # Summary metrics
        st.markdown("### 📈 Forecast")
        last_obs_rom = history.sort_values("session_index").rom_deg.iloc[-1]
        last_obs_mqs = history.sort_values("session_index").mqs.iloc[-1]
        last_obs_hrv = history.sort_values("session_index").hrv_ms.iloc[-1]

        last_fc = forecast_long[forecast_long.session_index == forecast_long.session_index.max()]
        f_rom = last_fc[last_fc.outcome == "rom_deg"].iloc[0]
        f_mqs = last_fc[last_fc.outcome == "mqs"].iloc[0]
        f_hrv = last_fc[last_fc.outcome == "hrv_ms"].iloc[0]

        m1, m2, m3 = st.columns(3)
        m1.metric("ROM (end of horizon)",
                  f"{f_rom.p50:.1f}°",
                  f"{f_rom.p50 - last_obs_rom:+.1f}° vs last session")
        m2.metric("MQS (end of horizon)",
                  f"{f_mqs.p50:.1f}",
                  f"{f_mqs.p50 - last_obs_mqs:+.1f} vs last session")
        m3.metric("HRV (end of horizon)",
                  f"{f_hrv.p50:.1f} ms",
                  f"{f_hrv.p50 - last_obs_hrv:+.1f} ms vs last session")

        # ---- Risk flags ----
        patient_baseline_hrv = float(traits[traits.patient_id == patient_id].iloc[0].baseline_hrv)
        current_policy = Policy(
            intensity_pct=float(intensity),
            volume_reps=int(volume),
            target_rom=float(target_rom),
            rest_days_per_week=int(rest_days),
        )
        flags = evaluate_risks(forecast_long, history, current_policy, patient_baseline_hrv)

        if flags:
            st.markdown("#### ⚠️ Risk Assessment")
            for f in flags:
                if f.severity == "danger":
                    st.error(f"**{f.title}** — {f.detail}")
                elif f.severity == "warning":
                    st.warning(f"**{f.title}** — {f.detail}")
                else:
                    st.info(f"**{f.title}** — {f.detail}")
        else:
            st.success("✅ No risk flags detected for this prescription.")

        # ---- Forecast charts ----
        st.plotly_chart(_forecast_chart(history, forecast_long, "rom_deg"), use_container_width=True)
        fc1, fc2 = st.columns(2)
        with fc1:
            st.plotly_chart(_forecast_chart(history, forecast_long, "mqs"), use_container_width=True)
        with fc2:
            st.plotly_chart(_forecast_chart(history, forecast_long, "hrv_ms"), use_container_width=True)

        # ---- Policy comparison ----
        st.markdown("---")
        st.markdown("### ⚖️ Compare with another policy")
        st.caption("Run the same patient under a second policy to see both forecasts side-by-side.")

        ccol1, ccol2, ccol3, ccol4 = st.columns(4)
        with ccol1:
            intensity_b = st.slider("Intensity B (%)", 50, 100, 60, 5, key="int_b")
        with ccol2:
            volume_b = st.slider("Volume B", 10, 50, 15, 5, key="vol_b")
        with ccol3:
            target_rom_b = st.slider("Target ROM B (°)", 80, 140, 110, 5, key="trom_b")
        with ccol4:
            rest_days_b = st.slider("Rest days B", 0, 3, 2, key="rest_b")

        if st.button("🔁 Run comparison", use_container_width=True):
            fc_b_long, _ = _run_forecast(
                patient_id=patient_id,
                intensity=float(intensity_b),
                volume=int(volume_b),
                target_rom=float(target_rom_b),
                rest_days=int(rest_days_b),
                horizon_weeks=int(horizon),
            )
            st.session_state["forecast_b"] = fc_b_long
            st.session_state["policy_b_label"] = (
                f"B: int {intensity_b}%, vol {volume_b}, ROM {target_rom_b}°, rest {rest_days_b}"
            )

        if "forecast_b" in st.session_state:
            policy_a_label = (
                f"A: int {intensity} %, vol {volume}, ROM {target_rom}°, rest {rest_days}"
            )
            st.plotly_chart(
                _compare_chart(history, forecast_long, st.session_state["forecast_b"],
                               policy_a_label, st.session_state["policy_b_label"], "rom_deg"),
                use_container_width=True,
            )

            # Delta summary
            last_idx_a = forecast_long.session_index.max()
            last_a_rom = forecast_long[(forecast_long.session_index == last_idx_a) &
                                       (forecast_long.outcome == "rom_deg")].iloc[0].p50
            fc_b = st.session_state["forecast_b"]
            last_idx_b = fc_b.session_index.max()
            last_b_rom = fc_b[(fc_b.session_index == last_idx_b) &
                              (fc_b.outcome == "rom_deg")].iloc[0].p50
            delta = last_a_rom - last_b_rom
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Policy A → ROM", f"{last_a_rom:.1f}°")
            cc2.metric("Policy B → ROM", f"{last_b_rom:.1f}°")
            cc3.metric("Δ (A − B)", f"{delta:+.1f}°")

        with st.expander("🔬 Forecast data (raw)"):
            st.dataframe(forecast_long, use_container_width=True, hide_index=True)
    else:
        st.info("Adjust the sliders above and click **Run Simulation** to generate a forecast.")