"""
Phoenix Twin — Streamlit UI.

Add-on simulation module for the Phoenix Rehabilitation Platform.
Runs standalone; designed to be embeddable as the Simulation tab
under Sensors → Wearables in the main Phoenix product.
"""

import streamlit as st

# ---------- page config ----------
st.set_page_config(
    page_title="Phoenix Twin",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- minimal custom styling to echo the Phoenix dark theme ----------
st.markdown(
    """
    <style>
      /* tighten default Streamlit padding */
      .block-container { padding-top: 2.5rem; padding-bottom: 2rem; max-width: 1400px; }
      /* header */
      .phx-header {
          display: flex; align-items: center; gap: 0.75rem;
          padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #2a2f3a;
          margin-bottom: 1rem;
      }
      .phx-logo {
          width: 36px; height: 36px; border-radius: 8px;
          background: linear-gradient(135deg, #6366f1, #22d3ee);
          display: flex; align-items: center; justify-content: center;
          color: white; font-weight: 700;
      }
      .phx-title { font-size: 1.25rem; font-weight: 600; margin: 0; }
      .phx-subtitle { color: #94a3b8; font-size: 0.85rem; margin: 0; }
      /* coming-soon block */
      .coming-soon {
          padding: 3rem; text-align: center; color: #94a3b8;
          border: 1px dashed #334155; border-radius: 12px;
          background: rgba(30, 41, 59, 0.3);
      }
      .coming-soon h3 { color: #cbd5e1; margin-bottom: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- header ----------
st.markdown(
    """
    <div class="phx-header">
        <div class="phx-logo">🔥</div>
        <div>
            <p class="phx-title">Phoenix Twin</p>
            <p class="phx-subtitle">Patient Digital Twin · What-if Rehabilitation Simulator</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- tab bar ----------
tab_live, tab_replay, tab_ai, tab_sim, tab_compare = st.tabs(
    ["📡  Live", "⏯  Replay", "🧠  AI Analysis", "🎯  Simulation", "⚖️  Compare"]
)


def render_coming_soon(name: str, blurb: str):
    """Placeholder content for tabs that will be wired up in future iterations."""
    st.markdown(
        f"""
        <div class="coming-soon">
            <h3>{name} — coming soon</h3>
            <p>{blurb}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


with tab_live:
    render_coming_soon(
        "Live monitoring",
        "Already implemented in the main Phoenix Wearables view. "
        "This tab will mirror real-time IMU and HR streams here for in-context comparison."
    )

with tab_replay:
    render_coming_soon(
        "Session replay",
        "Scrub through past sessions, inspect movement, and compare session-over-session changes."
    )

with tab_ai:
    render_coming_soon(
        "AI analysis",
        "Per-session form correction, asymmetry analysis, MQS, and AI-generated insights."
    )

with tab_sim:
    # The active tab — full implementation lives in src/ui_simulation.py
    from src.ui_simulation import render_simulation_tab
    render_simulation_tab()

with tab_compare:
    render_coming_soon(
        "Compare",
        "Patient vs. ideal movement, side-by-side replay, angle and trajectory overlap."
    )