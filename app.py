"""
Phoenix Twin — Streamlit UI styled to match the live Phoenix Rehabilitation Platform.

Layout mirrors aiphoenix.kz/patient/sensors:
  Top nav (Home / My Health / Exercises / My Progress / My Doctors / Messages / Sensors / Settings / Profile)
    → Sensor Services row (Rehabilitation / Polar · HR/HRV / Wearables)
       → Phoenix Wearables sub-tabs (Live / Replay / AI Analysis / Simulation / Compare)
          → Simulation tab is the working A5 digital-twin feature.

All non-Simulation tabs are visual mocks of the live product.
"""

import base64
from pathlib import Path
import streamlit as st


# ---------- page config ----------
st.set_page_config(
    page_title="Phoenix — Rehabilitation Platform",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- helpers ----------
def _b64_logo() -> str:
    """Encode the Phoenix logo as base64 so we can inline it in HTML."""
    path = Path(__file__).parent / "assets" / "phoenix-logo.png"
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


LOGO_B64 = _b64_logo()


# ---------- global styling ----------
st.markdown(
    """
    <style>
      /* Hide Streamlit chrome */
      #MainMenu { visibility: hidden; }
      header[data-testid="stHeader"] { display: none; }
      footer { visibility: hidden; }

      /* Light Phoenix theme */
      .stApp {
        background: #f8faf9;
        color: #1f2937;
      }
      .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1280px;
      }

      /* Sidebar styling */
      section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
      }
      section[data-testid="stSidebar"] * { color: #1f2937; }

      /* ---------- Phoenix top nav ---------- */
      .phx-topnav {
        display: flex;
        align-items: center;
        gap: 28px;
        padding: 14px 8px;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 20px;
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
      }
      .phx-brand {
        display: flex; align-items: center; gap: 10px;
        font-size: 1.25rem; font-weight: 700; color: #1f2937;
      }
      .phx-brand img { width: 32px; height: 32px; border-radius: 8px; }
      .phx-nav-items {
        display: flex; gap: 4px; flex: 1; justify-content: center;
        flex-wrap: wrap;
      }
      .phx-nav-item {
        padding: 8px 14px; border-radius: 8px;
        color: #6b7280; font-size: 0.92rem; font-weight: 500;
        display: inline-flex; align-items: center; gap: 6px;
        cursor: default;
      }
      .phx-nav-item.active {
        background: #ecfdf5; color: #059669;
      }
      .phx-nav-right {
        display: flex; align-items: center; gap: 10px;
      }
      .phx-avatar {
        width: 32px; height: 32px; border-radius: 50%;
        background: #f3f4f6; color: #6b7280;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.8rem; font-weight: 600;
      }

      /* ---------- Sensor Services row ---------- */
      .phx-section-title { font-size: 1.6rem; font-weight: 700; margin: 4px 0 4px 0; }
      .phx-section-sub { color: #6b7280; font-size: 0.9rem; margin-bottom: 18px; }

      .phx-service-row {
        display: flex; gap: 12px; margin-bottom: 24px;
      }
      .phx-service-card {
        flex: 1;
        padding: 16px 20px;
        border-radius: 12px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        font-weight: 600; color: #6b7280;
        display: flex; align-items: center; justify-content: center; gap: 8px;
        cursor: default;
      }
      .phx-service-card.active {
        background: #ecfdf5; border-color: #6ee7b7; color: #059669;
      }

      /* ---------- mock notice ---------- */
      .phx-mock-notice {
        background: #fef3c7;
        border-left: 3px solid #f59e0b;
        color: #92400e;
        padding: 10px 14px; border-radius: 8px;
        font-size: 0.85rem; margin-bottom: 16px;
      }

      /* ---------- coming-soon placeholder for inner sub-tabs ---------- */
      .coming-soon {
        padding: 3rem; text-align: center; color: #6b7280;
        border: 1px dashed #d1d5db; border-radius: 12px;
        background: #ffffff;
      }
      .coming-soon h3 { color: #374151; margin-bottom: 8px; }

      /* Streamlit tab styling tweak: native tabs (we use them for sub-tabs inside Wearables) */
      button[data-baseweb="tab"] { font-weight: 500; color: #6b7280; }
      button[data-baseweb="tab"][aria-selected="true"] { color: #059669; font-weight: 600; }
      div[data-baseweb="tab-highlight"] { background: #059669 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- top nav (mock of main Phoenix nav) ----------
logo_html = f'<img src="data:image/png;base64,{LOGO_B64}" />' if LOGO_B64 else "🔥"

st.markdown(
    f"""
    <div class="phx-topnav">
        <div class="phx-brand">{logo_html} Phoenix</div>
        <div class="phx-nav-items">
            <span class="phx-nav-item">🏠 Home</span>
            <span class="phx-nav-item">❤️ My Health</span>
            <span class="phx-nav-item">🏃 Exercises</span>
            <span class="phx-nav-item">📈 My Progress</span>
            <span class="phx-nav-item">🩺 My Doctors</span>
            <span class="phx-nav-item">💬 Messages</span>
            <span class="phx-nav-item active">📡 Sensors</span>
            <span class="phx-nav-item">⚙️ Settings</span>
            <span class="phx-nav-item">👤 Profile</span>
        </div>
        <div class="phx-nav-right">
            <div class="phx-avatar">fd</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Small notice so reviewers know the nav above is a mock
st.markdown(
    """
    <div class="phx-mock-notice">
      🧪 This is the standalone <b>Phoenix Twin</b> module. The top nav and Sensor Services tabs above mirror the
      live Phoenix UI for context. The working feature lives under <b>Sensors → Wearables → Simulation</b>.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- Sensor Services section ----------
st.markdown('<div class="phx-section-title">Sensor Services</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="phx-section-sub">Polar Bluetooth pairing is preserved across navigations.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="phx-service-row">
        <div class="phx-service-card">📈 Rehabilitation</div>
        <div class="phx-service-card">❤️ Polar · HR/HRV</div>
        <div class="phx-service-card active">⌚ Wearables</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- Phoenix Wearables sub-tabs ----------
tab_live, tab_replay, tab_ai, tab_sim, tab_compare = st.tabs(
    ["📡  Live", "⏯  Replay", "🧠  AI Analysis", "🎯  Simulation", "⚖️  Compare"]
)


def render_coming_soon(name: str, blurb: str):
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
    from src.ui_simulation import render_simulation_tab
    render_simulation_tab()

with tab_compare:
    render_coming_soon(
        "Compare",
        "Patient vs. ideal movement, side-by-side replay, angle and trajectory overlap."
    )