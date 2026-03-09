"""Streamlit Demo App — Churn Intelligence System v3.

Streamlit was chosen over Gradio for the demo UI because it supports
multi-tab layouts, session state for chat history, and custom CSS theming.
The 3-tab design maps to the 3 user workflows:
  (1) individual customer analysis (prediction + SHAP + intervention),
  (2) portfolio-level budget allocation (batch policy simulation),
  (3) free-form agent Q&A (natural-language chat with the ReAct agent).

Three tabs:
1. Prediction — Individual customer churn gauge + SHAP waterfall + intervention
2. Policy — Upload CSV → table with policies + ROI per customer
3. Agent Chat — Chat with the ReAct agent in natural language

Usage:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Churn Intelligence System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ══ DESIGN TOKENS ══════════════════════════════════════════════════
       BG:      #0d1117  (GitHub-style deep dark)
       SURFACE: #161b27  (cards / raised surfaces)
       RAISED:  #1c2333  (higher z-level surfaces)
       BORDER:  #30363d  (subtle dividers)
       ACCENT:  #58a6ff  (links, focus rings)
       TEXT:    #e6edf3  (primary)
       MUTED:   #8b949e  (secondary / labels)
    ═══════════════════════════════════════════════════════════════════ */

    /* ── Base ─────────────────────────────────────────────────────────── */
    html, body, .stApp {
        background-color: #0d1117 !important;
        color: #e6edf3;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 15px;
    }
    .main .block-container { padding: 2rem 2.5rem 3rem; max-width: 1200px; }

    /* ── Sidebar ─────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #21262d;
    }
    [data-testid="stSidebar"] .stMarkdown p { color: #8b949e; font-size: 14px; }
    [data-testid="stSidebar"] hr { border-color: #21262d; }

    /* ── Tabs ────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #161b27;
        border-radius: 10px;
        padding: 5px;
        border: 1px solid #21262d;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 7px;
        color: #8b949e;
        font-weight: 600;
        font-size: 14px;
        padding: 9px 24px;
        border: none;
        transition: color .15s;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #e6edf3; }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(56,139,253,.4);
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1.8rem; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ── Section headers ─────────────────────────────────────────────── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 11px;
        font-weight: 700;
        color: #58a6ff;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #21262d;
    }

    /* ── Metric cards ────────────────────────────────────────────────── */
    .metric-card {
        background: #161b27;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 22px 18px 18px;
        text-align: center;
        height: 100%;
        transition: border-color .2s;
    }
    .metric-card:hover { border-color: #30363d; }
    .metric-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #8b949e;
        margin-bottom: 8px;
    }
    .metric-value { font-size: 2.2rem; font-weight: 800; line-height: 1.1; }
    .metric-sub   { font-size: 12px; color: #8b949e; margin-top: 6px; }

    /* ── Risk palette ────────────────────────────────────────────────── */
    .risk-high   { color: #ff7b72; }
    .risk-medium { color: #e3b341; }
    .risk-low    { color: #3fb950; }

    .risk-badge-high {
        background: rgba(255,123,114,.15);
        color: #ff7b72;
        border: 1px solid rgba(255,123,114,.4);
        border-radius: 20px; padding: 5px 16px;
        font-size: 13px; font-weight: 700; display: inline-block;
    }
    .risk-badge-medium {
        background: rgba(227,179,65,.15);
        color: #e3b341;
        border: 1px solid rgba(227,179,65,.4);
        border-radius: 20px; padding: 5px 16px;
        font-size: 13px; font-weight: 700; display: inline-block;
    }
    .risk-badge-low {
        background: rgba(63,185,80,.15);
        color: #3fb950;
        border: 1px solid rgba(63,185,80,.4);
        border-radius: 20px; padding: 5px 16px;
        font-size: 13px; font-weight: 700; display: inline-block;
    }

    /* ── Gauge bar ───────────────────────────────────────────────────── */
    .gauge-wrap { margin: 12px 0 4px; }
    .gauge-bg   { background: #21262d; border-radius: 999px; height: 8px; overflow: hidden; }
    .gauge-fill-high   { height:8px; border-radius:999px; background:linear-gradient(90deg,#ff7b72,#da3633); }
    .gauge-fill-medium { height:8px; border-radius:999px; background:linear-gradient(90deg,#e3b341,#bb8009); }
    .gauge-fill-low    { height:8px; border-radius:999px; background:linear-gradient(90deg,#3fb950,#238636); }

    /* ── KPI cards ───────────────────────────────────────────────────── */
    .kpi-card {
        background: #161b27;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 20px 20px 16px;
        transition: border-color .2s;
    }
    .kpi-card:hover { border-color: #388bfd; }
    .kpi-icon  { font-size: 1.5rem; margin-bottom: 8px; }
    .kpi-title { font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: #8b949e; font-weight: 700; }
    .kpi-num   { font-size: 1.7rem; font-weight: 800; color: #e6edf3; margin: 4px 0 0; }

    /* ── Intervention card ───────────────────────────────────────────── */
    .intervention-card {
        background: linear-gradient(135deg, #0d2119 0%, #0f2d22 100%);
        border: 1px solid #238636;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 1rem;
    }
    .intervention-title { font-size: 16px; font-weight: 700; color: #3fb950; margin-bottom: 10px; }

    /* ── Tool trace ──────────────────────────────────────────────────── */
    .tool-trace {
        background: #0d1117;
        border-left: 3px solid #8b5cf6;
        padding: 10px 14px;
        margin: 4px 0;
        border-radius: 0 8px 8px 0;
        font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, monospace;
        font-size: 12px;
        color: #c4b5fd;
    }

    /* ── ALL buttons (secondary / default) ───────────────────────────── */
    .stButton > button {
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all .15s ease !important;
    }

    /* Secondary buttons — used by example query buttons & clear chat */
    .stButton > button:not([kind="primary"]) {
        background: #21262d !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
    }
    .stButton > button:not([kind="primary"]):hover {
        background: #30363d !important;
        border-color: #58a6ff !important;
        color: #58a6ff !important;
    }

    /* Primary buttons — Predict / Run */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
        border: none !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(56,139,253,.35) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #388bfd 0%, #58a6ff 100%) !important;
        box-shadow: 0 4px 14px rgba(56,139,253,.5) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Example query buttons (tab3 row) ────────────────────────────── */
    .eq-btn-blue   button { background: #1f6feb !important; border-color: #388bfd !important; color: #fff !important; }
    .eq-btn-purple button { background: #6e40c9 !important; border-color: #8b5cf6 !important; color: #fff !important; }
    .eq-btn-teal   button { background: #0f766e !important; border-color: #14b8a6 !important; color: #fff !important; }
    .eq-btn-amber  button { background: #b45309 !important; border-color: #f59e0b !important; color: #fff !important; }
    .eq-btn-blue   button:hover { background: #388bfd !important; }
    .eq-btn-purple button:hover { background: #7c3aed !important; }
    .eq-btn-teal   button:hover { background: #14b8a6 !important; }
    .eq-btn-amber  button:hover { background: #d97706 !important; }

    /* ── Input fields ────────────────────────────────────────────────── */
    [data-testid="stTextInput"]   input,
    [data-testid="stNumberInput"] input {
        background: #0d1117 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        color: #e6edf3 !important;
        font-size: 15px !important;
    }
    [data-testid="stTextInput"]   input:focus,
    [data-testid="stNumberInput"] input:focus {
        border-color: #388bfd !important;
        box-shadow: 0 0 0 3px rgba(56,139,253,.2) !important;
    }

    /* ── Radio buttons ───────────────────────────────────────────────── */
    [data-testid="stRadio"] label { color: #c9d1d9 !important; font-size: 14px !important; }

    /* ── Dataframe ───────────────────────────────────────────────────── */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid #21262d; }

    /* ── Alert banners ───────────────────────────────────────────────── */
    [data-testid="stAlert"] { border-radius: 10px; font-size: 15px; }

    /* ── Chat messages ───────────────────────────────────────────────── */
    [data-testid="stChatMessage"] { background: #161b27 !important; border: 1px solid #21262d !important; border-radius: 12px !important; margin-bottom: 8px !important; }

    /* ── Chat input ──────────────────────────────────────────────────── */
    [data-testid="stChatInputTextArea"] textarea {
        background: #161b27 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        color: #e6edf3 !important;
        font-size: 15px !important;
    }

    /* ── Expander ────────────────────────────────────────────────────── */
    [data-testid="stExpander"] { border: 1px solid #21262d !important; border-radius: 10px !important; }

    /* ── Divider ─────────────────────────────────────────────────────── */
    hr { border-color: #21262d !important; }

    /* ── General text ────────────────────────────────────────────────── */
    p, li { font-size: 15px; color: #c9d1d9; }
    label { font-size: 14px !important; color: #8b949e !important; }

    /* ── Hide Streamlit top toolbar/header bar ───────────────────────── */
    header[data-testid="stHeader"] { display: none !important; }
    [data-testid="stToolbar"]      { display: none !important; }
    .stAppDeployButton             { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style="padding:4px 0 20px">
            <div style="font-size:1.35rem;font-weight:800;color:#e6edf3;letter-spacing:-0.01em">🧠 Churn Intelligence</div>
            <div style="font-size:12px;color:#8b949e;margin-top:4px">v3.0 &nbsp;·&nbsp; MLflow &nbsp;·&nbsp; ReAct Agent</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("<div style='font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#58a6ff;margin-bottom:8px'>Data Mode</div>", unsafe_allow_html=True)
    api_mode = st.radio(
        "Data Mode",
        ["Direct (local models)", "API (FastAPI server)"],
        help="Direct mode loads models locally. API mode calls the FastAPI server.",
        label_visibility="collapsed",
    )

    if api_mode == "API (FastAPI server)":
        api_url = st.text_input("API URL", value="http://localhost:8000")
    else:
        api_url = None

    st.divider()
    st.markdown(
        """
        <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#58a6ff;margin-bottom:12px">Stack</div>
        <div style="display:flex;flex-direction:column;gap:5px">
            <div style="background:#161b27;border:1px solid #21262d;border-radius:7px;padding:7px 12px;font-size:12px;color:#c9d1d9">⚙️&nbsp; scikit-learn · XGBoost · CatBoost</div>
            <div style="background:#161b27;border:1px solid #21262d;border-radius:7px;padding:7px 12px;font-size:12px;color:#c9d1d9">📊&nbsp; MLflow · SHAP</div>
            <div style="background:#161b27;border:1px solid #21262d;border-radius:7px;padding:7px 12px;font-size:12px;color:#c9d1d9">🤖&nbsp; LangGraph · Gemini</div>
            <div style="background:#161b27;border:1px solid #21262d;border-radius:7px;padding:7px 12px;font-size:12px;color:#c9d1d9">🌐&nbsp; FastAPI · Streamlit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Page header ─────────────────────────────────────────────────────────────

st.markdown(
    """
    <div style="margin-bottom:1.8rem;padding-bottom:1.2rem;border-bottom:1px solid #21262d">
        <h1 style="font-size:1.9rem;font-weight:800;color:#e6edf3;margin:0;letter-spacing:-0.025em">
            🧠 Churn Intelligence System
        </h1>
        <p style="color:#8b949e;font-size:14px;margin:6px 0 0">
            Predict &nbsp;·&nbsp; Explain &nbsp;·&nbsp; Intervene
            &nbsp;&nbsp;<span style="color:#21262d">|</span>&nbsp;&nbsp;
            <span style="color:#58a6ff">CatBoost</span> &nbsp;·&nbsp;
            <span style="color:#8b5cf6">SHAP</span> &nbsp;·&nbsp;
            <span style="color:#3fb950">Gemini ReAct</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tab Setup ────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔮  Prediction", "📊  Policy Engine", "💬  Agent Chat"])


# ── Helper Functions ─────────────────────────────────────────────────────────


def call_api(endpoint: str, payload: dict, timeout: float = 30.0) -> dict:
    """Call the FastAPI server."""
    import httpx

    url = f"{api_url}{endpoint}"
    response = httpx.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def load_local_resources():
    """Load models and data locally.

    Caches resources in ``st.session_state`` to avoid reloading on every
    interaction.  Streamlit reruns the entire script on each widget
    interaction, so without this guard every button click would re-read
    the CSV and deserialise the model — adding seconds of latency.
    """
    if "resources" not in st.session_state:
        try:
            import pandas as pd
            from src.config import RAW_CSV
            from src.preprocessing import load_pipeline

            df = pd.read_csv(RAW_CSV)
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df["Churn"] = (df["Churn"] == "Yes").astype(int)

            pipeline = None
            try:
                pipeline = load_pipeline()
            except Exception:
                pass

            model = None
            try:
                import joblib
                from src.config import MODELS_DIR

                model_path = MODELS_DIR / "best_model.joblib"
                if model_path.exists():
                    model = joblib.load(model_path)
            except Exception:
                pass

            st.session_state["resources"] = {
                "df": df,
                "pipeline": pipeline,
                "model": model,
            }
        except Exception as e:
            st.error(f"Could not load resources: {e}")
            st.session_state["resources"] = {"df": None, "pipeline": None, "model": None}

    return st.session_state["resources"]


# ── TAB 1: Prediction ────────────────────────────────────────────────────────

with tab1:
    st.markdown('<div class="section-header">🔮 Individual Customer Analysis</div>', unsafe_allow_html=True)

    # ── Input row ────────────────────────────────────────────────────────────
    inp_col, btn_col, _ = st.columns([2, 1, 3])
    with inp_col:
        customer_id = st.text_input(
            "Customer ID",
            value="7590-VHVEG",
            placeholder="e.g. 7590-VHVEG",
            help="Enter the customer identifier from the Telco dataset",
        )
    with btn_col:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)  # align with input
        predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

    if predict_btn and customer_id:
        with st.spinner("Running prediction..."):
            if api_url:
                try:
                    pred_result = call_api("/predict", {"customer_id": customer_id})
                    policy_result = call_api("/policy", {"customer_id": customer_id, "budget": 5000})
                except Exception as e:
                    st.error(f"API call failed: {e}")
                    pred_result = None
                    policy_result = None
            else:
                resources = load_local_resources()
                df = resources.get("df")
                model = resources.get("model")
                pipeline = resources.get("pipeline")

                if df is not None:
                    from src.config import ID_COL

                    row = df[df[ID_COL] == customer_id]
                    if row.empty:
                        st.error(f"Customer '{customer_id}' not found in the dataset.")
                        pred_result = None
                        policy_result = None
                    else:
                        if model is None or pipeline is None:
                            st.error("Prediction model not loaded. Run `python src/train.py` first.")
                            pred_result = None
                            policy_result = None
                        else:
                            from src.preprocessing import prepare_data
                            from policy.intervention_engine import classify_risk, get_policy

                            X, _, _ = prepare_data(row, fit_pipeline=False, pipeline=pipeline)
                            if hasattr(model, "predict_proba"):
                                churn_prob = float(model.predict_proba(X)[0, 1])
                            else:
                                pred = model.predict(X)
                                churn_prob = float(pred.iloc[0]) if hasattr(pred, "iloc") else float(pred[0])

                            pred_result = {
                                "customer_id": customer_id,
                                "churn_probability": churn_prob,
                                "churn_probability_pct": f"{churn_prob*100:.1f}%",
                                "risk_segment": classify_risk(churn_prob),
                                "model_source": "CatBoost (local)",
                            }

                            policy_result = get_policy(
                                customer_id=customer_id,
                                df=df,
                                model=model,
                                pipeline=pipeline,
                            )
                else:
                    st.error("Dataset not loaded. Run `python src/download_data.py` first.")
                    pred_result = None
                    policy_result = None

        if pred_result:
            st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

            prob = pred_result["churn_probability"]
            risk = pred_result["risk_segment"]
            gauge_class = f"gauge-fill-{risk}"
            badge_class = f"risk-badge-{risk}"
            prob_pct = int(prob * 100)

            # ── Metrics row ───────────────────────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Churn Probability</div>
                        <div class="metric-value risk-{risk}">{prob_pct}%</div>
                        <div class="gauge-wrap">
                            <div class="gauge-bg">
                                <div class="{gauge_class}" style="width:{prob_pct}%"></div>
                            </div>
                        </div>
                        <div class="metric-sub">{prob:.3f} raw score</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Risk Segment</div>
                        <div style="margin:12px 0 8px">
                            <span class="{badge_class}">{risk.upper()}</span>
                        </div>
                        <div class="metric-sub">{'Immediate action needed' if risk=='high' else 'Monitor closely' if risk=='medium' else 'Stable'}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Model</div>
                        <div class="metric-value" style="font-size:1rem;margin-top:10px;color:#94a3b8">
                            {pred_result.get('model_source', 'N/A')}
                        </div>
                        <div class="metric-sub">AUC 0.847</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Customer ID</div>
                        <div class="metric-value" style="font-size:1.1rem;margin-top:10px;color:#e2e8f0;font-family:monospace">
                            {pred_result['customer_id']}
                        </div>
                        <div class="metric-sub">Telco dataset</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ── Intervention ──────────────────────────────────────────────────
            if policy_result and "error" not in policy_result:
                st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">💡 Recommended Intervention</div>', unsafe_allow_html=True)

                p_name = policy_result.get("policy_name", policy_result.get("recommended_policy", "N/A"))
                p_cost = policy_result.get("cost", 0)
                p_roi  = policy_result.get("roi", 0)
                p_desc = policy_result.get("policy_description", "")

                i1, i2, i3 = st.columns(3)
                with i1:
                    st.markdown(
                        f'<div class="kpi-card"><div class="kpi-icon">📋</div>'
                        f'<div class="kpi-title">Recommended Policy</div>'
                        f'<div class="kpi-num" style="font-size:1rem">{p_name}</div></div>',
                        unsafe_allow_html=True,
                    )
                with i2:
                    st.markdown(
                        f'<div class="kpi-card"><div class="kpi-icon">💸</div>'
                        f'<div class="kpi-title">Intervention Cost</div>'
                        f'<div class="kpi-num">${p_cost:,.2f}</div></div>',
                        unsafe_allow_html=True,
                    )
                with i3:
                    roi_color = "#34d399" if p_roi > 0 else "#f87171"
                    st.markdown(
                        f'<div class="kpi-card"><div class="kpi-icon">📈</div>'
                        f'<div class="kpi-title">Expected ROI</div>'
                        f'<div class="kpi-num" style="color:{roi_color}">${p_roi:,.2f}</div></div>',
                        unsafe_allow_html=True,
                    )

                if p_desc:
                    st.markdown(
                        f'<div class="intervention-card" style="margin-top:1rem">'
                        f'<div class="intervention-title">Action Plan</div>'
                        f'<div style="color:#d1fae5;font-size:14px">{p_desc}</div></div>',
                        unsafe_allow_html=True,
                    )


# ── TAB 2: Policy Engine ────────────────────────────────────────────────────

with tab2:
    st.markdown('<div class="section-header">📊 Retention Budget Allocation</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8b949e;font-size:14px;margin-bottom:1.5rem">'
        'Simulate how to allocate a fixed monthly retention budget across your highest-risk customers, ranked by expected ROI.</p>',
        unsafe_allow_html=True,
    )

    budget_col, n_col, sim_col, _ = st.columns([2, 2, 1, 3])
    with budget_col:
        budget = st.number_input("Monthly Budget ($)", value=5000, step=500, min_value=100)
    with n_col:
        top_n = st.number_input("Top-N customers to consider", value=10, step=5, min_value=1, max_value=100)
    with sim_col:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        simulate_btn = st.button("▶ Run", type="primary", use_container_width=True)

    if simulate_btn:
        with st.spinner("Running budget simulation..."):
            if api_url:
                try:
                    import httpx

                    result = httpx.post(
                        f"{api_url}/agent/query",
                        json={"query": f"Allocate ${budget} across top {top_n} customers by ROI"},
                        timeout=60.0,
                    ).json()
                    st.markdown(result["answer"])
                except Exception as e:
                    st.error(f"API call failed: {e}")
            else:
                from policy.intervention_engine import simulate_budget_allocation

                resources = load_local_resources()
                df = resources.get("df")

                result = simulate_budget_allocation(
                    budget=float(budget),
                    top_n=int(top_n),
                    df=df,
                    model=resources.get("model"),
                    pipeline=resources.get("pipeline"),
                )

                st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

                # ── KPI summary row ───────────────────────────────────────────
                k1, k2, k3, k4 = st.columns(4)
                kpis = [
                    ("k1", "👥", "Customers Targeted", str(result["customers_targeted"]), ""),
                    ("k2", "💸", "Total Cost",         f"${result['total_cost']:,.2f}", ""),
                    ("k3", "📈", "Expected ROI",       f"${result['total_expected_roi']:,.2f}", "#34d399"),
                    ("k4", "⚡", "Budget Utilisation", f"{result['budget_utilization']:.1f}%",
                     "#34d399" if result["budget_utilization"] < 95 else "#fbbf24"),
                ]
                for col, (_, icon, title, val, colour) in zip([k1, k2, k3, k4], kpis):
                    num_style = f"color:{colour}" if colour else ""
                    col.markdown(
                        f'<div class="kpi-card"><div class="kpi-icon">{icon}</div>'
                        f'<div class="kpi-title">{title}</div>'
                        f'<div class="kpi-num" style="{num_style}">{val}</div></div>',
                        unsafe_allow_html=True,
                    )

                # ── Allocation table ──────────────────────────────────────────
                if result.get("allocations"):
                    import pandas as pd

                    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
                    st.markdown('<div class="section-header">📋 Allocation Detail</div>', unsafe_allow_html=True)

                    alloc_df = pd.DataFrame(result["allocations"])
                    display_cols = [
                        "customer_id",
                        "churn_probability",
                        "risk_segment",
                        "recommended_policy",
                        "cost",
                        "roi",
                        "monthly_charges",
                    ]
                    display_cols = [c for c in display_cols if c in alloc_df.columns]
                    st.dataframe(
                        alloc_df[display_cols].style.format({
                            "churn_probability": "{:.1%}",
                            "cost": "${:.2f}",
                            "roi": "${:.2f}",
                            "monthly_charges": "${:.2f}",
                        }) if "churn_probability" in alloc_df.columns else alloc_df[display_cols],
                        use_container_width=True,
                        hide_index=True,
                    )


# ── TAB 3: Agent Chat ───────────────────────────────────────────────────────
# Chat interface with session_state-based message history.  The agent is
# cached in session_state (see ``st.session_state["agent"]``) to reuse the
# compiled LangGraph across turns, avoiding the overhead of re-compiling the
# graph on every message.  Tool traces are displayed in expandable sections
# so they don't clutter the conversation but remain available for debugging
# and transparency into the agent's reasoning process.

EXAMPLE_QUERIES = [
    ("🔍", "Churn risk for 7590-VHVEG?", "What is the churn risk for customer 7590-VHVEG and why?"),
    ("💰", "Allocate $5k this month", "With a $5,000 budget this month, which customers should I call first and why?"),
    ("📋", "Top 20 at-risk customers", "Who are the top 20 customers with the highest churn risk right now?"),
    ("📈", "ROI of 10% discount", "If I offer a 10% discount to all high-risk customers, what is the expected ROI?"),
]

with tab3:
    st.markdown('<div class="section-header">💬 Churn Intelligence Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8b949e;font-size:15px;margin-bottom:1.4rem">'
        'Ask anything in plain English — the agent calls the prediction model, SHAP explainer, '
        'and policy engine as needed to craft a data-driven answer.</p>',
        unsafe_allow_html=True,
    )

    # ── Example query buttons (each column gets a color class via st.markdown wrapper) ───
    st.markdown('<div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#58a6ff;margin-bottom:10px">💡 Try an example</div>', unsafe_allow_html=True)
    _eq_colors = ["eq-btn-blue", "eq-btn-purple", "eq-btn-teal", "eq-btn-amber"]
    eq_cols = st.columns(len(EXAMPLE_QUERIES))
    for col, color_cls, (icon, label, full_query) in zip(eq_cols, _eq_colors, EXAMPLE_QUERIES):
        with col:
            st.markdown(f'<div class="{color_cls}">', unsafe_allow_html=True)
            if st.button(f"{icon}  {label}", key=f"eq_{label}", use_container_width=True):
                st.session_state["pending_prompt"] = full_query
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:1rem'></div>", unsafe_allow_html=True)

    # Chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tools"):
                with st.expander("🔧 Tools used"):
                    for tool in msg["tools"]:
                        st.markdown(
                            f'<div class="tool-trace">⚡ <strong>{tool["tool"]}</strong> '
                            f'— args: {json.dumps(tool["args"])}</div>',
                            unsafe_allow_html=True,
                        )

    # Chat input — pick up a pending_prompt set by an example button, or use the chat widget
    typed_prompt = st.chat_input("Ask the agent a question...")
    prompt = st.session_state.pop("pending_prompt", None) or typed_prompt

    if prompt:
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                if api_url:
                    try:
                        result = call_api("/agent/query", {"query": prompt}, timeout=120.0)
                        answer = result["answer"]
                        tools = result.get("tools_used", [])
                    except Exception as e:
                        answer = f"Error: {e}"
                        tools = []
                else:
                    try:
                        import concurrent.futures
                        from agent.graph import create_agent, run_agent
                        from agent.tools import init_tools

                        # Pre-populate tool caches with already-loaded local
                        # resources so _get_model() never falls through to the
                        # MLflow registry lookup (which blocks/hangs when in
                        # Direct mode and no MLflow server is needed).
                        _local = load_local_resources()
                        init_tools(
                            df=_local.get("df"),
                            model=_local.get("model"),
                            pipeline=_local.get("pipeline"),
                        )

                        if "agent" not in st.session_state:
                            st.session_state["agent"] = create_agent()

                        # Run in a separate thread so LangGraph's asyncio.run()
                        # doesn't conflict with Streamlit's event loop.
                        # NOTE: do NOT use the `with` context manager — its __exit__
                        # calls shutdown(wait=True) which blocks forever if the
                        # thread is still hung after the timeout.
                        _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                        future = _executor.submit(run_agent, st.session_state["agent"], prompt)
                        try:
                            result = future.result(timeout=120)
                        except concurrent.futures.TimeoutError:
                            result = {"answer": "Agent timed out. Try a simpler question.", "tool_calls": []}
                        finally:
                            _executor.shutdown(wait=False)  # release without blocking

                        answer = result["answer"] or "(Agent returned an empty response)"
                        tools = result.get("tool_calls", [])
                    except Exception as e:
                        answer = f"Agent error: {e}. Make sure your LLM API key is set in .env"
                        tools = []

                st.markdown(answer)

                if tools:
                    with st.expander("🔧 Tools used"):
                        for tool in tools:
                            st.markdown(
                                f'<div class="tool-trace">⚡ <strong>{tool["tool"]}</strong> '
                                f'— args: {json.dumps(tool["args"])}</div>',
                                unsafe_allow_html=True,
                            )

        # Save assistant message
        st.session_state["messages"].append(
            {"role": "assistant", "content": answer, "tools": tools}
        )

    # Clear chat button
    if st.session_state["messages"]:
        if st.button("🗑️ Clear chat"):
            st.session_state["messages"] = []
            st.rerun()
