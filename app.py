import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from config import ARTIFACTS_DIR, SAFETY_THRESHOLDS, TARGETS
from data.pipeline import run as run_pipeline
from models.predict import forecast, load_models
from llm.advisor import TelemetryAdvisoryEngine

st.set_page_config(
    page_title="Water Quality Telemetry & Forecast System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, pre, .mono-text {
    font-family: 'JetBrains Mono', monospace !important;
}

.stApp {
    background-color: #0f172a;
    color: #e2e8f0;
}

/* Header & station bar */
.station-header {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.station-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.01em;
}

.station-meta {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 2px;
}

.status-tag {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.status-nominal { background: #064e3b; color: #34d399; border: 1px solid #059669; }
.status-caution { background: #451a03; color: #fbbf24; border: 1px solid #d97706; }
.status-alert   { background: #450a0a; color: #f87171; border: 1px solid #dc2626; }

/* Control cards */
.telemetry-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 14px;
    margin-bottom: 12px;
}

/* Console logs */
.console-box {
    background: #090d16;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.84rem;
    color: #cbd5e1;
    line-height: 1.5;
}

/* Section styling */
h2, h3 {
    color: #f1f5f9 !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
}

hr {
    border-color: #334155 !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_telemetry_series() -> pd.DataFrame:
    return run_pipeline()


@st.cache_resource(show_spinner=False)
def load_forecast_regressors():
    artifacts = Path(ARTIFACTS_DIR)
    if not artifacts.exists() or not any(artifacts.iterdir()):
        from models.train import run as train_run
        train_run()
    return load_models()


@st.cache_resource
def get_advisory_engine() -> TelemetryAdvisoryEngine:
    return TelemetryAdvisoryEngine()


def evaluate_parameter_state(param: str, value: float) -> Tuple[str, str]:
    spec = SAFETY_THRESHOLDS.get(param, {})
    if not spec:
        return "UNKNOWN", "status-nominal"
    if value >= spec["safe"]:
        return "ALERT", "status-alert"
    if value >= spec["caution"]:
        return "CAUTION", "status-caution"
    return "NOMINAL", "status-nominal"


def render_gauge_chart(param: str, value: float) -> go.Figure:
    spec = SAFETY_THRESHOLDS[param]
    safe_limit = spec["safe"]
    axis_max = safe_limit * 1.4

    if value >= safe_limit:
        bar_color = "#ef4444"
    elif value >= spec["caution"]:
        bar_color = "#f59e0b"
    else:
        bar_color = "#10b981"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={
            "text": f"<b>{spec['name']}</b><br><span style='font-size:11px;color:#94a3b8;'>Limit: {safe_limit} {spec['unit']}</span>",
            "font": {"color": "#f8fafc", "size": 13},
        },
        number={"font": {"color": "#f8fafc", "size": 26}, "suffix": f" {spec['unit']}"},
        gauge={
            "axis": {
                "range": [0, axis_max],
                "tickcolor": "#475569",
                "tickfont": {"color": "#64748b", "size": 10},
            },
            "bar": {"color": bar_color, "thickness": 0.25},
            "bgcolor": "#1e293b",
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, spec["caution"]], "color": "rgba(16, 185, 129, 0.08)"},
                {"range": [spec["caution"], spec["safe"]], "color": "rgba(245, 158, 11, 0.12)"},
                {"range": [spec["safe"], axis_max], "color": "rgba(239, 68, 68, 0.15)"},
            ],
            "threshold": {
                "line": {"color": "#ef4444", "width": 2},
                "thickness": 0.8,
                "value": safe_limit,
            },
        },
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=15, r=15, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_forecast_trend(
    param: str,
    hist_time: list,
    hist_vals: list,
    pred_time: list,
    pred_vals: list,
) -> go.Figure:
    spec = SAFETY_THRESHOLDS.get(param, {})
    unit = spec.get("unit", "")
    alert_val = spec.get("safe")
    caution_val = spec.get("caution")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist_time,
        y=hist_vals,
        name="Observed",
        line=dict(color="#38bdf8", width=1.8),
        mode="lines",
    ))

    fig.add_trace(go.Scatter(
        x=pred_time,
        y=pred_vals,
        name="Forecast (6h)",
        line=dict(color="#c084fc", width=2, dash="dash"),
        mode="lines+markers",
        marker=dict(size=5, color="#c084fc"),
    ))

    if caution_val:
        fig.add_hline(
            y=caution_val,
            line_dash="dot",
            line_color="#f59e0b",
            annotation_text=f"Caution ({caution_val} {unit})",
            annotation_font_color="#f59e0b",
            annotation_font_size=10,
        )
    if alert_val:
        fig.add_hline(
            y=alert_val,
            line_dash="dot",
            line_color="#ef4444",
            annotation_text=f"Alert Limit ({alert_val} {unit})",
            annotation_font_color="#ef4444",
            annotation_font_size=10,
        )

    fig.update_layout(
        title=dict(text=f"{spec['name']} Telemetry & Model Projection", font=dict(color="#f8fafc", size=13)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1e293b",
        font=dict(color="#94a3b8", size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1", size=10)),
        xaxis=dict(gridcolor="#334155", tickfont=dict(color="#64748b")),
        yaxis=dict(gridcolor="#334155", tickfont=dict(color="#64748b"), title=f"{param} ({unit})"),
        height=260,
        margin=dict(l=35, r=25, t=45, b=25),
    )
    return fig


if "interaction_history" not in st.session_state:
    st.session_state.interaction_history = []
if "forecast_cache" not in st.session_state:
    st.session_state.forecast_cache = None
if "current_readings_cache" not in st.session_state:
    st.session_state.current_readings_cache = None


df_telemetry = load_telemetry_series()
models_dict = load_forecast_regressors()
advisory_service = get_advisory_engine()


with st.sidebar:
    st.markdown("### Telemetry Controls")
    sim_index = st.slider(
        "Historical Time Index",
        min_value=100,
        max_value=len(df_telemetry) - 1,
        value=min(8500, len(df_telemetry) - 1),
        step=24,
        help="Replay telemetry feed at selected historical index",
    )

    recalculate_forecast = st.button("Compute Forecast Horizon", use_container_width=True)
    clear_session = st.button("Reset Operational Console", use_container_width=True)

    if clear_session:
        st.session_state.interaction_history = []
        advisory_service.reset()

    st.markdown("---")
    st.markdown("### Guideline Specifications")
    for key, spec in SAFETY_THRESHOLDS.items():
        st.caption(f"**{spec['name']}**")
        st.markdown(
            f"- Caution: `{spec['caution']} {spec['unit']}`\n"
            f"- Alert Limit: `{spec['safe']} {spec['unit']}`"
        )


df_window = df_telemetry.iloc[max(0, sim_index - 168):sim_index].copy()
current_row = df_telemetry.iloc[sim_index]

current_telemetry = {
    "NO3": float(current_row["NO3"]),
    "Turbidity": float(current_row["Turbidity"]),
    "Temp": float(current_row.get("Temp", 25.0)),
    "Conductivity": float(current_row.get("Conductivity", 0.0)),
}

if recalculate_forecast or st.session_state.forecast_cache is None:
    preds = forecast(df_window, models_dict, steps=6)
    st.session_state.forecast_cache = preds
    st.session_state.current_readings_cache = current_telemetry

forecast_output = st.session_state.forecast_cache
current_telemetry = st.session_state.current_readings_cache or current_telemetry


timestamp_str = pd.to_datetime(current_row["Timestamp"]).strftime("%Y-%m-%d %H:%M:%S UTC")
st.markdown(f"""
<div class="station-header">
    <div>
        <div class="station-title">Station: Johnstone River - Coquette Point (ID: JR-CP-01)</div>
        <div class="station-meta">Coordinates: 17.525° S, 146.042° E | Sampling: 1-hour interval | Sensor Quality: Operational</div>
    </div>
    <div style="text-align:right;">
        <div class="station-meta">Telemetry Timestamp</div>
        <div class="mono-text" style="color:#f8fafc;font-weight:600;font-size:0.95rem;">{timestamp_str}</div>
    </div>
</div>
""", unsafe_allow_html=True)


st.subheader("Current Telemetry Acquisition")

gauge_cols = st.columns(3)
monitored_params = ["NO3", "Turbidity", "Temp"]

for col, param in zip(gauge_cols, monitored_params):
    reading = current_telemetry.get(param, 0.0)
    status_label, status_class = evaluate_parameter_state(param, reading)
    with col:
        st.plotly_chart(render_gauge_chart(param, reading), use_container_width=True, key=f"gauge_{param}")
        st.markdown(
            f"<div style='text-align:center;margin-top:-10px;margin-bottom:12px;'><span class='status-tag {status_class}'>{status_label}</span></div>",
            unsafe_allow_html=True,
        )

metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("Conductivity", f"{current_telemetry.get('Conductivity', 0.0):.1f} µS/cm")
with metric_cols[1]:
    st.metric("7-Day Trend Window", f"{len(df_window)} hours")
with metric_cols[2]:
    st.metric("Engine Status", "Online" if advisory_service.is_llm_available else "Offline Fallback")
with metric_cols[3]:
    st.metric("Forecast Model", "XGBoost (Autoregressive)")

st.markdown("---")


st.subheader("Hydrological Forecasting & Excursion Assessment")

if forecast_output:
    chart_cols = st.columns(2)
    forecast_timestamps = forecast_output.get("timestamps", [])
    history_timestamps = df_window["Timestamp"].astype(str).tolist()[-48:]

    for idx, target_param in enumerate(TARGETS):
        history_series = df_window[target_param].tolist()[-48:]
        predicted_series = forecast_output.get(target_param, [])

        with chart_cols[idx % 2]:
            fig = render_forecast_trend(
                target_param,
                history_timestamps,
                history_series,
                forecast_timestamps,
                predicted_series,
            )
            st.plotly_chart(fig, use_container_width=True, key=f"plot_{target_param}")

            if forecast_timestamps and predicted_series:
                spec = SAFETY_THRESHOLDS.get(target_param, {})
                table_rows = []
                for ts_item, p_val in zip(forecast_timestamps, predicted_series):
                    s_label, _ = evaluate_parameter_state(target_param, p_val)
                    table_rows.append({
                        "Horizon Timestamp": pd.to_datetime(ts_item).strftime("%H:%M UTC"),
                        f"{target_param} ({spec.get('unit', '')})": round(p_val, 3),
                        "Classification": s_label,
                    })
                df_tab = pd.DataFrame(table_rows)
                st.dataframe(df_tab, use_container_width=True, hide_index=True)


st.markdown("---")


st.subheader("Operational Decision Support Console")
st.caption("Automated telemetry reasoning engine grounded in sensor baselines and forecast models.")

for item in st.session_state.interaction_history:
    if item["role"] == "user":
        st.markdown(f"**Operator Query:** `{item['content']}`")
    else:
        st.markdown(f"""
        <div class="console-box">
            {item['content']}
        </div>
        <br>
        """, unsafe_allow_html=True)

query_cols = st.columns(4)
operational_queries = [
    ("Summary Assessment", "Provide an operational status summary across all monitored parameters."),
    ("Horizon Threshold Check", "Are any parameter excursions predicted over the next 6 hours?"),
    ("Agricultural Safety", "Evaluate suitability for agricultural catchment and irrigation intake."),
    ("Recreational Advisory", "State safety determination for downstream public and recreational zones."),
]

for col, (btn_title, prompt_text) in zip(query_cols, operational_queries):
    with col:
        if st.button(btn_title, use_container_width=True, key=f"btn_{btn_title}"):
            with st.spinner("Evaluating telemetry context..."):
                response_text = advisory_service.evaluate(prompt_text, current_telemetry, forecast_output)
            st.session_state.interaction_history.append({"role": "user", "content": prompt_text})
            st.session_state.interaction_history.append({"role": "assistant", "content": response_text})
            st.rerun()

with st.form("operator_inquiry_form", clear_on_submit=True):
    col_input, col_submit = st.columns([5, 1])
    with col_input:
        inquiry_text = st.text_input("Operational inquiry:", placeholder="Enter technical inquiry or operational question...", label_visibility="collapsed")
    with col_submit:
        submitted_inquiry = st.form_submit_button("Submit", use_container_width=True)

    if submitted_inquiry and inquiry_text.strip():
        with st.spinner("Processing inquiry against telemetry matrix..."):
            advisory_eval = advisory_service.evaluate(inquiry_text, current_telemetry, forecast_output)
        st.session_state.interaction_history.append({"role": "user", "content": inquiry_text})
        st.session_state.interaction_history.append({"role": "assistant", "content": advisory_eval})
        st.rerun()


st.markdown("---")
st.subheader("Model Validation & Feature Importance")

metrics_filepath = Path(f"{ARTIFACTS_DIR}/metrics.json")
if metrics_filepath.exists():
    with open(metrics_filepath, encoding="utf-8") as mf:
        metrics_data = json.load(mf)

    perf_cols = st.columns(len(metrics_data))
    for col, (target_name, perf_info) in zip(perf_cols, metrics_data.items()):
        scores = perf_info["metrics"]
        with col:
            st.markdown(f"**Target Parameter: {target_name}**")
            sub_m_cols = st.columns(3)
            sub_m_cols[0].metric("RMSE", f"{scores['rmse']:.4f}")
            sub_m_cols[1].metric("MAE", f"{scores['mae']:.4f}")
            sub_m_cols[2].metric("R² Score", f"{scores['r2']:.4f}")

            importance_dict = perf_info.get("importances", {})
            if importance_dict:
                imp_df = pd.DataFrame(
                    list(importance_dict.items())[:8],
                    columns=["Predictor", "Importance Gain"],
                )
                fig_imp = px.bar(
                    imp_df,
                    x="Importance Gain",
                    y="Predictor",
                    orientation="h",
                    title=f"Relative Feature Contribution ({target_name})",
                    color_discrete_sequence=["#38bdf8"],
                )
                fig_imp.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#1e293b",
                    font=dict(color="#94a3b8", size=10),
                    yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
                    xaxis=dict(gridcolor="#334155"),
                    height=240,
                    margin=dict(l=15, r=15, t=35, b=20),
                )
                st.plotly_chart(fig_imp, use_container_width=True, key=f"imp_{target_name}")

st.markdown("---")
st.caption("Environmental Telemetry Monitoring & Forecast System · Station ID: JR-CP-01")
