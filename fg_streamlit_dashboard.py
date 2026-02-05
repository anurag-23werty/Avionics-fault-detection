import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Use the OperationV2-coupled logic from flightgearV3 (smoothed confidence + error_threshold).
from flightgearV3 import (
    HOST,
    PORT,
    UPDATE_RATE,
    connect,
    get_property,
    detect_flight_phase,
    load_models,
    init_buffers,
    ewma,
    forward_fill,
    W1,
    W2,
    ALPHA_CONF,
    RESIDUAL_SCALE,
)
from kalman import Kalman1D


SENSORS = {
    "airspeed_kt": "/velocities/airspeed-kt",
    "groundspeed_kt": "/velocities/groundspeed-kt",
    "altitude_ft": "/position/altitude-ft",
    "ground_elev_ft": "/position/ground-elev-ft",
    "vertical_speed_fps": "/velocities/vertical-speed-fps",
    "temp_c": "/environment/temperature-degc",
    "wind_speed_kt": "/environment/wind-speed-kt",
    "wind_dir_deg": "/environment/wind-from-heading-deg",
    "pitch_deg": "/orientation/pitch-deg",
    "roll_deg": "/orientation/roll-deg",
    "heading_deg": "/orientation/heading-deg",
}


def _css():
    st.markdown(
        """
        <style>
          /* Dark + neon green vibe */
          .stApp { background: radial-gradient(1200px 700px at 15% 10%, #0d1a17 0%, #070a0a 55%, #050707 100%); }
          html, body, [class*="css"]  { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
          .block-container { padding-top: 1.2rem; }
          h1, h2, h3, h4 { letter-spacing: -0.02em; }

          /* Metric cards */
          div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(7,12,12,0.92), rgba(5,10,10,0.92));
            border: 1px solid rgba(0, 255, 175, 0.14);
            border-radius: 14px;
            padding: 14px 14px 10px 14px;
            box-shadow: 0 0 0 1px rgba(0, 255, 175, 0.05), 0 12px 40px rgba(0,0,0,0.35);
          }

          /* Phase pill */
          .phase-pill {
            display: inline-block;
            padding: 8px 12px;
            border-radius: 999px;
            border: 1px solid rgba(0, 255, 175, 0.20);
            background: rgba(0, 255, 175, 0.08);
            color: #b6fff0;
            font-weight: 700;
            letter-spacing: 0.06em;
          }

          .status-card {
            background: linear-gradient(180deg, rgba(7,12,12,0.92), rgba(5,10,10,0.92));
            border: 1px solid rgba(0, 255, 175, 0.14);
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 0 0 1px rgba(0, 255, 175, 0.05), 0 12px 40px rgba(0,0,0,0.35);
          }

          /* Sidebar */
          section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(6,10,10,0.95), rgba(4,6,6,0.95));
            border-right: 1px solid rgba(0, 255, 175, 0.12);
          }

          /* Subtle scanlines */
          .stApp:before {
            content: "";
            position: fixed;
            inset: 0;
            background: repeating-linear-gradient(
              to bottom,
              rgba(0, 255, 175, 0.02),
              rgba(0, 255, 175, 0.02) 1px,
              rgba(0, 0, 0, 0) 2px,
              rgba(0, 0, 0, 0) 6px
            );
            pointer-events: none;
            mix-blend-mode: overlay;
            opacity: 0.30;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _chart(df: pd.DataFrame, y: str, title: str, color: str):
    if df.empty or y not in df.columns:
        return alt.Chart(pd.DataFrame({"t": [], y: []})).mark_line()

    base = (
        alt.Chart(df)
        .mark_line(color=color, strokeWidth=2.2)
        .encode(
            x=alt.X("t:T", title=None),
            y=alt.Y(f"{y}:Q", title=None),
            tooltip=[
                alt.Tooltip("t:T", title="time"),
                alt.Tooltip(f"{y}:Q", format=".4f"),
                alt.Tooltip("phase:N"),
            ],
        )
        .properties(height=240, title=title)
    )

    return (
        base.configure(background="#050707")
        .configure_title(color="#b6fff0", fontSize=14, anchor="start")
        .configure_axis(labelColor="#8bf5d0", titleColor="#8bf5d0", gridColor="#16302b")
        .configure_view(strokeWidth=0)
    )


def _multi_series_chart(df_long: pd.DataFrame, title: str, palette: list[str]):
    if df_long.empty:
        return alt.Chart(pd.DataFrame({"t": [], "value": [], "series": [], "phase": []})).mark_line()

    base = (
        alt.Chart(df_long)
        .mark_line(strokeWidth=2.0)
        .encode(
            x=alt.X("t:T", title=None),
            y=alt.Y("value:Q", title=None),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(range=palette),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[
                alt.Tooltip("t:T", title="time"),
                alt.Tooltip("series:N"),
                alt.Tooltip("value:Q", format=".4f"),
                alt.Tooltip("phase:N"),
            ],
        )
        .properties(height=240, title=title)
    )

    return (
        base.configure(background="#050707")
        .configure_title(color="#b6fff0", fontSize=14, anchor="start")
        .configure_axis(labelColor="#8bf5d0", titleColor="#8bf5d0", gridColor="#16302b")
        .configure_legend(labelColor="#8bf5d0")
        .configure_view(strokeWidth=0)
    )


def _ensure_session():
    if "tn" not in st.session_state:
        st.session_state.tn = None
    if "models" not in st.session_state:
        st.session_state.models = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = {
            "last_valid": {"airspeed": None, "altitude": None, "temperature": None},
            "buffers": init_buffers(),
            "ewm": {"air": 0.0, "alt": 0.0, "temp": 0.0},
            "kf": None,
            "initialized": False,
            "ground_distance_nm": 0.0,
            "last_t": time.time(),
            "confidence_prev": None,
        }


def _step_once(update_rate_s: float):
    # Lazy connect (and re-connect on failure).
    if st.session_state.tn is None:
        st.session_state.tn = connect()

    if st.session_state.models is None:
        st.session_state.models = load_models()

    tn = st.session_state.tn
    models = st.session_state.models
    p = st.session_state.pipeline

    now = time.time()
    dt_s = max(0.0, now - p["last_t"])
    p["last_t"] = now

    # Read sensors
    air = get_property(tn, SENSORS["airspeed_kt"])
    gs = get_property(tn, SENSORS["groundspeed_kt"])
    alt_ft = get_property(tn, SENSORS["altitude_ft"])
    ground_ft = get_property(tn, SENSORS["ground_elev_ft"])
    vs_fps = get_property(tn, SENSORS["vertical_speed_fps"])
    temp_c = get_property(tn, SENSORS["temp_c"])
    wind_kt = get_property(tn, SENSORS["wind_speed_kt"])
    wind_dir = get_property(tn, SENSORS["wind_dir_deg"])
    pitch = get_property(tn, SENSORS["pitch_deg"])
    roll = get_property(tn, SENSORS["roll_deg"])
    heading = get_property(tn, SENSORS["heading_deg"])

    air = forward_fill(air, p["last_valid"]["airspeed"])
    alt_ft = forward_fill(alt_ft, p["last_valid"]["altitude"])
    temp_c = forward_fill(temp_c, p["last_valid"]["temperature"])

    p["last_valid"]["airspeed"] = air
    p["last_valid"]["altitude"] = alt_ft
    p["last_valid"]["temperature"] = temp_c

    if not p["initialized"] and all(v is not None for v in p["last_valid"].values()):
        p["ewm"] = {"air": float(air), "alt": float(alt_ft), "temp": float(temp_c)}
        p["kf"] = {
            "air": Kalman1D(p["ewm"]["air"]),
            "alt": Kalman1D(p["ewm"]["alt"]),
            "temp": Kalman1D(p["ewm"]["temp"]),
        }
        p["initialized"] = True

    if gs is not None:
        p["ground_distance_nm"] += (float(gs) / 3600.0) * dt_s

    phase = detect_flight_phase(p["ground_distance_nm"], vs_fps, alt_ft, ground_ft)

    # Defaults (show raw immediately; show smooth/residual/model once warmed up).
    smooth_air = np.nan
    smooth_alt = np.nan
    smooth_temp = np.nan
    res_air = np.nan
    res_alt = np.nan
    res_temp = np.nan
    error = np.nan
    confidence = np.nan
    confidence_raw = np.nan

    if p["initialized"]:
        raw = {"air": air, "alt": alt_ft, "temp": temp_c}
        for s in raw:
            p["buffers"]["raw"][s].append(raw[s])

        for s in raw:
            p["ewm"][s] = ewma(p["ewm"][s], float(raw[s]))
            p["buffers"]["smooth"][s].append(p["ewm"][s])
            p["buffers"]["feat"][s].append(p["ewm"][s])

        smooth_air = float(p["ewm"]["air"])
        smooth_alt = float(p["ewm"]["alt"])
        smooth_temp = float(p["ewm"]["temp"])

        if len(p["buffers"]["feat"]["air"]) >= 20:
            feat = {}
            for s in ["air", "alt", "temp"]:
                feat[s] = {
                    "mean": float(np.mean(p["buffers"]["feat"][s])),
                    "std": float(np.std(p["buffers"]["feat"][s])),
                }

            for s in ["air", "alt", "temp"]:
                pred = p["kf"][s].predict()
                res = feat[s]["mean"] - pred
                p["kf"][s].update(feat[s]["mean"])
                # OperationV2: normalize residuals to account for unit scale differences.
                p["buffers"]["res"][s].append(float(res) / float(RESIDUAL_SCALE[s]))

            if len(p["buffers"]["res"]["air"]) > 0:
                res_air = float(p["buffers"]["res"]["air"][-1])
                res_alt = float(p["buffers"]["res"]["alt"][-1])
                res_temp = float(p["buffers"]["res"]["temp"][-1])

            if len(p["buffers"]["res"]["air"]) >= 30:
                X = []
                for s in ["air", "alt", "temp"]:
                    X.append(float(np.mean(p["buffers"]["res"][s])))
                    X.append(float(np.std(p["buffers"]["res"][s])))
                X = np.array(X).reshape(1, -1)

                model = models[phase]
                scaler = model["scaler"]
                pca = model["pca"]

                feature_names = list(
                    getattr(scaler, "feature_names_in_", [f"f{i}" for i in range(X.shape[1])])
                )
                X_df = pd.DataFrame(X, columns=feature_names)
                Xs = scaler.transform(X_df)
                Xr = pca.inverse_transform(pca.transform(Xs))
                error = float(np.mean((Xs - Xr) ** 2))
                threshold = float(model.get("error_threshold", 0.0))
                score = float(error / threshold) if threshold > 0 else 1.0
                confidence_raw = float(np.clip(1 - score, 0, 1))
                prev = p.get("confidence_prev", None)
                confidence = (
                    float(ALPHA_CONF) * float(prev) + (1 - float(ALPHA_CONF)) * confidence_raw
                    if prev is not None
                    else confidence_raw
                )
                p["confidence_prev"] = confidence
            else:
                # OperationV2 warm-up masking: assume healthy until residual window fills.
                confidence = 1.0
                p["confidence_prev"] = confidence

    st.session_state.history.append(
        {
            "t": datetime.now(),
            "confidence": confidence,
            "confidence_raw": confidence_raw,
            "error": error,
            "phase": phase,
            "airspeed_kt": float(air) if air is not None else np.nan,
            "airspeed_smooth": smooth_air,
            "groundspeed_kt": float(gs) if gs is not None else np.nan,
            "altitude_ft": float(alt_ft) if alt_ft is not None else np.nan,
            "altitude_smooth": smooth_alt,
            "vertical_speed_fps": float(vs_fps) if vs_fps is not None else np.nan,
            "temp_c": float(temp_c) if temp_c is not None else np.nan,
            "temp_smooth": smooth_temp,
            "wind_speed_kt": float(wind_kt) if wind_kt is not None else np.nan,
            "wind_dir_deg": float(wind_dir) if wind_dir is not None else np.nan,
            "pitch_deg": float(pitch) if pitch is not None else np.nan,
            "roll_deg": float(roll) if roll is not None else np.nan,
            "heading_deg": float(heading) if heading is not None else np.nan,
            "ground_distance_nm": float(p["ground_distance_nm"]),
            "res_air": res_air,
            "res_alt": res_alt,
            "res_temp": res_temp,
        }
    )

    # Keep history bounded
    max_points = int(st.session_state.get("max_points", 300))
    if len(st.session_state.history) > max_points:
        st.session_state.history = st.session_state.history[-max_points:]


def main():
    st.set_page_config(
        page_title="FlightGear Live Telemetry",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _css()
    _ensure_session()

    st.title("FlightGear Live Telemetry + PCA Confidence")
    st.caption(f"Telnet: {HOST}:{PORT}  |  Update: {UPDATE_RATE:.1f}s (default)")

    with st.sidebar:
        st.header("Controls")
        run = st.toggle("Run Live", value=True)
        st.session_state.max_points = st.slider("History Points", 60, 1200, 300, step=30)
        update_rate_s = st.slider("Refresh (sec)", 0.25, 60.0, 20.0, step=0.25)
        st.caption("Tip: if connection drops, it will reconnect automatically on the next refresh.")

        cols = st.columns(2)
        if cols[0].button("Reconnect"):
            st.session_state.tn = None
        if cols[1].button("Clear History"):
            st.session_state.history = []

    def render(df: pd.DataFrame):
        status = st.empty()
        charts = st.empty()

        if df.empty:
            with status.container():
                st.markdown(
                    "<div class='status-card'>Waiting for data... "
                    "(warmup: W1=20 and W2=30 windows, so ~50 seconds at 1 Hz)</div>",
                    unsafe_allow_html=True,
                )
            return

        latest = df.iloc[-1].to_dict()
        phase = str(latest.get("phase", "-"))
        conf = float(latest.get("confidence", 0.0)) if pd.notna(latest.get("confidence", np.nan)) else np.nan

        with status.container():
            col1, col2, col3, col4 = st.columns([1.2, 1.0, 2.2, 1.6])
            with col1:
                if np.isnan(conf):
                    st.metric("Confidence", "warming up…")
                else:
                    st.metric("Confidence", f"{conf:.4f}")
            with col2:
                st.markdown(f"<div class='phase-pill'>{phase}</div>", unsafe_allow_html=True)
                st.caption("Phase")
            with col3:
                st.caption("Key Telemetry")
                st.write(
                    f"Airspeed `{latest.get('airspeed_kt', np.nan):.2f} kt`  |  "
                    f"Altitude `{latest.get('altitude_ft', np.nan):.1f} ft`  |  "
                    f"Wind `{latest.get('wind_speed_kt', np.nan):.2f} kt`  |  "
                    f"VS `{latest.get('vertical_speed_fps', np.nan):.2f} fps`"
                )
            with col4:
                st.caption("Model")
                st.write(
                    f"Error `{latest.get('error', np.nan):.6f}`  |  "
                    f"Dist `{latest.get('ground_distance_nm', np.nan):.2f} nm`"
                )

        with charts.container():
            left, right = st.columns(2)
            with left:
                st.altair_chart(_chart(df, "confidence", "Confidence Trace", "#00ffaf"), width="stretch")

                air_long = df[["t", "phase", "airspeed_kt", "airspeed_smooth"]].melt(
                    id_vars=["t", "phase"],
                    value_vars=["airspeed_kt", "airspeed_smooth"],
                    var_name="series",
                    value_name="value",
                )
                air_long["series"] = air_long["series"].map(
                    {"airspeed_kt": "Airspeed (raw)", "airspeed_smooth": "Airspeed (smooth)"}
                )
                st.altair_chart(
                    _multi_series_chart(air_long.dropna(), "Airspeed: Raw vs Smooth", ["#7CFF6B", "#00ffaf"]),
                    width="stretch",
                )

            with right:
                alt_long = df[["t", "phase", "altitude_ft", "altitude_smooth"]].melt(
                    id_vars=["t", "phase"],
                    value_vars=["altitude_ft", "altitude_smooth"],
                    var_name="series",
                    value_name="value",
                )
                alt_long["series"] = alt_long["series"].map(
                    {"altitude_ft": "Altitude (raw)", "altitude_smooth": "Altitude (smooth)"}
                )
                st.altair_chart(
                    _multi_series_chart(alt_long.dropna(), "Altitude: Raw vs Smooth", ["#ffd166", "#00ffaf"]),
                    width="stretch",
                )

                temp_long = df[["t", "phase", "temp_c", "temp_smooth"]].melt(
                    id_vars=["t", "phase"],
                    value_vars=["temp_c", "temp_smooth"],
                    var_name="series",
                    value_name="value",
                )
                temp_long["series"] = temp_long["series"].map(
                    {"temp_c": "Temp (raw)", "temp_smooth": "Temp (smooth)"}
                )
                st.altair_chart(
                    _multi_series_chart(temp_long.dropna(), "Temperature: Raw vs Smooth", ["#ff5c7a", "#00ffaf"]),
                    width="stretch",
                )

                res_long = df[["t", "phase", "res_air", "res_alt", "res_temp"]].melt(
                    id_vars=["t", "phase"],
                    value_vars=["res_air", "res_alt", "res_temp"],
                    var_name="series",
                    value_name="value",
                )
                res_long["series"] = res_long["series"].map(
                    {"res_air": "Residual (air)", "res_alt": "Residual (alt)", "res_temp": "Residual (temp)"}
                )
                st.altair_chart(
                    _multi_series_chart(
                        res_long.dropna(),
                        "Residual Error (Kalman residuals)",
                        ["#c77dff", "#00d2ff", "#7CFF6B"],
                    ),
                    width="stretch",
                )

            st.altair_chart(_chart(df, "wind_speed_kt", "Wind Speed (kt)", "#00d2ff"), width="stretch")
            st.altair_chart(_chart(df, "error", "Reconstruction Error (MSE)", "#c77dff"), width="stretch")

            with st.expander("Latest Sample (raw)", expanded=False):
                st.json({k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in latest.items()})

    # Prefer partial refresh to avoid freezing / scroll-to-top on every update.
    if hasattr(st, "fragment"):
        @st.fragment(run_every=update_rate_s if run else None)
        def live_fragment():
            if run:
                try:
                    _step_once(update_rate_s)
                except Exception as e:
                    st.session_state.tn = None
                    st.error(f"Live read failed: {e}")
            df = pd.DataFrame(st.session_state.history)
            render(df)

        live_fragment()
    else:
        # Fallback for older Streamlit: no automatic rerun (to preserve scroll). Use the Refresh button.
        st.warning(
            "Your Streamlit version doesn't support partial refresh (`st.fragment`). "
            "Auto-refresh would reset scroll position, so this dashboard uses manual refresh."
        )
        if st.button("Refresh Now"):
            try:
                if run:
                    _step_once(update_rate_s)
            except Exception as e:
                st.session_state.tn = None
                st.error(f"Live read failed: {e}")
            st.rerun()
        df = pd.DataFrame(st.session_state.history)
        render(df)


if __name__ == "__main__":
    main()
