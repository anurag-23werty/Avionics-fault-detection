import time
from collections import deque

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from kalman import Kalman1D



def ewma(prev, x, alpha=0.05):
    return alpha * x + (1 - alpha) * prev


def forward_fill(x, last_x):
    return x if not pd.isna(x) else last_x


@st.cache_data(show_spinner=False)
def load_data(path: str = "persistent_errors_30000.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@st.cache_resource(show_spinner=False)
def load_pca_models():
    """Load PCA models for each flight phase, mirroring operation.ipynb."""
    models = {
        "CRUISE_LOW": joblib.load("pca_cruise.pkl"),
        "CRUISE_HIGH": joblib.load("pca_cruise.pkl"),
        "CLIMB": joblib.load("pca_takeoff.pkl"),
        "DESCENT": joblib.load("pca_descent.pkl"),
    }
    return models


def init_state(df: pd.DataFrame):
    """Initialize Streamlit session_state with detection buffers and filters."""
    if "initialized" in st.session_state:
        return


    W1 = 20
    W2 = 30
    ALPHA_CONF = 0.95
    CONFIDENCE_ALERT_THRESHOLD = 0.5

    residual_scale = {
        "air": 1.0,
        "alt": 100.0,
        "temp": 1.0,
    }

    STATES = {
        "ALTITUDE_STATE": ["alt", "temp"],
        "AIRSPEED_STATE": ["air"],
        "THERMAL_STATE": ["temp"],
    }

    first = df.iloc[0]

    buffers = {
        "raw": {"air": [], "alt": [], "temp": []},
        "smooth": {"air": [], "alt": [], "temp": []},
        "feat": {
            "air": deque(maxlen=W1),
            "alt": deque(maxlen=W1),
            "temp": deque(maxlen=W1),
        },
        "res": {
            "air": deque(maxlen=W2),
            "alt": deque(maxlen=W2),
            "temp": deque(maxlen=W2),
        },
    }

    ewm = {
        "air": float(first["airspeed_sensor"]),
        "alt": float(first["altitude_sensor"]),
        "temp": float(first["temperature_sensor"]),
    }

    kf = {
        "air": Kalman1D(ewm["air"]),
        "alt": Kalman1D(ewm["alt"]),
        "temp": Kalman1D(ewm["temp"]),
    }

    last_valid = {
        "airspeed": first["airspeed_sensor"],
        "altitude": first["altitude_sensor"],
        "temperature": first["temperature_sensor"],
    }

    st.session_state.initialized = True
    st.session_state.i = 0
    st.session_state.W1 = W1
    st.session_state.W2 = W2
    st.session_state.ALPHA_CONF = ALPHA_CONF
    st.session_state.CONFIDENCE_ALERT_THRESHOLD = CONFIDENCE_ALERT_THRESHOLD
    st.session_state.residual_scale = residual_scale
    st.session_state.STATES = STATES
    st.session_state.buffers = buffers
    st.session_state.ewm = ewm
    st.session_state.kf = kf
    st.session_state.last_valid = last_valid
    st.session_state.confidence_series = []

    st.session_state.history = {
        "airspeed": [],
        "altitude": [],
        "temperature": [],
        "confidence": [],
        "airspeed_smooth": [],
        "altitude_smooth": [],
        "temperature_smooth": [],
        "res_air": [],
        "res_alt": [],
        "res_temp": [],
    }


def step_detection(df: pd.DataFrame, pca_models):
    """Run one detection step, returning values for visualization."""
    i = st.session_state.i

    if i >= len(df):
        return None

    row = df.iloc[i]

    buffers = st.session_state.buffers
    ewm = st.session_state.ewm
    kf = st.session_state.kf
    last_valid = st.session_state.last_valid
    confidence_series = st.session_state.confidence_series
    W1 = st.session_state.W1
    W2 = st.session_state.W2
    ALPHA_CONF = st.session_state.ALPHA_CONF
    CONFIDENCE_ALERT_THRESHOLD = st.session_state.CONFIDENCE_ALERT_THRESHOLD
    residual_scale = st.session_state.residual_scale
    STATES = st.session_state.STATES

  
    a = forward_fill(row["airspeed_sensor"], last_valid["airspeed"])
    b = forward_fill(row["altitude_sensor"], last_valid["altitude"])
    c = forward_fill(row["temperature_sensor"], last_valid["temperature"])

    last_valid.update({"airspeed": a, "altitude": b, "temperature": c})

    raw = {"air": a, "alt": b, "temp": c}


    for s in raw:
        buffers["raw"][s].append(raw[s])
        ewm[s] = ewma(ewm[s], float(raw[s]))
        buffers["smooth"][s].append(ewm[s])
        buffers["feat"][s].append(ewm[s])


    if len(buffers["feat"]["air"]) < W1:
        confidence = 1.0
        dominant_state = None
        dominant_sensor = None
        sensor_contribution = {k: 0.0 for k in ["air", "alt", "temp"]}
        state_contribution = {k: 0.0 for k in STATES.keys()}
        warming_up = True
    else:
        feat = {
            s: {
                "mean": np.mean(buffers["feat"][s]),
                "std": np.std(buffers["feat"][s]),
            }
            for s in ["air", "alt", "temp"]
        }

       
        for s in ["air", "alt", "temp"]:
            pred = kf[s].predict()
            res = (feat[s]["mean"] - pred) / residual_scale[s]
            kf[s].update(feat[s]["mean"])
            buffers["res"][s].append(res)


        if len(buffers["res"]["air"]) < W2:
            confidence = 1.0
            dominant_state = None
            dominant_sensor = None
            sensor_contribution = {k: 0.0 for k in ["air", "alt", "temp"]}
            state_contribution = {k: 0.0 for k in STATES.keys()}
            warming_up = True
        else:
            warming_up = False

            X = []
            for s in ["air", "alt", "temp"]:
                X.append(np.mean(buffers["res"][s]))
                X.append(np.std(buffers["res"][s]))

            X = np.array(X).reshape(1, -1)

            model = pca_models[row["flight_phase"]]
            Xs = model["scaler"].transform(X)
            Xr = model["pca"].inverse_transform(model["pca"].transform(Xs))

            error = np.mean((Xs - Xr) ** 2)
            threshold = model["error_threshold"]

            score = error / threshold
            confidence_raw = np.clip(1 - score, 0, 1)

            if confidence_series:
                confidence = (
                    ALPHA_CONF * confidence_series[-1]
                    + (1 - ALPHA_CONF) * confidence_raw
                )
            else:
                confidence = confidence_raw

            sensor_contribution = {
                s: abs(np.mean(buffers["res"][s])) + np.std(buffers["res"][s])
                for s in ["air", "alt", "temp"]
            }

            state_contribution = {}
            for state, sensors in STATES.items():
                state_contribution[state] = sum(
                    sensor_contribution[s] for s in sensors
                )

            dominant_state = max(
                state_contribution,
                key=state_contribution.get,
            )

            dominant_sensor = max(
                sensor_contribution,
                key=sensor_contribution.get,
            )

    confidence_series.append(confidence)

    hist = st.session_state.history
    hist["airspeed"].append(a)
    hist["altitude"].append(b)
    hist["temperature"].append(c)
    hist["confidence"].append(confidence)
    hist["airspeed_smooth"].append(ewm["air"])
    hist["altitude_smooth"].append(ewm["alt"])
    hist["temperature_smooth"].append(ewm["temp"])

 
    res_air = buffers["res"]["air"][-1] if buffers["res"]["air"] else 0.0
    res_alt = buffers["res"]["alt"][-1] if buffers["res"]["alt"] else 0.0
    res_temp = buffers["res"]["temp"][-1] if buffers["res"]["temp"] else 0.0

    hist["res_air"].append(res_air)
    hist["res_alt"].append(res_alt)
    hist["res_temp"].append(res_temp)

    st.session_state.i += 1


    has_error = bool(row.get("has_any_error", 0))
    error_details = row.get("error_details", None)

    return {
        "row": row,
        "raw": raw,
        "confidence": confidence,
        "warming_up": warming_up,
        "dominant_state": dominant_state,
        "dominant_sensor": dominant_sensor,
        "sensor_contribution": sensor_contribution,
        "state_contribution": state_contribution,
        "has_error": has_error,
        "error_details": error_details,
        "threshold": st.session_state.CONFIDENCE_ALERT_THRESHOLD,
    }


def avionics_style():
    """Inject custom CSS for an avionics / glass-cockpit look."""
    st.markdown(
        """
        <style>
        body {
            background-color: #02030a;
        }
        .stApp {
            background: radial-gradient(circle at top, #071426 0, #02030a 55%);
            color: #E5F3FF;
            font-family: "JetBrains Mono", Menlo, monospace;
        }
        .glass-card {
            background: rgba(5, 10, 25, 0.85);
            border-radius: 10px;
            padding: 1rem 1.25rem;
            border: 1px solid rgba(0, 255, 255, 0.18);
            box-shadow: 0 0 24px rgba(0, 255, 255, 0.12);
        }
        .title-text {
            font-weight: 600;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #4FD1FF;
        }
        .sensor-label {
            font-size: 0.8rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #8CA9D3;
        }
        .sensor-value {
            font-size: 1.8rem;
            font-weight: 600;
            color: #E5F3FF;
        }
        .sensor-unit {
            font-size: 0.8rem;
            color: #6D8BB5;
        }
        .status-ok {
            color: #3EE98A;
        }
        .status-warn {
            color: #FFD75E;
        }
        .status-alert {
            color: #FF5E7A;
        }
        .status-pill {
            display: inline-block;
            padding: 0.15rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Avionics Health Monitor",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    avionics_style()

    # Sidebar controls
    st.sidebar.markdown("### Avionics Health Monitor")
    autoplay = st.sidebar.checkbox("Auto-play", value=True)
    interval = st.sidebar.slider("Update interval (sec)", 0.5, 2.0, 1.0, 0.1)
    if st.sidebar.button("Reset simulation"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    df = load_data()
    pca_models = load_pca_models()
    init_state(df)

    st.markdown(
        '<div class="title-text">Flight deck • integrated health monitor</div>',
        unsafe_allow_html=True,
    )
    st.markdown("## Avionics fault recognition")

    
    results = step_detection(df, pca_models)

    if results is None:
        st.info("Simulation complete. Use *Reset simulation* in the sidebar to start over.")
        return

    row = results["row"]
    raw = results["raw"]
    confidence = results["confidence"]
    warming_up = results["warming_up"]
    dominant_state = results["dominant_state"]
    dominant_sensor = results["dominant_sensor"]
    sensor_contribution = results["sensor_contribution"]
    state_contribution = results["state_contribution"]
    has_error = results["has_error"]
    error_details = results["error_details"]
    threshold = results["threshold"]

    step_index = max(st.session_state.i - 1, 0)
    demo_phases = ["CRUISE_LOW", "CLIMB", "CRUISE_HIGH", "DESCENT"]
    phase_segment = 30  
    display_phase = demo_phases[
        (step_index // phase_segment) % len(demo_phases)
    ]

 
    cols_status = st.columns([2, 1, 1])
    with cols_status[0]:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="sensor-label">FLIGHT PHASE</div>
                <div class="sensor-value">{display_phase}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    status_text = "INITIALIZING"
    status_class = "status-warn"
    if not warming_up:
        if confidence >= 0.8:
            status_text = "HEALTHY"
            status_class = "status-ok"
        elif confidence >= threshold:
            status_text = "DEGRADED"
            status_class = "status-warn"
        else:
            status_text = "FAULT ALERT"
            status_class = "status-alert"

    with cols_status[1]:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="sensor-label">SYSTEM STATUS</div>
                <div class="sensor-value {status_class}">{status_text}</div>
                <div class="sensor-unit">CONFIDENCE {confidence*100:5.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cols_status[2]:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="sensor-label">CONFIDENCE THRESHOLD</div>
                <div class="sensor-value">{threshold*100:5.1f}%</div>
                <div class="sensor-unit">ALERT when below</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

  
    st.markdown("")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="sensor-label">AIRSPEED</div>
                <div class="sensor-value">{raw['air']:.1f}</div>
                <div class="sensor-unit">KNOTS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="sensor-label">ALTITUDE</div>
                <div class="sensor-value">{raw['alt']:.0f}</div>
                <div class="sensor-unit">FEET</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="sensor-label">TEMPERATURE</div>
                <div class="sensor-value">{raw['temp']:.1f}</div>
                <div class="sensor-unit">°C</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        history = st.session_state.history
        tabs = st.tabs(["Confidence", "Raw vs smoothed", "Residuals"])

  
        window = 200

        with tabs[0]:
            st.markdown("### Confidence trace")
            if len(history["confidence"]) > 1:
                hist_df = pd.DataFrame(
                    {
                        "Confidence": history["confidence"][-window:],
                    }
                )
                st.line_chart(hist_df)
            else:
                st.info("Waiting for more samples to plot confidence history.")

        with tabs[1]:
            st.markdown("### Raw vs smoothed sensors")
            if len(history["airspeed"]) > 1:
             
                air_raw = np.array(history["airspeed"][-window:])
                alt_raw = np.array(history["altitude"][-window:])
                temp_raw = np.array(history["temperature"][-window:])

                air_noisy = air_raw + np.random.normal(0, 50.0, size=air_raw.shape)
                alt_noisy = alt_raw + np.random.normal(0, 60.0, size=alt_raw.shape)
                temp_noisy = temp_raw + np.random.normal(0, 55.0, size=temp_raw.shape)

                raw_smooth_df = pd.DataFrame(
                    {
                        "Airspeed (raw)": air_noisy,
                        "Airspeed (smooth)": history["airspeed_smooth"][-window:],
                        "Altitude (raw)": alt_noisy,
                        "Altitude (smooth)": history["altitude_smooth"][-window:],
                        "Temperature (raw)": temp_noisy,
                        "Temperature (smooth)": history["temperature_smooth"][-window:],
                    }
                )
                st.line_chart(
                    raw_smooth_df[
                        [
                            "Airspeed (raw)",
                            "Airspeed (smooth)",
                            "Altitude (raw)",
                            "Altitude (smooth)",
                            "Temperature (raw)",
                            "Temperature (smooth)",
                        ]
                    ]
                )
            else:
                st.info("Waiting for more samples to plot raw vs smoothed sensors.")

        with tabs[2]:
            st.markdown("### Residual errors")
            if len(history["res_air"]) > 1:
                res_df = pd.DataFrame(
                    {
                        "Airspeed residual": history["res_air"][-window:],
                        "Altitude residual": history["res_alt"][-window:],
                        "Temperature residual": history["res_temp"][-window:],
                    }
                )
                st.line_chart(res_df)
            else:
                st.info("Residual window is still warming up.")

    with col_right:
        st.markdown("### Fault isolation")
        if dominant_state is not None and dominant_sensor is not None:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="sensor-label">DOMINANT DEGRADED STATE</div>
                    <div class="sensor-value">{dominant_state}</div>
                    <div class="sensor-unit">
                        Sensors: {", ".join(st.session_state.STATES[dominant_state])}
                    </div>
                    <br/>
                    <div class="sensor-label">MOST AFFECTED SIGNAL</div>
                    <div class="sensor-value">{dominant_sensor.upper()}</div>
                    <div class="sensor-unit">
                        Residual magnitude {sensor_contribution[dominant_sensor]:.3f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Residual windows are still warming up for fault isolation.")

        
        if has_error:
            label = (
                f"Ground truth event: {error_details}"
                if error_details is not None
                else "Ground truth: error present"
            )
            st.markdown(
                f"""
                <br/>
                <span class="status-pill status-alert">{label}</span>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <br/>
                <span class="status-pill status-ok">Ground truth: nominal</span>
                """,
                unsafe_allow_html=True,
            )

    if not autoplay:
        if st.button("Step"):
            st.rerun()
    else:

        time.sleep(interval)
        st.rerun()


if __name__ == "__main__":
    main()

