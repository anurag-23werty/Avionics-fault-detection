import os
import re
import socket
import time
from collections import deque
from datetime import datetime
import csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from kalman import Kalman1D

# ============================================================
# Connection / Runtime
# ============================================================
HOST = os.environ.get("FG_HOST", "localhost")
PORT = int(os.environ.get("FG_PORT", "5500"))
UPDATE_RATE = float(os.environ.get("FG_UPDATE_RATE", "1.0"))  # seconds
CONNECT_TIMEOUT_S = float(os.environ.get("FG_CONNECT_TIMEOUT", "3.0"))
CONNECT_RETRIES = int(os.environ.get("FG_CONNECT_RETRIES", "60"))
CSV_LOG_PATH = os.environ.get("FG_CSV_LOG", "").strip()
CSV_FLUSH_EVERY = int(os.environ.get("FG_CSV_FLUSH_EVERY", "1"))

# Terminal style
GREEN_LOGS = os.environ.get("FG_GREEN_LOGS", "1") != "0"
ANSI_GREEN = "\033[92m"
ANSI_RESET = "\033[0m"


def green(s: str) -> str:
    return f"{ANSI_GREEN}{s}{ANSI_RESET}" if GREEN_LOGS else s


class FGTelnetClient:
    """
    Minimal, telnetlib-free client for FlightGear's property server (Python 3.13 compatible).
    """

    def __init__(self, host: str, port: int, timeout_s: float):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.sock = socket.create_connection((host, port), timeout=timeout_s)
        self.sock.settimeout(timeout_s)
        self._buf = b""

    def write(self, data: bytes):
        self.sock.sendall(data)

    def read_until(self, delim: bytes = b"\r\n", max_bytes: int = 65536) -> bytes:
        try:
            while delim not in self._buf and len(self._buf) < max_bytes:
                try:
                    chunk = self.sock.recv(4096)
                except socket.timeout:
                    break
                if not chunk:
                    break
                self._buf += chunk
        except TimeoutError:
            pass

        idx = self._buf.find(delim)
        if idx == -1:
            out = self._buf
            self._buf = b""
            return out

        out = self._buf[: idx + len(delim)]
        self._buf = self._buf[idx + len(delim) :]
        return out

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def connect():
    last_err = None
    for attempt in range(1, CONNECT_RETRIES + 1):
        try:
            tn = FGTelnetClient(HOST, PORT, timeout_s=CONNECT_TIMEOUT_S)
            print(green(f"[INFO] Connected to FlightGear at {HOST}:{PORT}"))
            return tn
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            last_err = e
            if attempt == 1:
                print(
                    green(
                        f"[WARN] Can't connect to FlightGear telnet at {HOST}:{PORT} "
                        f"(attempt {attempt}/{CONNECT_RETRIES})."
                    )
                )
                print(
                    green(
                        "[HINT] Ensure FlightGear is running and the telnet/property server is enabled "
                        f"(commonly `--telnet={PORT}`)."
                    )
                )
            else:
                print(green(f"[WARN] Retry {attempt}/{CONNECT_RETRIES} failed: {e}"))
            time.sleep(1.0)

    raise RuntimeError(
        f"Failed to connect to FlightGear telnet at {HOST}:{PORT} after "
        f"{CONNECT_RETRIES} retries. Last error: {last_err}"
    )


def get_property(tn, path: str):
    tn.write(f"get {path}\r\n".encode())
    try:
        raw = tn.read_until(b"\r\n")
    except (socket.timeout, TimeoutError, OSError):
        return None
    if not raw:
        return None
    response = raw.decode(errors="ignore")
    match = re.search(r"'([^']+)'", response)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def is_missing(x):
    return x is None or (isinstance(x, float) and np.isnan(x))


def forward_fill(x, last_x):
    return last_x if is_missing(x) else x


def ewma(prev, x, alpha=0.2):
    return alpha * x + (1 - alpha) * prev


# ============================================================
# CSV Logging (modular)
# ============================================================
CSV_FIELDNAMES = [
    "timestamp",
    "phase",
    "airspeed_kt",
    "groundspeed_kt",
    "altitude_ft",
    "ground_elev_ft",
    "vertical_speed_fps",
    "temp_c",
    "wind_speed_kt",
    "ground_distance_nm",
    "confidence",
    "confidence_raw",
    "recon_error",
    "dominant_state",
    "dominant_sensor",
]


def init_csv_logger(path: str = "", flush_every: int = 1):
    """
    Create a CSV logger that appends rows continuously during the run.

    Returns a dict that is passed to `append_csv_row(...)` and closed by `close_csv_logger(...)`.
    """
    if not path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"flightgear_log_{ts}.csv"

    log_path = Path(path)
    csv_file = log_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)

    if log_path.stat().st_size == 0:
        writer.writeheader()
        csv_file.flush()

    return {
        "path": log_path,
        "file": csv_file,
        "writer": writer,
        "flush_every": max(1, int(flush_every)),
        "rows_since_flush": 0,
    }


def append_csv_row(logger: dict, **values):
    """
    Append one row to the run CSV.

    Any missing field will be written as "" (empty) to keep a consistent schema.
    """
    row = {k: values.get(k, "") for k in CSV_FIELDNAMES}
    logger["writer"].writerow(row)
    logger["rows_since_flush"] += 1
    if logger["rows_since_flush"] >= logger["flush_every"]:
        logger["file"].flush()
        logger["rows_since_flush"] = 0


def close_csv_logger(logger: dict):
    try:
        logger["file"].flush()
        logger["file"].close()
    except OSError:
        pass


# ============================================================
# Flight Phase Detection (same heuristic as V2)
# ============================================================
CLIMB_THRESHOLD_FPS = 5.0
DESCENT_THRESHOLD_FPS = -5.0
CRUISE_LOW_MAX_AGL_FT = 10000.0


def detect_flight_phase(ground_distance_nm, vertical_speed_fps, altitude_ft, ground_elev_ft):
    if vertical_speed_fps is None:
        vertical_speed_fps = 0.0

    early_climb_fps = max(1.0, CLIMB_THRESHOLD_FPS / 2.0)
    if ground_distance_nm is not None and ground_distance_nm < 5.0 and vertical_speed_fps >= early_climb_fps:
        return "CLIMB"

    if vertical_speed_fps >= CLIMB_THRESHOLD_FPS:
        return "CLIMB"
    if vertical_speed_fps <= DESCENT_THRESHOLD_FPS:
        return "DESCENT"

    if altitude_ft is None or ground_elev_ft is None:
        return "CRUISE_LOW"

    agl_ft = max(0.0, altitude_ft - ground_elev_ft)
    return "CRUISE_LOW" if agl_ft < CRUISE_LOW_MAX_AGL_FT else "CRUISE_HIGH"


# ============================================================
# OperationV2-style Coupled / Normalized Residual Reasoning
# ============================================================
W1 = int(os.environ.get("FG_W1", "20"))
W2 = int(os.environ.get("FG_W2", "30"))
ALPHA_CONF = float(os.environ.get("FG_ALPHA_CONF", "0.95"))
CONFIDENCE_ALERT_THRESHOLD = float(os.environ.get("FG_CONF_ALERT", "0.5"))

# Residual normalization (sensor scaling) — matches operationV2.ipynb
RESIDUAL_SCALE = {
    "air": float(os.environ.get("FG_RES_SCALE_AIR", "1.0")),
    "alt": float(os.environ.get("FG_RES_SCALE_ALT", "100.0")),
    "temp": float(os.environ.get("FG_RES_SCALE_TEMP", "1.0")),
}

# "Coupling" at state level (altitude depends on thermal conditions)
STATES = {
    "ALTITUDE_STATE": ["alt", "temp"],
    "AIRSPEED_STATE": ["air"],
    "THERMAL_STATE": ["temp"],
}


PCA_MODELS = {
    "CRUISE_LOW": "pca_cruise.pkl",
    "CRUISE_HIGH": "pca_cruise.pkl",
    "CLIMB": "pca_takeoff.pkl",
    "DESCENT": "pca_descent.pkl",
}


def load_models():
    return {phase: joblib.load(path) for phase, path in PCA_MODELS.items()}


def init_buffers():
    return {
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


# ============================================================
# FlightGear Property Paths
# ============================================================
SENSORS = {
    "airspeed_kt": "/velocities/airspeed-kt",
    "groundspeed_kt": "/velocities/groundspeed-kt",
    "altitude_ft": "/position/altitude-ft",
    "ground_elev_ft": "/position/ground-elev-ft",
    "vertical_speed_fps": "/velocities/vertical-speed-fps",
    "temp_c": "/environment/temperature-degc",
    "wind_speed_kt": "/environment/wind-speed-kt",
}


def main():
    tn = connect()
    models = load_models()

    buffers = init_buffers()
    confidence_prev = None

    last_valid = {"airspeed": None, "altitude": None, "temperature": None}

    ewm = {"air": 0.0, "alt": 0.0, "temp": 0.0}
    kf = None
    initialized = False

    last_t = time.time()
    ground_distance_nm = 0.0

    # ------------------------------------------------------------
    # Continuous CSV logging (one row per loop)
    # ------------------------------------------------------------
    logger = init_csv_logger(CSV_LOG_PATH, flush_every=CSV_FLUSH_EVERY)

    try:
        while True:
            now = time.time()
            dt_s = max(0.0, now - last_t)
            last_t = now

            air = get_property(tn, SENSORS["airspeed_kt"])
            gs = get_property(tn, SENSORS["groundspeed_kt"])
            alt = get_property(tn, SENSORS["altitude_ft"])
            ground = get_property(tn, SENSORS["ground_elev_ft"])
            vs = get_property(tn, SENSORS["vertical_speed_fps"])
            temp = get_property(tn, SENSORS["temp_c"])
            wind = get_property(tn, SENSORS["wind_speed_kt"])

            air = forward_fill(air, last_valid["airspeed"])
            alt = forward_fill(alt, last_valid["altitude"])
            temp = forward_fill(temp, last_valid["temperature"])

            last_valid["airspeed"] = air
            last_valid["altitude"] = alt
            last_valid["temperature"] = temp

            if not initialized and all(v is not None for v in last_valid.values()):
                ewm = {"air": float(air), "alt": float(alt), "temp": float(temp)}
                kf = {
                    "air": Kalman1D(ewm["air"]),
                    "alt": Kalman1D(ewm["alt"]),
                    "temp": Kalman1D(ewm["temp"]),
                }
                initialized = True

            if gs is not None:
                ground_distance_nm += (float(gs) / 3600.0) * dt_s

            phase = detect_flight_phase(ground_distance_nm, vs, alt, ground)

            if not initialized:
                print(green(f"[WARMUP] Waiting for valid sensors... Phase={phase}"))
                append_csv_row(
                    logger,
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    phase=phase,
                    airspeed_kt=air,
                    groundspeed_kt=gs,
                    altitude_ft=alt,
                    ground_elev_ft=ground,
                    vertical_speed_fps=vs,
                    temp_c=temp,
                    wind_speed_kt=wind,
                    ground_distance_nm=ground_distance_nm,
                )
                time.sleep(UPDATE_RATE)
                continue

            raw = {"air": air, "alt": alt, "temp": temp}
            for s in raw:
                buffers["raw"][s].append(raw[s])
                ewm[s] = ewma(ewm[s], float(raw[s]))
                buffers["smooth"][s].append(ewm[s])
                buffers["feat"][s].append(ewm[s])

            if len(buffers["feat"]["air"]) < W1:
                print(green(f"[WARMUP] Feature window filling ({len(buffers['feat']['air'])}/{W1})"))
                append_csv_row(
                    logger,
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    phase=phase,
                    airspeed_kt=air,
                    groundspeed_kt=gs,
                    altitude_ft=alt,
                    ground_elev_ft=ground,
                    vertical_speed_fps=vs,
                    temp_c=temp,
                    wind_speed_kt=wind,
                    ground_distance_nm=ground_distance_nm,
                    confidence=1.0,
                )
                time.sleep(UPDATE_RATE)
                continue

            feat = {
                s: {"mean": float(np.mean(buffers["feat"][s])), "std": float(np.std(buffers["feat"][s]))}
                for s in ["air", "alt", "temp"]
            }

            for s in ["air", "alt", "temp"]:
                pred = kf[s].predict()
                # OperationV2: normalized residuals (scale differs by sensor).
                res = (feat[s]["mean"] - pred) / RESIDUAL_SCALE[s]
                kf[s].update(feat[s]["mean"])
                buffers["res"][s].append(float(res))

            if len(buffers["res"]["air"]) < W2:
                # OperationV2: warm-up masking => treat as healthy initially.
                confidence = 1.0
                confidence_prev = confidence
                print(
                    green(
                        f"[WARMUP] Residual window filling ({len(buffers['res']['air'])}/{W2}) | "
                        f"Phase={phase} | Confidence={confidence:.3f}"
                    )
                )
                append_csv_row(
                    logger,
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    phase=phase,
                    airspeed_kt=air,
                    groundspeed_kt=gs,
                    altitude_ft=alt,
                    ground_elev_ft=ground,
                    vertical_speed_fps=vs,
                    temp_c=temp,
                    wind_speed_kt=wind,
                    ground_distance_nm=ground_distance_nm,
                    confidence=confidence,
                )
                time.sleep(UPDATE_RATE)
                continue

            X = []
            for s in ["air", "alt", "temp"]:
                X.append(float(np.mean(buffers["res"][s])))
                X.append(float(np.std(buffers["res"][s])))
            X = np.array(X).reshape(1, -1)

            model = models[phase]
            scaler = model["scaler"]
            pca = model["pca"]
            threshold = float(model["error_threshold"])

            feature_names = list(
                getattr(scaler, "feature_names_in_", [f"f{i}" for i in range(X.shape[1])])
            )
            X_df = pd.DataFrame(X, columns=feature_names)

            Xs = scaler.transform(X_df)
            Xr = pca.inverse_transform(pca.transform(Xs))
            error = float(np.mean((Xs - Xr) ** 2))

            score = float(error / threshold) if threshold > 0 else 1.0
            confidence_raw = float(np.clip(1 - score, 0, 1))

            if confidence_prev is not None:
                confidence = float(ALPHA_CONF * confidence_prev + (1 - ALPHA_CONF) * confidence_raw)
            else:
                confidence = confidence_raw
            confidence_prev = confidence

            # Coupled state-level reasoning (from operationV2)
            sensor_contribution = {
                s: abs(float(np.mean(buffers["res"][s]))) + float(np.std(buffers["res"][s]))
                for s in ["air", "alt", "temp"]
            }
            state_contribution = {
                state: sum(sensor_contribution[s] for s in sensors) for state, sensors in STATES.items()
            }
            dominant_state = max(state_contribution, key=state_contribution.get)
            dominant_sensor = max(sensor_contribution, key=sensor_contribution.get)

            print(
                green(
                    f"Phase:{phase} | Error:{error:.2e} | ConfRaw:{confidence_raw:.3f} | "
                    f"Confidence:{confidence:.3f} | Air:{air:.2f}kt Alt:{alt:.1f}ft "
                    f"VS:{(vs if vs is not None else 0.0):.2f}fps Wind:{(wind if wind is not None else 0.0):.2f}kt"
                )
            )

            append_csv_row(
                logger,
                timestamp=datetime.now().isoformat(timespec="seconds"),
                phase=phase,
                airspeed_kt=air,
                groundspeed_kt=gs,
                altitude_ft=alt,
                ground_elev_ft=ground,
                vertical_speed_fps=vs,
                temp_c=temp,
                wind_speed_kt=wind,
                ground_distance_nm=ground_distance_nm,
                confidence=confidence,
                confidence_raw=confidence_raw,
                recon_error=error,
                dominant_state=dominant_state,
                dominant_sensor=dominant_sensor,
            )

            if confidence < CONFIDENCE_ALERT_THRESHOLD:
                print(green("! SYSTEM HEALTH ALERT"))
                print(green(f"  Confidence: {confidence:.3f}"))
                print(green(f"  Dominant degraded state: {dominant_state}"))
                print(
                    green(
                        f"  Most affected signal: {dominant_sensor.upper()} "
                        f"(residual magnitude {sensor_contribution[dominant_sensor]:.3f})"
                    )
                )
                print(green("-" * 60))

            time.sleep(UPDATE_RATE)

    except KeyboardInterrupt:
        print(green("\n[INFO] Stopping live inference"))
        close_csv_logger(logger)
        print(green(f"[INFO] Flight log saved to: {logger['path'].resolve()}"))
        tn.close()


if __name__ == "__main__":
    main()
