"""MQTT bridge between the ESP32 sensor node and the AI runtime."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import paho.mqtt.client as mqtt

try:
    from ai.main import FEATURE_COLS, format_pipeline_output, run_pipeline
except Exception:  # pragma: no cover - direct script execution
    from main import FEATURE_COLS, format_pipeline_output, run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = PROJECT_ROOT / "data" / "sensor_log.csv"

BROKER = os.getenv("ALG_MQTT_BROKER", "10.35.93.69")
PORT = int(os.getenv("ALG_MQTT_PORT", "1883"))
SENSOR_TOPIC = os.getenv("ALG_SENSOR_TOPIC", "alg1/sensors")
ACTION_TOPIC = os.getenv("ALG_ACTION_TOPIC", "alg1/actions")
ACTION_FORMAT = os.getenv("ALG_MQTT_ACTION_FORMAT", "json").lower()
RECONNECT_DELAY = 5

GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _banner(message: str, colour: str = CYAN) -> None:
    print(f"\n{BOLD}{colour}{'=' * 64}{RESET}")
    print(f"{BOLD}{colour}{message}{RESET}")
    print(f"{BOLD}{colour}{'=' * 64}{RESET}")


def _log(tag: str, message: str, colour: str = RESET) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"{BOLD}{BLUE}[{timestamp}][{tag:<7}]{RESET} {colour}{message}{RESET}")


def _validate_sensor_payload(payload: dict[str, Any]) -> dict[str, float]:
    missing = [key for key in FEATURE_COLS if key not in payload]
    if missing:
        raise ValueError(f"missing sensor fields: {missing}")

    clean = {key: float(payload[key]) for key in FEATURE_COLS}
    clean["Motion_Detected"] = int(clean["Motion_Detected"])
    if "Timestamp" in payload:
        clean["Timestamp"] = payload["Timestamp"]
    return clean


def _action_payload(result: dict[str, Any]) -> str:
    action = {key: value for key, value in result.items() if key != "_meta"}
    if ACTION_FORMAT in {"mode", "id", "int"}:
        return str(action.get("action_id", result.get("_meta", {}).get("action_id", 0)))
    return json.dumps(action, separators=(",", ":"))


def _write_log(sensor_data: dict[str, Any], result: dict[str, Any]) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    meta = result.get("_meta", {})
    action = {key: value for key, value in result.items() if key != "_meta"}
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sensor_timestamp": sensor_data.get("Timestamp", ""),
        **sensor_data,
        "state_label": meta.get("state_label", "Unknown"),
        "risk": meta.get("risk", "Unknown"),
        "risk_score": meta.get("risk_score", 0.0),
        "scenario_id": meta.get("scenario_id", 0),
        "scenario": meta.get("scenario", "Unknown"),
        "cluster_id": meta.get("cluster_id", 0),
        "gas_pred": meta.get("gas_pred", 0.0),
        "temp_pred": meta.get("temp_pred", 0.0),
        "trend": meta.get("trend", 0.0),
        "raw_trend": meta.get("raw_trend", 0.0),
        "spatial_risk": meta.get("spatial_risk", 0.0),
        "anomaly_score": meta.get("anomaly_score", 0.0),
        "is_anomaly": meta.get("is_anomaly", False),
        "guard_risk_class": meta.get("guard_risk_class", 0),
        "guard_scenario_class": meta.get("guard_scenario_class", 0),
        "guard_scenario_label": meta.get("guard_scenario_label", "Unknown"),
        "guard_confidence": meta.get("guard_confidence", 0.0),
        "reward": meta.get("reward", 0.0),
        **action,
    }

    fieldnames = [
        "timestamp",
        "sensor_timestamp",
        *FEATURE_COLS,
        "state_label",
        "risk",
        "risk_score",
        "scenario_id",
        "scenario",
        "cluster_id",
        "gas_pred",
        "temp_pred",
        "trend",
        "raw_trend",
        "spatial_risk",
        "anomaly_score",
        "is_anomaly",
        "guard_risk_class",
        "guard_scenario_class",
        "guard_scenario_label",
        "guard_confidence",
        "reward",
        "fan",
        "alarm",
        "servo",
        "buzzer",
        "rgb_led",
        "action_id",
    ]

    file_exists = LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        _banner("Adaptive Lab Guardian MQTT bridge connected", GREEN)
        _log("BROKER", f"{BROKER}:{PORT}", GREEN)
        _log("SUB", SENSOR_TOPIC, GREEN)
        _log("PUB", f"{ACTION_TOPIC} ({ACTION_FORMAT})", GREEN)
        _log("LOG", str(LOG_FILE), GREEN)
        client.subscribe(SENSOR_TOPIC)
    else:
        _log("ERROR", f"connection failed with rc={rc}", RED)


def on_disconnect(client, userdata, rc):
    if rc == 0:
        _log("MQTT", "disconnected cleanly", YELLOW)
    else:
        _log("MQTT", f"unexpected disconnect rc={rc}; reconnecting in {RECONNECT_DELAY}s", RED)
        time.sleep(RECONNECT_DELAY)


def on_message(client, userdata, msg):
    _log("RECV", f"{msg.topic} {len(msg.payload)} bytes", CYAN)

    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        sensor_data = _validate_sensor_payload(payload)
    except Exception as exc:
        _log("ERROR", f"bad sensor payload: {exc}", RED)
        return

    try:
        result = run_pipeline(sensor_data, verbose=False)
    except Exception as exc:
        _log("ERROR", f"pipeline failed: {exc}", RED)
        return

    print(format_pipeline_output(result, sensor_data))

    try:
        outgoing = _action_payload(result)
        client.publish(ACTION_TOPIC, outgoing)
        _write_log(sensor_data, result)
        _log("SEND", f"{ACTION_TOPIC}: {outgoing}", GREEN)
    except Exception as exc:
        _log("ERROR", f"publish/log failed: {exc}", RED)


def main() -> None:
    _banner("Adaptive Lab Guardian MQTT bridge", CYAN)
    _log("INIT", f"connecting to {BROKER}:{PORT}", YELLOW)

    client = mqtt.Client(client_id=os.getenv("ALG_MQTT_CLIENT_ID", ""), clean_session=True)
    client.reconnect_delay_set(min_delay=1, max_delay=10)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    try:
        client.connect(BROKER, PORT, keepalive=60)
    except Exception as exc:
        _log("ERROR", f"cannot reach broker: {exc}", RED)
        return

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        _banner("Stopping MQTT bridge", YELLOW)
        client.disconnect()


if __name__ == "__main__":
    main()
