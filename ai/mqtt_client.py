"""MQTT bridge between the ESP32 sensor node and the AI runtime."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import paho.mqtt.client as mqtt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = PROJECT_ROOT / "data" / "sensor_log.csv"
CONTROL_STATE_FILE = PROJECT_ROOT / "data" / "control_state.json"
FEATURE_COLS = ["Temp_C", "Humidity_pct", "Gas_AQI", "Light_Lux", "Motion_Detected"]

def load_env() -> None:
    """Manual parser for .env files to populate os.environ with zero external dependencies."""
    for env_path in [PROJECT_ROOT / ".env", PROJECT_ROOT / "dashboard" / ".env"]:
        if env_path.exists():
            try:
                with env_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            key, val = line.split("=", 1)
                            key = key.strip()
                            val = val.strip().strip("'\"")
                            if key and key not in os.environ:
                                os.environ[key] = val
            except Exception:
                pass

load_env()

PURE_IOT_MODE = os.getenv("ALG_PURE_IOT", "false").lower() in ("1", "true", "yes")
_ai_runtime: tuple[Callable[..., dict[str, Any]], Callable[..., str]] | None = None


BROKER = os.getenv("ALG_MQTT_BROKER", "127.0.0.1")
PORT = int(os.getenv("ALG_MQTT_PORT", "1883"))
SENSOR_TOPIC = os.getenv("ALG_SENSOR_TOPIC", "alg1/sensors")
ACTION_TOPIC = os.getenv("ALG_ACTION_TOPIC", "alg1/actions")
MODE_TOPIC = os.getenv("ALG_MODE_TOPIC", "alg1/mode")
ACTION_FORMAT = os.getenv("ALG_MQTT_ACTION_FORMAT", "json").lower()
RECONNECT_DELAY = 5

GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"
VALID_MODES = {"AI", "MANUAL"}
DEFAULT_SYSTEM_MODE = os.getenv("ALG_SYSTEM_MODE", "MANUAL").strip().upper()
if DEFAULT_SYSTEM_MODE not in VALID_MODES:
    DEFAULT_SYSTEM_MODE = "MANUAL"

DEFAULT_MANUAL_ACTION = {
    "fan": "OFF",
    "alarm": "OFF",
    "servo": "CLOSED",
    "buzzer": "OFF",
    "rgb_led": "GREEN",
    "action_id": 0,
}


def _normalise_mode(value: Any) -> str | None:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")

    text = str(value or "").strip()
    if not text:
        return None

    if text.startswith("{"):
        try:
            payload = json.loads(text)
            text = str(payload.get("mode", payload.get("system_mode", ""))).strip()
        except Exception:
            return None

    mode = text.upper()
    return mode if mode in VALID_MODES else None


def _read_persisted_system_mode() -> str:
    try:
        with CONTROL_STATE_FILE.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return _normalise_mode(payload.get("system_mode")) or DEFAULT_SYSTEM_MODE
    except Exception:
        return DEFAULT_SYSTEM_MODE


def _persist_system_mode(mode: str) -> None:
    try:
        CONTROL_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with CONTROL_STATE_FILE.open("w", encoding="utf-8") as f:
            json.dump(
                {"system_mode": mode, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                f,
                separators=(",", ":"),
            )
    except Exception as exc:
        _log("WARN", f"failed to persist control mode: {exc}", YELLOW)


def _set_system_mode(mode: str, *, persist: bool = True) -> None:
    global system_mode
    system_mode = mode
    if persist:
        _persist_system_mode(mode)


def _refresh_system_mode_from_disk() -> str:
    persisted = _read_persisted_system_mode()
    if persisted != system_mode:
        _set_system_mode(persisted, persist=False)
        _log("MODE", f"System mode synced from control state: {system_mode}", YELLOW)
    return system_mode


def _ai_enabled() -> bool:
    return system_mode == "AI"


def _pure_iot_result() -> dict[str, Any]:
    return {
        **DEFAULT_MANUAL_ACTION,
        "_meta": {
            "state_label": "Normal",
            "risk": "Safe",
            "risk_score": 0.0,
            "scenario_id": 0,
            "scenario": "Pure IoT Mode",
            "cluster_id": 0,
            "gas_pred": 0.0,
            "temp_pred": 0.0,
            "trend": 0.0,
            "raw_trend": 0.0,
            "spatial_risk": 0.0,
            "anomaly_score": 0.0,
            "is_anomaly": False,
            "guard_risk_class": 0,
            "guard_scenario_class": 0,
            "guard_scenario_label": "IoT Bridge",
            "guard_confidence": 1.0,
            "reward": 0.0,
        },
    }


def _load_ai_runtime() -> tuple[Callable[..., dict[str, Any]], Callable[..., str]]:
    """Lazy-load the heavy AI runtime only when AI control is active."""
    global _ai_runtime
    if _ai_runtime is not None:
        return _ai_runtime

    if PURE_IOT_MODE:
        def run_pipeline(*args, **kwargs):
            return _pure_iot_result()

        def format_pipeline_output(*args, **kwargs):
            return "PURE IOT MODE - AI DISABLED"

        _ai_runtime = (run_pipeline, format_pipeline_output)
        return _ai_runtime

    try:
        from ai.main import format_pipeline_output, run_pipeline
    except Exception:  # pragma: no cover - direct script execution
        from main import format_pipeline_output, run_pipeline

    _ai_runtime = (run_pipeline, format_pipeline_output)
    return _ai_runtime


def _coerce_action_id(value: Any) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = 0
    return max(0, min(3, parsed))


def _action_from_id(action_id: Any) -> dict[str, Any]:
    mode = _coerce_action_id(action_id)
    if mode == 1:
        return {
            "fan": "ON",
            "alarm": "OFF",
            "servo": "CLOSED",
            "buzzer": "OFF",
            "rgb_led": "YELLOW",
            "action_id": 1,
        }
    if mode == 2:
        return {
            "fan": "ON",
            "alarm": "ON",
            "servo": "OPEN",
            "buzzer": "ON",
            "rgb_led": "RED",
            "action_id": 2,
        }
    if mode == 3:
        return {
            "fan": "OFF",
            "alarm": "ON",
            "servo": "CLOSED",
            "buzzer": "ON",
            "rgb_led": "RED",
            "action_id": 3,
        }
    return DEFAULT_MANUAL_ACTION.copy()


def _normalise_action(action_like: dict[str, Any]) -> dict[str, Any]:
    fallback = _action_from_id(action_like.get("action_id", 0))
    return {
        "fan": str(action_like.get("fan", fallback["fan"])).upper(),
        "alarm": str(action_like.get("alarm", fallback["alarm"])).upper(),
        "servo": str(action_like.get("servo", fallback["servo"])).upper(),
        "buzzer": str(action_like.get("buzzer", fallback["buzzer"])).upper(),
        "rgb_led": str(action_like.get("rgb_led", fallback["rgb_led"])).upper(),
        "action_id": _coerce_action_id(action_like.get("action_id", fallback["action_id"])),
    }


def _manual_result(action: dict[str, Any]) -> dict[str, Any]:
    return {
        **_normalise_action(action),
        "_meta": {
            "state_label": "Normal",
            "risk": "Safe",
            "risk_score": 0.0,
            "scenario_id": 0,
            "scenario": "Manual Control Active",
            "cluster_id": 0,
            "gas_pred": 0.0,
            "temp_pred": 0.0,
            "trend": 0.0,
            "raw_trend": 0.0,
            "spatial_risk": 0.0,
            "anomaly_score": 0.0,
            "is_anomaly": False,
            "guard_risk_class": 0,
            "guard_scenario_class": 0,
            "guard_scenario_label": "Manual",
            "guard_confidence": 1.0,
            "reward": 0.0,
            "action_id": _coerce_action_id(action.get("action_id", 0)),
        },
    }


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


def _action_payload(result: dict[str, Any], *, source: str = "ai") -> str:
    action = {key: value for key, value in result.items() if key != "_meta"}
    if ACTION_FORMAT in {"mode", "id", "int"}:
        return str(action.get("action_id", result.get("_meta", {}).get("action_id", 0)))
    action["source"] = source
    action["control_mode"] = system_mode
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
        _log("SUB", MODE_TOPIC, GREEN)
        _log("SUB", ACTION_TOPIC, GREEN)
        _log("PUB", f"{ACTION_TOPIC} ({ACTION_FORMAT})", GREEN)
        _log("MODE", system_mode, GREEN if _ai_enabled() else YELLOW)
        _log("LOG", str(LOG_FILE), GREEN)
        client.subscribe(SENSOR_TOPIC)
        client.subscribe(MODE_TOPIC)
        client.subscribe(ACTION_TOPIC)
    else:
        _log("ERROR", f"connection failed with rc={rc}", RED)


def on_disconnect(client, userdata, rc):
    if rc == 0:
        _log("MQTT", "disconnected cleanly", YELLOW)
    else:
        _log("MQTT", f"unexpected disconnect rc={rc}; reconnecting in {RECONNECT_DELAY}s", RED)
        time.sleep(RECONNECT_DELAY)


# Global system mode (AI or MANUAL) and manual state tracking
system_mode = _read_persisted_system_mode()
last_manual_action = DEFAULT_MANUAL_ACTION.copy()

def on_message(client, userdata, msg):
    global system_mode, last_manual_action
    topic = msg.topic

    # Handle system mode updates
    if topic == MODE_TOPIC:
        try:
            next_mode = _normalise_mode(msg.payload)
            if next_mode:
                _set_system_mode(next_mode)
                _log("MODE", f"System mode set to: {system_mode}", YELLOW)
            else:
                _log("ERROR", f"bad mode payload: {msg.payload!r}", RED)
        except Exception as exc:
            _log("ERROR", f"bad mode payload: {exc}", RED)
        return

    # Track manual actions from the dashboard
    if topic == ACTION_TOPIC:
        try:
            action_payload = msg.payload.decode("utf-8").strip()
            if action_payload.startswith("{"):
                parsed_action = json.loads(action_payload)
                if "action_id" in parsed_action:
                    source = str(parsed_action.get("source", "")).strip().lower()
                    if source != "ai":
                        last_manual_action = _normalise_action(parsed_action)
                        _log("MANUAL", f"Manual action tracked: mode {last_manual_action['action_id']}", YELLOW)
            elif action_payload:
                if system_mode == "MANUAL":
                    last_manual_action = _action_from_id(action_payload)
                    _log("MANUAL", f"Manual action tracked: mode {last_manual_action['action_id']}", YELLOW)
        except Exception:
            pass
        return

    # Handle sensor updates
    if topic == SENSOR_TOPIC:
        _log("RECV", f"{msg.topic} {len(msg.payload)} bytes", CYAN)

        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            sensor_data = _validate_sensor_payload(payload)
        except Exception as exc:
            _log("ERROR", f"bad sensor payload: {exc}", RED)
            return



        try:
            run_pipeline, format_pipeline_output = _load_ai_runtime()
            result = run_pipeline(sensor_data, verbose=True)
        except Exception as exc:
            _log("ERROR", f"pipeline failed: {exc}", RED)
            return

        print(format_pipeline_output(result, sensor_data))

        if not _ai_enabled():
            _log("MANUAL", "System in MANUAL mode. Overriding AI actuator decisions.", YELLOW)
            for k, v in last_manual_action.items():
                if k != "_meta":
                    result[k] = v
            result["_meta"]["action_id"] = last_manual_action.get("action_id", 0)

        try:
            _write_log(sensor_data, result)

            # Publish AI decisions only while AI control is still active.
            if _ai_enabled():
                outgoing = _action_payload(result, source="ai")
                client.publish(ACTION_TOPIC, outgoing)
                _log("SEND", f"{ACTION_TOPIC}: {outgoing}", GREEN)
            else:
                _log("SKIP", "AI action skipped (System is in MANUAL mode)", YELLOW)
        except Exception as exc:
            _log("ERROR", f"failed to send or log: {exc}", RED)


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
