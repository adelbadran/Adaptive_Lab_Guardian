import json
import time
import paho.mqtt.client as mqtt
from ai.main import run_pipeline   
import ai.gnn

# =============================================================================
#  CONFIGURATION
# =============================================================================

BROKER = "broker.hivemq.com"
PORT = 1883
SENSOR_TOPIC = "alg1/sensors"
ACTION_TOPIC = "alg1/actions"

# Required keys that must exist in every ESP32 message
REQUIRED_KEYS = ["Temp_C", "Humidity_pct", "Gas_AQI", "Light_Lux", "Motion_Detected"]

# How long to wait (seconds) between reconnect attempts
RECONNECT_DELAY = 5


GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _banner(msg: str, colour: str = CYAN):
    print(f"\n{BOLD}{colour}{'='*60}{RESET}")
    print(f"{BOLD}{colour}  {msg}{RESET}")
    print(f"{BOLD}{colour}{'='*60}{RESET}")

def _log(tag: str, msg: str, colour: str = RESET):
    timestamp = time.strftime("%H:%M:%S")
    print(f"  {BOLD}{BLUE}[{timestamp}][{tag}]{RESET} {colour}{msg}{RESET}")

# =============================================================================
#  MQTT CALLBACKS
# =============================================================================

def on_connect(client, userdata, flags, rc):
    """
    Called automatically when the client connects to the broker.
    rc = 0 → success | rc != 0 → connection refused (check broker/port)
    """
    if rc == 0:
        _banner("✅  Connected to MQTT Broker", GREEN)
        _log("MQTT", f"Broker : {BROKER}:{PORT}",       GREEN)
        _log("MQTT", f"Sub    : {SENSOR_TOPIC}",         GREEN)
        _log("MQTT", f"Pub    : {ACTION_TOPIC}",         GREEN)
        _log("MQTT", "Waiting for sensor data...\n",     GREEN)

        # Subscribe to the sensor topic INSIDE on_connect
        # (re-subscribes automatically after any reconnect)
        client.subscribe(SENSOR_TOPIC)
    else:
        _log("MQTT", f"Connection failed — return code: {rc}", RED)


def on_disconnect(client, userdata, rc):
    """
    Called automatically when the client disconnects from the broker.
    If unexpected (rc != 0), it will try to reconnect.
    """
    if rc == 0:
        _log("MQTT", "Disconnected cleanly.", YELLOW)
    else:
        _log("MQTT", f"Unexpected disconnection (rc={rc}). Reconnecting in {RECONNECT_DELAY}s...", RED)
        time.sleep(RECONNECT_DELAY)
        # paho-mqtt auto-reconnect is handled by loop_forever()


def on_message(client, userdata, msg):
    """
    Called automatically every time a message arrives on a subscribed topic.
    This is the core callback — it:
      1. Decodes the incoming JSON from ESP32
      2. Validates required sensor keys
      3. Calls run_pipeline() to get the action decision
      4. Strips _meta (not needed by ESP32)
      5. Publishes the action JSON back to ESP32
    """
    print(f"\n{BOLD}{'─'*60}{RESET}")
    _log("RECV", f"Topic: {msg.topic}", CYAN)

    # ── Step 1: Decode JSON safely ────────────────────────────────────────────
    try:
        payload_str = msg.payload.decode("utf-8")
        sensor_data = json.loads(payload_str)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        _log("ERROR", f"Invalid JSON received — skipping. ({e})", RED)
        return

    # ── Step 2: Validate required sensor keys ────────────────────────────────
    missing_keys = [k for k in REQUIRED_KEYS if k not in sensor_data]
    if missing_keys:
        _log("ERROR", f"Missing sensor fields: {missing_keys} — skipping.", RED)
        return

    # ── Step 3: Log incoming sensor data ─────────────────────────────────────
    _log("DATA", "Sensor reading received:", GREEN)
    for key in REQUIRED_KEYS:
        print(f"         {key:<20}: {sensor_data[key]}")

    # ── Step 4: Run AI pipeline ───────────────────────────────────────────────
    try:
        result = run_pipeline(sensor_data, verbose=True)
    except Exception as e:
        _log("ERROR", f"Pipeline error — {e}", RED)
        return

    # ── Step 5: Strip _meta (ESP32 only needs actuator commands) ──────────────
    action = {k: v for k, v in result.items() if k != "_meta"}

    # ── Step 6: Publish action back to ESP32 ──────────────────────────────────
    try:
        action_json = json.dumps(action)
        client.publish(ACTION_TOPIC, action_json)
        _log("SEND", f"Action published to [{ACTION_TOPIC}]:", GREEN)
        for actuator, state in action.items():
            colour = RED if state in ("ON", "OPEN") else GREEN
            print(f"         {actuator:<12}: {colour}{BOLD}{state}{RESET}")
    except Exception as e:
        _log("ERROR", f"Failed to publish action — {e}", RED)

    # ── Future improvement: log to sensor_log.csv ─────────────────────────────
    # Uncomment the block below when ready to enable logging:
    #
    # import csv, os
    # LOG_FILE = "data/sensor_log.csv"
    # os.makedirs("data", exist_ok=True)
    # file_exists = os.path.isfile(LOG_FILE)
    # with open(LOG_FILE, "a", newline="") as f:
    #     writer = csv.DictWriter(f, fieldnames=REQUIRED_KEYS + list(action.keys()) + ["timestamp"])
    #     if not file_exists:
    #         writer.writeheader()
    #     writer.writerow({**sensor_data, **action, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})

    print(f"{BOLD}{'─'*60}{RESET}")


# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main function:
      - Creates the MQTT client
      - Registers all callbacks
      - Connects to the broker
      - Starts the blocking network loop (runs until Ctrl+C)
    """
    _banner("Smart Adaptive Environment — MQTT Client", CYAN)
    _log("INIT", f"Connecting to broker: {BROKER}:{PORT} ...", YELLOW)

    # ── Create client instance ────────────────────────────────────────────────
    # client_id="" → auto-generate unique ID (avoids conflicts on public broker)
    client = mqtt.Client(client_id="", clean_session=True)

    # ── Reconnect policy: wait 1s first, max 10s between attempts ────────────
    client.reconnect_delay_set(min_delay=1, max_delay=10)

    # ── Register callbacks ────────────────────────────────────────────────────
    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect
    client.on_message    = on_message

    # ── Connect to broker ─────────────────────────────────────────────────────
    try:
        client.connect(BROKER, PORT, keepalive=60)
    except Exception as e:
        _log("ERROR", f"Cannot reach broker: {e}", RED)
        _log("HINT",  "Check your internet connection or broker address.", YELLOW)
        return

    # ── Start blocking network loop ───────────────────────────────────────────
    # loop_forever():
    #   ✔ keeps connection alive
    #   ✔ handles automatic reconnect on drop
    #   ✔ blocks here until Ctrl+C
    try:
        _log("LOOP", "Entering network loop — press Ctrl+C to stop.\n", YELLOW)
        client.loop_forever()
    except KeyboardInterrupt:
        _banner("🛑  Shutting down MQTT client...", YELLOW)
        client.disconnect()
        _log("BYE", "Disconnected cleanly. Goodbye!", GREEN)


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
