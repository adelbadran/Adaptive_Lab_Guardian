# Adaptive Lab Guardian

Adaptive Lab Guardian is a smart lab monitoring pipeline that reads ESP32 sensor data, filters it, evaluates risk with multiple AI modules, and sends actuator commands back over MQTT.

## Architecture

The code follows the project diagram:

1. **Input**: sensor payload with temperature, humidity, gas AQI, light, and motion.
2. **PCA noise filter**: normalizes and filters the input vector.
3. **Intelligence fan-out**:
   - ART2 detects anomalous patterns.
   - RBF estimates trend pressure and scales it into the fuzzy range `[-5, 5]`.
   - GNN estimates spatial/sensor relationship risk.
   - SOM assigns the current state cluster.
   - A lightweight KNN risk/scenario guard compares live readings with the historical CSV.
4. **Decision**: fuzzy logic creates a baseline action, then RL refines it without escalating non-critical states.
5. **Evolution**: reward metadata is emitted for GA/offline tuning.

## Scenario Mapping

The original `True_Scenario` labels are preserved as four runtime scenarios:

| Raw label | Runtime id | Scenario | Typical data shape |
| --- | ---: | --- | --- |
| `1` | `0` | Normal | Low gas, low motion, usually low light |
| `2` | `1` | Crowded | Higher temperature and high light |
| `3` | `2` | Chemical | Highest gas AQI with high temperature/light |
| `4` | `3` | Security | Very low light, low temperature, high humidity, low gas |

The temporal test split does not contain chemical samples, so `train_report.json` reports chemical test recall as `null` rather than pretending it was tested. Security overlaps strongly with normal in the original sensor columns, so the runtime also treats dark PIR motion as a security breach.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline smoke test:

```bash
python -m ai.main
```

Train and save all runtime model artifacts from the real CSV:

```bash
python -m ai.train_models
```

The training step reports PCA variance, class balancing before/after SMOTE-style resampling, GA thresholds, false-positive/false-negative rates, and held-out runtime accuracy in `ai/models/train_report.json`.

Run the MQTT bridge:

```bash
python -m ai.mqtt_client
```

By default the bridge uses the local broker from the ESP32 sketch:

```bash
ALG_MQTT_BROKER=10.35.93.69
ALG_MQTT_PORT=1883
ALG_SENSOR_TOPIC=alg1/sensors
ALG_ACTION_TOPIC=alg1/actions
```

Set `ALG_MQTT_ACTION_FORMAT=mode` if you want the Python bridge to publish only the numeric ESP32 mode instead of JSON.

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

## MQTT Contract

ESP32 publishes sensor JSON to `alg1/sensors`:

```json
{
  "Temp_C": 24.2,
  "Humidity_pct": 55.0,
  "Gas_AQI": 120.0,
  "Light_Lux": 400.0,
  "Motion_Detected": 0
}
```

Python publishes actions to `alg1/actions`:

```json
{
  "fan": "OFF",
  "alarm": "OFF",
  "servo": "CLOSED",
  "buzzer": "OFF",
  "rgb_led": "GREEN",
  "action_id": 0
}
```

`action_id` maps to the ESP32 modes: `0` normal, `1` ventilation, `2` chemical, `3` security breach. The ESP32 sketch accepts either this JSON payload or a plain numeric mode for backwards compatibility.

## Notes

The runtime is defensive: if trained models are missing from `ai/models`, the pipeline uses deterministic fallbacks so the ESP32, MQTT client, and dashboard can still run end to end.
