# Adaptive Lab Guardian Dashboard

React/Vite dashboard for the Adaptive Lab Guardian project. The visual structure stays the same as the 3D dashboard, but the data now comes from the project MQTT topics and the AI bridge log.

## Data Flow

- ESP32 publishes sensors to `alg1/sensors`.
- Python AI bridge publishes actuator decisions to `alg1/actions` and writes `data/sensor_log.csv`.
- `dashboard/server.mjs` subscribes to both MQTT topics, reads the CSV log when available, and streams dashboard state over `/api/events`.
- React reads `/api/state` and sends manual override modes with `POST /api/manual`.

## Run Locally

Install frontend dependencies:

```bash
npm install
```

Run the API/MQTT bridge and the Vite UI together:

```bash
npm run dev:all
```

Or run them separately:

```bash
npm run api
npm run dev
```

Open the dashboard at:

```text
http://localhost:3000
```

The API bridge listens on `http://localhost:8765` by default.

## Configuration

Copy `.env.example` to `.env.local` and adjust the broker if needed:

```env
ALG_MQTT_BROKER=10.35.93.69
ALG_MQTT_PORT=1883
ALG_SENSOR_TOPIC=alg1/sensors
ALG_ACTION_TOPIC=alg1/actions
ALG_DASHBOARD_PORT=8765
VITE_DASHBOARD_API_URL=http://localhost:8765
```

Manual override buttons publish the same ESP32 modes used by the project:

- `0` normal reset
- `1` ventilation
- `2` chemical alert
- `3` security breach
