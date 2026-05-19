import { useCallback, useEffect, useMemo, useState } from 'react';

export interface SensorSnapshot {
  Timestamp: string | null;
  Temp_C: number;
  Humidity_pct: number;
  Gas_AQI: number;
  Light_Lux: number;
  Motion_Detected: number;
}

export interface ActuatorState {
  fan: string;
  alarm: string;
  servo: string;
  buzzer: string;
  rgb_led: string;
  action_id: number;
}

export interface GuardianMeta {
  state_label: string;
  risk: string;
  risk_score: number;
  scenario_id: number;
  scenario: string;
  cluster_id: number;
  gas_pred: number;
  temp_pred: number;
  humidity_level: number;
  light_level: number;
  trend: number;
  raw_trend: number;
  spatial_risk: number;
  anomaly_score: number;
  is_anomaly: boolean;
  guard_confidence: number;
  reward: number;
  action_id: number;
}

export interface HistoryPoint {
  timestamp: string;
  Temp_C: number;
  Humidity_pct: number;
  Gas_AQI: number;
  Light_Lux: number;
  Motion_Detected: number;
  state_label: string;
  risk: string;
  risk_score: number;
  scenario: string;
  is_anomaly: boolean;
  action_id: number;
}

export interface DashboardLog {
  id: string;
  time: string;
  message: string;
  type: 'info' | 'warning' | 'success';
}

export interface GuardianMetrics {
  accuracy: number;
  scenarioAccuracy: number;
  falseAlertRate: number;
  warningMissRate: number;
  pcaExplained: number;
  trainRows: number;
  testRows: number;
  gnnAccuracy: number;
  art2Categories?: number;
  rbfSigma?: number;
  gaScore?: number;
  gnnEdges?: number;
}

export interface GuardianState {
  version: number;
  source: string;
  connection: string;
  lastUpdated: string | null;
  mqtt: {
    broker: string;
    port: number;
    sensorTopic: string;
    actionTopic: string;
    modeTopic: string;
    connected: boolean;
    lastPacketAt: string | null;
    error: string | null;
  };
  sensor: SensorSnapshot;
  action: ActuatorState;
  meta: GuardianMeta;
  history: HistoryPoint[];
  logs: DashboardLog[];
  metrics: GuardianMetrics;
  system_mode: 'AI' | 'MANUAL';
}

const defaultSensor: SensorSnapshot = {
  Timestamp: null,
  Temp_C: 24.2,
  Humidity_pct: 55,
  Gas_AQI: 70,
  Light_Lux: 400,
  Motion_Detected: 0,
};

const defaultAction: ActuatorState = {
  fan: 'OFF',
  alarm: 'OFF',
  servo: 'CLOSED',
  buzzer: 'OFF',
  rgb_led: 'GREEN',
  action_id: 0,
};

const defaultMeta: GuardianMeta = {
  state_label: 'Normal',
  risk: 'Safe',
  risk_score: 0,
  scenario_id: 0,
  scenario: 'Normal',
  cluster_id: 0,
  gas_pred: 0.07,
  temp_pred: 0.48,
  humidity_level: 0.55,
  light_level: 0.004,
  trend: 0,
  raw_trend: 0,
  spatial_risk: 0,
  anomaly_score: 0,
  is_anomaly: false,
  guard_confidence: 0.72,
  reward: 0,
  action_id: 0,
};

const defaultMetrics: GuardianMetrics = {
  accuracy: 86.2,
  scenarioAccuracy: 86.2,
  falseAlertRate: 6.1,
  warningMissRate: 0.2,
  pcaExplained: 97.3,
  trainRows: 8064,
  testRows: 2016,
  gnnAccuracy: 94.8,
  art2Categories: 5,
  rbfSigma: 0.98,
  gaScore: 0.4516,
  gnnEdges: 20,
};

export const defaultGuardianState: GuardianState = {
  version: 1,
  source: 'local fallback',
  connection: 'offline',
  lastUpdated: null,
  mqtt: {
    broker: '10.35.93.69',
    port: 1883,
    sensorTopic: 'alg1/sensors',
    actionTopic: 'alg1/actions',
    modeTopic: 'alg1/mode',
    connected: false,
    lastPacketAt: null,
    error: null,
  },
  sensor: defaultSensor,
  action: defaultAction,
  meta: defaultMeta,
  history: createFallbackHistory(),
  logs: [
    {
      id: 'fallback-1',
      time: new Date().toLocaleTimeString([], { hour12: false }),
      message: 'Waiting for dashboard API and MQTT bridge.',
      type: 'info',
    },
  ],
  metrics: defaultMetrics,
  system_mode: 'AI',
};

function createFallbackHistory(): HistoryPoint[] {
  return Array.from({ length: 24 }, (_, index) => {
    const drift = Math.sin(index / 3);
    return {
      timestamp: new Date(Date.now() - (24 - index) * 5000).toISOString(),
      Temp_C: 24.2 + drift * 0.6,
      Humidity_pct: 55 + drift * 1.8,
      Gas_AQI: 70 + drift * 4,
      Light_Lux: 400 + drift * 45,
      Motion_Detected: 0,
      state_label: 'Normal',
      risk: 'Safe',
      risk_score: 8,
      scenario: 'Normal',
      is_anomaly: false,
      action_id: 0,
    };
  });
}

function normaliseGuardianState(input: Partial<GuardianState>): GuardianState {
  return {
    ...defaultGuardianState,
    ...input,
    mqtt: { ...defaultGuardianState.mqtt, ...(input.mqtt || {}) },
    sensor: { ...defaultGuardianState.sensor, ...(input.sensor || {}) },
    action: { ...defaultGuardianState.action, ...(input.action || {}) },
    meta: { ...defaultGuardianState.meta, ...(input.meta || {}) },
    metrics: { ...defaultGuardianState.metrics, ...(input.metrics || {}) },
    history: Array.isArray(input.history) && input.history.length ? input.history : defaultGuardianState.history,
    logs: Array.isArray(input.logs) && input.logs.length ? input.logs : defaultGuardianState.logs,
  };
}

async function fetchState(): Promise<GuardianState> {
  const response = await fetch('/api/state', { cache: 'no-store' });
  if (!response.ok) throw new Error(`Dashboard API returned ${response.status}`);
  return normaliseGuardianState(await response.json());
}

export function useGuardianTelemetry() {
  const [state, setState] = useState<GuardianState>(defaultGuardianState);
  const [apiReachable, setApiReachable] = useState(false);
  const [lastError, setLastError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const nextState = await fetchState();
        if (!cancelled) {
          setState((prevState) => {
            if (prevState.lastUpdated === nextState.lastUpdated && prevState.system_mode === nextState.system_mode && prevState.action.action_id === nextState.action.action_id) {
              return prevState;
            }
            return nextState;
          });
          setApiReachable(true);
          setLastError(null);
        }
      } catch (error) {
        if (!cancelled) {
          setApiReachable(false);
          setLastError(error instanceof Error ? error.message : 'Dashboard API unavailable');
        }
      }
    };

    load();
    const poll = window.setInterval(load, 10000);
    const events = new EventSource('/api/events');

    const handleState = (event: MessageEvent<string>) => {
      try {
        const nextState = normaliseGuardianState(JSON.parse(event.data));
        setState((prevState) => {
          if (prevState.lastUpdated === nextState.lastUpdated && prevState.system_mode === nextState.system_mode && prevState.action.action_id === nextState.action.action_id) {
            return prevState;
          }
          return nextState;
        });
        setApiReachable(true);
        setLastError(null);
      } catch (error) {
        setLastError(error instanceof Error ? error.message : 'Bad live event payload');
      }
    };

    events.addEventListener('state', handleState as EventListener);
    events.onerror = () => {
      setApiReachable(false);
    };

    return () => {
      cancelled = true;
      window.clearInterval(poll);
      events.close();
    };
  }, []);

  const publishManualMode = useCallback(async (actionId: number) => {
    const response = await fetch('/api/manual', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action_id: actionId }),
    });
    if (!response.ok) throw new Error(`Manual override failed with ${response.status}`);
    const payload = await response.json();
    if (payload.state) setState(normaliseGuardianState(payload.state));
    return payload;
  }, []);

  const publishSystemMode = useCallback(async (mode: 'AI' | 'MANUAL') => {
    const response = await fetch('/api/mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode }),
    });
    if (!response.ok) throw new Error(`System mode switch failed with ${response.status}`);
    const payload = await response.json();
    if (payload.state) setState(normaliseGuardianState(payload.state));
    return payload;
  }, []);

  return useMemo(
    () => ({ state, apiReachable, lastError, publishManualMode, publishSystemMode }),
    [state, apiReachable, lastError, publishManualMode, publishSystemMode],
  );
}
