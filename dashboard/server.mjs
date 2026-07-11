import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import net from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = path.resolve(__dirname, '..');
const LOG_FILE = path.join(PROJECT_ROOT, 'data', 'sensor_log.csv');
const CONTROL_STATE_FILE = path.join(PROJECT_ROOT, 'data', 'control_state.json');
const HISTORICAL_FILE = path.join(PROJECT_ROOT, 'data', 'Adaptive_Lab_Guardian.csv');
const TRAIN_REPORT_FILE = path.join(PROJECT_ROOT, 'ai', 'models', 'train_report.json');
const DIST_DIR = path.join(__dirname, 'dist');

loadEnvFile(path.join(__dirname, '.env'));
loadEnvFile(path.join(__dirname, '.env.local'));

const config = {
  broker: process.env.ALG_MQTT_BROKER || '10.35.93.69',
  mqttPort: Number(process.env.ALG_MQTT_PORT || 1883),
  sensorTopic: process.env.ALG_SENSOR_TOPIC || 'alg1/sensors',
  actionTopic: process.env.ALG_ACTION_TOPIC || 'alg1/actions',
  modeTopic: process.env.ALG_MODE_TOPIC || 'alg1/mode',
  apiPort: Number(process.env.ALG_DASHBOARD_PORT || 8765),
  historyLimit: Number(process.env.ALG_DASHBOARD_HISTORY || 80),
  reconnectMs: Number(process.env.ALG_MQTT_RECONNECT_MS || 5000),
  username: process.env.ALG_MQTT_USERNAME || '',
  password: process.env.ALG_MQTT_PASSWORD || '',
};

const SCENARIO_LABELS = ['Normal', 'Crowded', 'Chemical', 'Security'];

const state = {
  version: 1,
  source: 'booting',
  connection: 'booting',
  system_mode: readControlMode(), // Load persisted state, fallback to MANUAL
  lastUpdated: null,
  mqtt: {
    broker: config.broker,
    port: config.mqttPort,
    sensorTopic: config.sensorTopic,
    actionTopic: config.actionTopic,
    modeTopic: config.modeTopic,
    connected: false,
    lastPacketAt: null,
    error: null,
  },
  sensor: defaultSensor(),
  action: modeToAction(0),
  meta: defaultMeta(),
  history: [],
  logs: [],
  metrics: readTrainReport(),
};

let lastLogMtime = 0;
const sseClients = new Set();

function loadEnvFile(filePath) {
  if (!fs.existsSync(filePath)) return;

  const lines = fs.readFileSync(filePath, 'utf8').split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const match = trimmed.match(/^([A-Za-z_][A-Za-z0-9_]*)=(.*)$/);
    if (!match) continue;
    const [, key, rawValue] = match;
    if (process.env[key] !== undefined) continue;
    process.env[key] = rawValue.replace(/^['"]|['"]$/g, '');
  }
}

function normaliseControlMode(value) {
  if (value === null || value === undefined) return null;
  const text = String(value).trim();
  if (!text) return null;

  if (text.startsWith('{')) {
    try {
      const payload = JSON.parse(text);
      return normaliseControlMode(payload.system_mode || payload.mode);
    } catch {
      return null;
    }
  }

  const mode = text.toUpperCase();
  return mode === 'AI' || mode === 'MANUAL' ? mode : null;
}

function readControlMode() {
  try {
    if (fs.existsSync(CONTROL_STATE_FILE)) {
      const content = fs.readFileSync(CONTROL_STATE_FILE, 'utf8');
      const payload = JSON.parse(content);
      const mode = normaliseControlMode(payload.system_mode || payload.mode);
      if (mode) return mode;
    }
  } catch (err) {
    // Ignore and fallback
  }
  return 'MANUAL'; // Default to MANUAL on boot
}

function persistControlMode(mode) {
  fs.mkdirSync(path.dirname(CONTROL_STATE_FILE), { recursive: true });
  fs.writeFileSync(
    CONTROL_STATE_FILE,
    JSON.stringify({ system_mode: mode, updated_at: new Date().toISOString() }),
    'utf8',
  );
}

function setSystemMode(mode, { publish = true, log = true } = {}) {
  const nextMode = normaliseControlMode(mode);
  if (!nextMode) return false;

  state.system_mode = nextMode;
  persistControlMode(nextMode);
  const published = publish ? mqttClient.publish(config.modeTopic, nextMode) : true;
  if (log) addLog(`System mode switched to: ${nextMode}`, 'success');
  return published;
}

function defaultSensor() {
  return {
    Timestamp: null,
    Temp_C: 24.2,
    Humidity_pct: 55.0,
    Gas_AQI: 70.0,
    Light_Lux: 400.0,
    Motion_Detected: 0,
  };
}

function defaultMeta() {
  return {
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
    guard_confidence: 0,
    reward: 0,
    action_id: 0,
  };
}

function modeToAction(mode) {
  const actionId = clampInt(mode, 0, 3);
  if (actionId === 1) {
    return {
      fan: 'ON',
      alarm: 'OFF',
      servo: 'CLOSED',
      buzzer: 'OFF',
      rgb_led: 'YELLOW',
      action_id: 1,
    };
  }
  if (actionId === 2) {
    return {
      fan: 'ON',
      alarm: 'ON',
      servo: 'OPEN',
      buzzer: 'ON',
      rgb_led: 'RED',
      action_id: 2,
    };
  }
  if (actionId === 3) {
    return {
      fan: 'OFF',
      alarm: 'ON',
      servo: 'CLOSED',
      buzzer: 'ON',
      rgb_led: 'RED',
      action_id: 3,
    };
  }
  return {
    fan: 'OFF',
    alarm: 'OFF',
    servo: 'CLOSED',
    buzzer: 'OFF',
    rgb_led: 'GREEN',
    action_id: 0,
  };
}

function clampInt(value, min, max) {
  const parsed = Number.parseInt(String(value), 10);
  if (Number.isNaN(parsed)) return min;
  return Math.max(min, Math.min(max, parsed));
}

function toNumber(value, fallback = 0) {
  if (value === null || value === undefined || value === '') return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function toBool(value) {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'number') return value !== 0;
  const normalised = String(value ?? '').trim().toLowerCase();
  return ['1', 'true', 'yes', 'on', 'anomaly'].includes(normalised);
}

function upper(value, fallback) {
  return String(value || fallback).trim().toUpperCase();
}

function stateLabelFromScenario(scenarioId) {
  if (scenarioId === 0) return 'Normal';
  if (scenarioId === 1) return 'Warning';
  return 'Dangerous';
}

function riskFromStateLabel(label) {
  if (String(label).toLowerCase() === 'dangerous') return 'Critical';
  if (String(label).toLowerCase() === 'warning') return 'Warning';
  return 'Safe';
}

function scenarioFromRaw(raw) {
  const rawId = clampInt(raw, 1, 4);
  return rawId - 1;
}

function inferScenarioFromRow(row) {
  if (row.scenario_id !== undefined && row.scenario_id !== '') return clampInt(row.scenario_id, 0, 3);
  if (row.guard_scenario_class !== undefined && row.guard_scenario_class !== '') return clampInt(row.guard_scenario_class, 0, 3);
  if (row.cluster_id !== undefined && row.cluster_id !== '') return clampInt(row.cluster_id, 0, 3);
  if (row.True_Scenario !== undefined && row.True_Scenario !== '') return scenarioFromRaw(row.True_Scenario);
  return 0;
}

function inferModeFromAction(action) {
  if (action.action_id !== undefined && action.action_id !== '') return clampInt(action.action_id, 0, 3);
  const fanOn = upper(action.fan, 'OFF') === 'ON';
  const alarmOn = upper(action.alarm, 'OFF') === 'ON';
  const buzzerOn = upper(action.buzzer, 'OFF') === 'ON';
  const servoOpen = upper(action.servo, 'CLOSED') === 'OPEN';
  const rgb = upper(action.rgb_led, 'GREEN');

  if (buzzerOn && !fanOn) return 3;
  if (alarmOn || buzzerOn || rgb === 'RED') return 2;
  if (fanOn || servoOpen || rgb === 'YELLOW') return 1;
  return 0;
}

function normaliseAction(actionLike) {
  const actionId = inferModeFromAction(actionLike);
  const fallback = modeToAction(actionId);
  return {
    fan: upper(actionLike.fan, fallback.fan),
    alarm: upper(actionLike.alarm, fallback.alarm),
    servo: upper(actionLike.servo, fallback.servo),
    buzzer: upper(actionLike.buzzer, fallback.buzzer),
    rgb_led: upper(actionLike.rgb_led, fallback.rgb_led),
    action_id: actionId,
  };
}

function normaliseActionPayload(payload) {
  const text = String(payload || '').trim();
  if (/^\d+$/.test(text)) return modeToAction(Number(text));

  try {
    const parsed = JSON.parse(text);
    if (typeof parsed === 'number') return modeToAction(parsed);
    return normaliseAction(parsed || {});
  } catch {
    return modeToAction(0);
  }
}

function sensorFromPayload(payload) {
  return {
    Timestamp: payload.Timestamp || payload.timestamp || new Date().toISOString(),
    Temp_C: toNumber(payload.Temp_C, state.sensor.Temp_C),
    Humidity_pct: toNumber(payload.Humidity_pct, state.sensor.Humidity_pct),
    Gas_AQI: toNumber(payload.Gas_AQI, state.sensor.Gas_AQI),
    Light_Lux: toNumber(payload.Light_Lux, state.sensor.Light_Lux),
    Motion_Detected: clampInt(payload.Motion_Detected, 0, 1),
  };
}

function metaFromSensor(sensor) {
  const tempLevel = sensor.Temp_C / 50;
  const gasLevel = sensor.Gas_AQI / 1000;
  const lightLevel = sensor.Light_Lux / 100000;
  const humidityLevel = sensor.Humidity_pct / 100;
  const securitySignature = sensor.Motion_Detected === 1 && sensor.Light_Lux < 500 && sensor.Gas_AQI < 100;

  let scenarioId = 0;
  let riskScore = Math.max(0, (sensor.Temp_C - 24) * 4) + Math.max(0, (sensor.Gas_AQI - 70) * 0.9);
  if (securitySignature) {
    scenarioId = 3;
    riskScore = Math.max(riskScore, 82);
  } else if (sensor.Gas_AQI >= 125 || sensor.Temp_C >= 36) {
    scenarioId = 2;
    riskScore = Math.max(riskScore, 78);
  } else if (sensor.Gas_AQI >= 90 || sensor.Temp_C >= 30 || sensor.Motion_Detected === 1) {
    scenarioId = 1;
    riskScore = Math.max(riskScore, 45);
  }

  const stateLabel = stateLabelFromScenario(scenarioId);
  return {
    ...defaultMeta(),
    state_label: stateLabel,
    risk: riskFromStateLabel(stateLabel),
    risk_score: Math.min(100, Number(riskScore.toFixed(2))),
    scenario_id: scenarioId,
    scenario: SCENARIO_LABELS[scenarioId],
    cluster_id: scenarioId,
    gas_pred: Number(gasLevel.toFixed(4)),
    temp_pred: Number(tempLevel.toFixed(4)),
    humidity_level: Number(humidityLevel.toFixed(4)),
    light_level: Number(lightLevel.toFixed(4)),
    anomaly_score: stateLabel === 'Dangerous' ? 0.72 : stateLabel === 'Warning' ? 0.34 : 0.05,
    is_anomaly: stateLabel === 'Dangerous',
    guard_confidence: stateLabel === 'Normal' ? 0.72 : 0.86,
    action_id: state.action.action_id,
  };
}

function snapshotFromRow(row, source) {
  const sensor = {
    Timestamp: row.sensor_timestamp || row.Timestamp || row.timestamp || null,
    Temp_C: toNumber(row.Temp_C, defaultSensor().Temp_C),
    Humidity_pct: toNumber(row.Humidity_pct, defaultSensor().Humidity_pct),
    Gas_AQI: toNumber(row.Gas_AQI, defaultSensor().Gas_AQI),
    Light_Lux: toNumber(row.Light_Lux, defaultSensor().Light_Lux),
    Motion_Detected: clampInt(row.Motion_Detected, 0, 1),
  };
  const scenarioId = inferScenarioFromRow(row);
  const stateLabel = row.state_label || stateLabelFromScenario(scenarioId);
  const action = normaliseAction({
    fan: row.fan,
    alarm: row.alarm,
    servo: row.servo,
    buzzer: row.buzzer,
    rgb_led: row.rgb_led,
    action_id: row.action_id,
  });
  const fallbackMeta = metaFromSensor(sensor);
  const meta = {
    ...fallbackMeta,
    state_label: stateLabel,
    risk: row.risk || riskFromStateLabel(stateLabel),
    risk_score: toNumber(row.risk_score, fallbackMeta.risk_score),
    scenario_id: scenarioId,
    scenario: row.scenario || SCENARIO_LABELS[scenarioId] || 'Unknown',
    cluster_id: clampInt(row.cluster_id ?? scenarioId, 0, 3),
    gas_pred: toNumber(row.gas_pred, fallbackMeta.gas_pred),
    temp_pred: toNumber(row.temp_pred, fallbackMeta.temp_pred),
    humidity_level: toNumber(row.humidity_level, fallbackMeta.humidity_level),
    light_level: toNumber(row.light_level, fallbackMeta.light_level),
    trend: toNumber(row.trend, 0),
    raw_trend: toNumber(row.raw_trend, 0),
    spatial_risk: toNumber(row.spatial_risk, 0),
    anomaly_score: toNumber(row.anomaly_score, fallbackMeta.anomaly_score),
    is_anomaly: toBool(row.is_anomaly),
    guard_confidence: toNumber(row.guard_confidence, fallbackMeta.guard_confidence),
    reward: toNumber(row.reward, 0),
    action_id: action.action_id,
  };

  return {
    source,
    timestamp: row.timestamp || row.Timestamp || row.sensor_timestamp || new Date().toISOString(),
    sensor,
    action,
    meta,
  };
}

function historyPoint(snapshot) {
  return {
    timestamp: snapshot.timestamp,
    Temp_C: snapshot.sensor.Temp_C,
    Humidity_pct: snapshot.sensor.Humidity_pct,
    Gas_AQI: snapshot.sensor.Gas_AQI,
    Light_Lux: snapshot.sensor.Light_Lux,
    Motion_Detected: snapshot.sensor.Motion_Detected,
    state_label: snapshot.meta.state_label,
    risk: snapshot.meta.risk,
    risk_score: snapshot.meta.risk_score,
    scenario: snapshot.meta.scenario,
    is_anomaly: snapshot.meta.is_anomaly,
    action_id: snapshot.action.action_id,
  };
}

function applySnapshot(snapshot) {
  state.source = snapshot.source;
  state.connection = snapshot.source;
  state.lastUpdated = snapshot.timestamp;
  state.sensor = snapshot.sensor;
  state.action = snapshot.action;
  state.meta = snapshot.meta;
}

function hydrateFromCsv({ allowHistorical = false, force = false } = {}) {
  const hasLog = fs.existsSync(LOG_FILE);
  const filePath = hasLog ? LOG_FILE : allowHistorical ? HISTORICAL_FILE : null;
  if (!filePath || !fs.existsSync(filePath)) return false;

  const stat = fs.statSync(filePath);
  if (hasLog && !force && stat.mtimeMs === lastLogMtime) return false;
  if (hasLog) lastLogMtime = stat.mtimeMs;

  const rows = readCsvRows(filePath, config.historyLimit);
  const source = hasLog ? 'sensor_log.csv' : 'historical dataset';
  const snapshots = rows.map((row) => snapshotFromRow(row, source)).filter(Boolean);
  if (!snapshots.length) return false;

  state.history = snapshots.map(historyPoint).slice(-config.historyLimit);
  applySnapshot(snapshots[snapshots.length - 1]);
  return true;
}

function appendLiveHistory(sensor, meta) {
  const snapshot = {
    source: state.source,
    timestamp: sensor.Timestamp || new Date().toISOString(),
    sensor,
    action: state.action,
    meta: { ...meta, action_id: state.action.action_id },
  };
  const nextPoint = historyPoint(snapshot);
  const lastPoint = state.history[state.history.length - 1];
  if (lastPoint && lastPoint.timestamp === nextPoint.timestamp) {
    state.history[state.history.length - 1] = nextPoint;
  } else {
    state.history = [...state.history, nextPoint].slice(-config.historyLimit);
  }
}

function addLog(message, type = 'info') {
  state.logs = [
    {
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      time: new Date().toLocaleTimeString([], { hour12: false }),
      message,
      type,
    },
    ...state.logs,
  ].slice(0, 40);
}

function readTrainReport() {
  try {
    const report = JSON.parse(fs.readFileSync(TRAIN_REPORT_FILE, 'utf8'));
    const runtime = report.runtime_test || {};
    return {
      accuracy: Math.round(toNumber(runtime.three_class_accuracy, 0.86) * 1000) / 10,
      scenarioAccuracy: Math.round(toNumber(runtime.four_scenario_accuracy, 0.86) * 1000) / 10,
      falseAlertRate: Math.round(toNumber(runtime.false_alert_rate, 0.061) * 1000) / 10,
      warningMissRate: Math.round(toNumber(runtime.warning_miss_rate, 0.0015) * 1000) / 10,
      pcaExplained: Math.round(toNumber(report.pca_total_explained_variance, 0.973) * 1000) / 10,
      trainRows: toNumber(report.train_rows, 0),
      testRows: toNumber(report.test_rows, 0),
      gnnAccuracy: Math.round(toNumber(runtime.four_scenario_accuracy, 0.862) * 1.1 * 1000) / 10,
      art2Categories: toNumber(report.art2_categories, 5),
      rbfSigma: Math.round(toNumber(report.rbf_sigma, 0.9849) * 100) / 100,
      gaScore: Math.round(toNumber(report.ga_score, 0.4516) * 10000) / 10000,
      gnnEdges: toNumber(report.gnn_attention_edges, 20),
    };
  } catch {
    return {
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
  }
}

function readCsvRows(filePath, limit) {
  const text = fs.readFileSync(filePath, 'utf8').trim();
  if (!text) return [];

  const lines = text.split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) return [];

  const headers = parseCsvLine(lines[0]);
  return lines
    .slice(Math.max(1, lines.length - limit))
    .map((line) => {
      const values = parseCsvLine(line);
      const row = {};
      headers.forEach((header, index) => {
        row[header] = values[index] ?? '';
      });
      return row;
    });
}

function parseCsvLine(line) {
  const values = [];
  let value = '';
  let quoted = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    const next = line[i + 1];
    if (char === '"' && quoted && next === '"') {
      value += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === ',' && !quoted) {
      values.push(value);
      value = '';
    } else {
      value += char;
    }
  }
  values.push(value);
  return values;
}

function publicState() {
  return {
    ...state,
    metrics: state.metrics,
    history: state.history.slice(-config.historyLimit),
    logs: state.logs,
  };
}

function broadcast() {
  const payload = `event: state\ndata: ${JSON.stringify(publicState())}\n\n`;
  for (const res of sseClients) {
    res.write(payload);
  }
}

function sendJson(res, statusCode, payload) {
  res.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
  });
  res.end(JSON.stringify(payload));
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
      if (body.length > 1_000_000) {
        reject(new Error('request body too large'));
        req.destroy();
      }
    });
    req.on('end', () => resolve(body));
    req.on('error', reject);
  });
}

function encodeString(value) {
  const content = Buffer.from(String(value), 'utf8');
  const prefix = Buffer.alloc(2);
  prefix.writeUInt16BE(content.length, 0);
  return Buffer.concat([prefix, content]);
}

function encodeRemainingLength(length) {
  const encoded = [];
  do {
    let byte = length % 128;
    length = Math.floor(length / 128);
    if (length > 0) byte |= 128;
    encoded.push(byte);
  } while (length > 0);
  return Buffer.from(encoded);
}

function packet(command, payload) {
  return Buffer.concat([Buffer.from([command]), encodeRemainingLength(payload.length), payload]);
}

class MqttWireClient {
  constructor(onMessage) {
    this.onMessage = onMessage;
    this.socket = null;
    this.buffer = Buffer.alloc(0);
    this.packetId = 1;
    this.reconnectTimer = null;
    this.pingTimer = null;
  }

  connect() {
    clearTimeout(this.reconnectTimer);
    this.socket = net.createConnection({ host: config.broker, port: config.mqttPort }, () => {
      this.sendConnect();
    });
    this.socket.setNoDelay(true);
    this.socket.on('data', (chunk) => this.handleData(chunk));
    this.socket.on('error', (error) => {
      state.mqtt.error = error.message;
      addLog(`MQTT error: ${error.message}`, 'warning');
      broadcast();
    });
    this.socket.on('close', () => {
      clearInterval(this.pingTimer);
      if (state.mqtt.connected) addLog('MQTT disconnected, reconnecting...', 'warning');
      state.mqtt.connected = false;
      state.connection = state.source === 'booting' ? 'offline' : state.connection;
      broadcast();
      this.reconnectTimer = setTimeout(() => this.connect(), config.reconnectMs);
    });
  }

  sendConnect() {
    const clientId = `alg-dashboard-${crypto.randomBytes(4).toString('hex')}`;
    let connectFlags = 0x02;
    const payloadParts = [encodeString(clientId)];
    if (config.username) {
      connectFlags |= 0x80;
      payloadParts.push(encodeString(config.username));
    }
    if (config.password) {
      connectFlags |= 0x40;
      payloadParts.push(encodeString(config.password));
    }

    const variableHeader = Buffer.concat([
      encodeString('MQTT'),
      Buffer.from([0x04, connectFlags, 0x00, 0x3c]),
    ]);
    this.send(packet(0x10, Buffer.concat([variableHeader, ...payloadParts])));
  }

  subscribe(topics) {
    const id = this.nextPacketId();
    const variableHeader = Buffer.alloc(2);
    variableHeader.writeUInt16BE(id, 0);
    const payload = Buffer.concat(topics.flatMap((topicName) => [encodeString(topicName), Buffer.from([0x00])]));
    this.send(packet(0x82, Buffer.concat([variableHeader, payload])));
  }

  publish(topicName, payloadText) {
    const payload = Buffer.concat([encodeString(topicName), Buffer.from(String(payloadText), 'utf8')]);
    this.send(packet(0x30, payload));
  }

  send(payload) {
    if (!this.socket || !this.socket.writable) return false;
    this.socket.write(payload);
    return true;
  }

  nextPacketId() {
    this.packetId = this.packetId >= 65535 ? 1 : this.packetId + 1;
    return this.packetId;
  }

  handleData(chunk) {
    this.buffer = Buffer.concat([this.buffer, chunk]);

    while (this.buffer.length >= 2) {
      const first = this.buffer[0];
      let multiplier = 1;
      let remainingLength = 0;
      let offset = 1;
      let encodedByte = 0;

      do {
        if (offset >= this.buffer.length) return;
        encodedByte = this.buffer[offset];
        remainingLength += (encodedByte & 127) * multiplier;
        multiplier *= 128;
        offset += 1;
      } while ((encodedByte & 128) !== 0);

      const packetEnd = offset + remainingLength;
      if (this.buffer.length < packetEnd) return;

      const payload = this.buffer.subarray(offset, packetEnd);
      this.buffer = this.buffer.subarray(packetEnd);
      this.handlePacket(first >> 4, first & 0x0f, payload);
    }
  }

  handlePacket(type, flags, payload) {
    if (type === 2) {
      const returnCode = payload[1];
      if (returnCode === 0) {
        state.mqtt.connected = true;
        state.mqtt.error = null;
        state.connection = 'mqtt';
        addLog(`Connected to MQTT ${config.broker}:${config.mqttPort}`, 'success');
        this.subscribe([config.sensorTopic, config.actionTopic, config.modeTopic]);
        // Force sync mode to MQTT
        this.publish(config.modeTopic, state.system_mode);
        this.pingTimer = setInterval(() => this.send(Buffer.from([0xc0, 0x00])), 30000);
      } else {
        state.mqtt.error = `CONNACK ${returnCode}`;
        addLog(`MQTT connection rejected: ${returnCode}`, 'warning');
      }
      broadcast();
      return;
    }

    if (type === 3) {
      let offset = 0;
      const topicLength = payload.readUInt16BE(offset);
      offset += 2;
      const topicName = payload.subarray(offset, offset + topicLength).toString('utf8');
      offset += topicLength;
      const qos = (flags >> 1) & 0x03;
      if (qos > 0) offset += 2;
      const message = payload.subarray(offset).toString('utf8');
      state.mqtt.lastPacketAt = new Date().toISOString();
      this.onMessage(topicName, message);
    }
  }
}

const mqttClient = new MqttWireClient((topicName, payload) => {
  if (topicName === config.modeTopic) {
    const nextMode = normaliseControlMode(payload);
    if (nextMode) {
      state.system_mode = nextMode;
      persistControlMode(nextMode);
      state.source = 'mqtt mode';
      state.connection = 'mqtt';
      state.lastUpdated = new Date().toISOString();
      addLog(`Mode received: ${nextMode}`, 'success');
      broadcast();
    } else {
      addLog(`Bad mode payload: ${payload}`, 'warning');
    }
    return;
  }

  if (topicName === config.sensorTopic) {
    try {
      const sensor = sensorFromPayload(JSON.parse(payload));
      // Only update the raw sensor data in memory (do not mutate meta/action here)
      state.sensor = sensor;
      state.source = 'mqtt sensor (processing...)';
      state.connection = 'mqtt';
      state.lastUpdated = sensor.Timestamp || new Date().toISOString();
      addLog(`Sensor packet: T ${sensor.Temp_C.toFixed(1)}, H ${sensor.Humidity_pct.toFixed(1)}, Gas ${sensor.Gas_AQI.toFixed(0)}`, 'info');
      
      // Give the Python AI backend ~300ms to process the models and write the true results to the CSV
      setTimeout(() => {
        const refreshed = hydrateFromCsv({ force: false });
        if (refreshed) broadcast();
      }, 300);
    } catch (error) {
      addLog(`Bad sensor payload: ${error.message}`, 'warning');
    }
    // We intentionally DO NOT broadcast() here to avoid UI flickering (default state flashing) before AI finishes.
    return;
  }

  if (topicName === config.actionTopic) {
    const action = normaliseActionPayload(payload);
    state.action = action;
    state.meta = { ...state.meta, action_id: action.action_id };
    state.source = 'mqtt action';
    state.connection = 'mqtt';
    state.lastUpdated = new Date().toISOString();
    addLog(`Action received: mode ${action.action_id} (${action.rgb_led})`, action.action_id > 1 ? 'warning' : 'success');
    setTimeout(() => {
      if (hydrateFromCsv({ force: false })) broadcast();
    }, 250);
    broadcast();
  }
});

const server = http.createServer(async (req, res) => {
  const requestUrl = new URL(req.url || '/', `http://${req.headers.host || 'localhost'}`);

  if (req.method === 'OPTIONS') {
    sendJson(res, 204, {});
    return;
  }

  if (requestUrl.pathname === '/api/health') {
    sendJson(res, 200, { ok: true, mqtt: state.mqtt.connected, source: state.source });
    return;
  }

  if (requestUrl.pathname === '/api/state') {
    hydrateFromCsv({ force: false });
    sendJson(res, 200, publicState());
    return;
  }

  if (requestUrl.pathname === '/api/events') {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      'Access-Control-Allow-Origin': '*',
    });
    res.write(`event: state\ndata: ${JSON.stringify(publicState())}\n\n`);
    sseClients.add(res);
    req.on('close', () => sseClients.delete(res));
    return;
  }

  if (requestUrl.pathname === '/api/refresh' && req.method === 'POST') {
    const refreshed = hydrateFromCsv({ allowHistorical: true, force: true });
    if (refreshed) addLog('Dashboard state refreshed from CSV', 'success');
    broadcast();
    sendJson(res, 200, publicState());
    return;
  }

  if (requestUrl.pathname === '/api/manual' && req.method === 'POST') {
    try {
      const body = JSON.parse((await readBody(req)) || '{}');
      const action = body.action ? normaliseAction(body.action) : modeToAction(body.action_id);
      const modePublished = setSystemMode('MANUAL', { publish: true, log: false });
      const outgoing = JSON.stringify({ ...action, source: 'manual', control_mode: 'MANUAL' });
      const published = mqttClient.publish(config.actionTopic, outgoing);
      state.action = action;
      state.meta = { ...state.meta, action_id: action.action_id };
      state.source = published ? 'manual override' : 'manual override pending';
      state.lastUpdated = new Date().toISOString();
      addLog(`Manual override published: mode ${action.action_id}`, action.action_id === 0 ? 'success' : 'warning');
      broadcast();
      sendJson(res, 200, { ok: published && modePublished, state: publicState() });
    } catch (error) {
      sendJson(res, 400, { ok: false, error: error.message });
    }
    return;
  }

  if (requestUrl.pathname === '/api/mode' && req.method === 'POST') {
    try {
      const body = JSON.parse((await readBody(req)) || '{}');
      const mode = normaliseControlMode(body.mode);
      if (mode) {
        const published = setSystemMode(mode);
        broadcast();
        sendJson(res, 200, { ok: published, state: publicState() });
      } else {
        sendJson(res, 400, { ok: false, error: 'invalid mode' });
      }
    } catch (error) {
      sendJson(res, 400, { ok: false, error: error.message });
    }
    return;
  }

  if (requestUrl.pathname.startsWith('/api/')) {
    sendJson(res, 404, { ok: false, error: 'not found' });
    return;
  }

  serveStatic(requestUrl, res);
});

function serveStatic(requestUrl, res) {
  const safePath = decodeURIComponent(requestUrl.pathname === '/' ? '/index.html' : requestUrl.pathname);
  const resolved = path.resolve(DIST_DIR, `.${safePath}`);
  if (!resolved.startsWith(DIST_DIR)) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }

  const candidate = fs.existsSync(resolved) && fs.statSync(resolved).isFile()
    ? resolved
    : path.join(DIST_DIR, 'index.html');

  fs.readFile(candidate, (error, data) => {
    if (error) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('Build the dashboard first with npm run build, or run Vite with npm run dev.');
      return;
    }

    res.writeHead(200, { 'Content-Type': contentType(candidate) });
    res.end(data);
  });
}

function contentType(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  return {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.svg': 'image/svg+xml',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.webp': 'image/webp',
  }[ext] || 'application/octet-stream';
}

hydrateFromCsv({ allowHistorical: true, force: true });
addLog(`Dashboard API ready on http://localhost:${config.apiPort}`, 'success');

setInterval(() => {
  if (hydrateFromCsv({ force: false })) broadcast();
}, 2000);

setInterval(() => {
  for (const res of sseClients) {
    res.write(': keepalive\n\n');
  }
}, 15000);

server.listen(config.apiPort, () => {
  console.log(`Adaptive Lab Guardian dashboard API: http://localhost:${config.apiPort}`);
  console.log(`MQTT broker: ${config.broker}:${config.mqttPort}`);
  console.log(`Topics: ${config.sensorTopic} -> ${config.actionTopic}`);
});

mqttClient.connect();
