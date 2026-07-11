// ╔══════════════════════════════════════════════════════════════════╗
// ║              ESP32 Smart Environment Monitor & Controller        ║
// ║              Sensors: DHT11 · ADS1115 · PIR · MQ135 · LDR      ║
// ║              Actuators: Fan · Buzzer · Servo · RGB LED           ║
// ║              Protocol: WiFi + MQTT                               ║
// ╚══════════════════════════════════════════════════════════════════╝

#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <DHT.h>
#include <ESP32Servo.h>

// ════════════════════════════════════════════════
//  WiFi & MQTT Configuration
// ════════════════════════════════════════════════
// UPDATE THESE TO MATCH YOUR NETWORK:
const char* ssid         = "miles";                // WiFi network name
const char* password     = "@del278AiB";           // WiFi password
const char* mqtt_server  = "172.20.10.3";            // MQTT broker IP — must match dashboard ALG_MQTT_BROKER
const int   mqtt_port    = 1883;                   // MQTT broker port
const char* sensor_topic = "alg1/sensors";
const char* action_topic = "alg1/actions";

// ════════════════════════════════════════════════
//  Pin Definitions
// ════════════════════════════════════════════════
#define PIR_PIN      34
#define DHT_PIN      14
#define DHT_TYPE     DHT11
#define FAN_PIN      33
#define BUZZER_PIN   23
#define SERVO_PIN    19
#define RED1_PIN     25
#define GREEN1_PIN   26
#define BLUE1_PIN    27

// ════════════════════════════════════════════════
//  Objects
// ════════════════════════════════════════════════
WiFiClient    espClient;
PubSubClient  client(espClient);
DHT           dht(DHT_PIN, DHT_TYPE);
Adafruit_ADS1115 ads;
Servo         servo1;

// ════════════════════════════════════════════════
//  Timing
// ════════════════════════════════════════════════
const unsigned long SENSOR_INTERVAL = 5000;    // 5 seconds — matches Code 2
const unsigned long ACTION_DELAY    = 5000;    // debounce for repeated actions
unsigned long lastSensorPublish     = 0;
unsigned long lastActionTime        = 0;

// ════════════════════════════════════════════════
//  State
// ════════════════════════════════════════════════
int  action_id    = 0;
int  lastMode     = -1;
int  lastActionId = -1;
bool windowIsOpen = false;

// ════════════════════════════════════════════════
//  EMA (Exponential Moving Average) Filter
//  alpha = 0.15 — matches Code 2
// ════════════════════════════════════════════════
const float alpha  = 0.15;
float smoothGas    = 0.0;
float smoothLDR    = 0.0;
bool  firstRead    = true;

// ════════════════════════════════════════════════
//  Dataset Bounds (for normalisation output)
// ════════════════════════════════════════════════
const float DATA_AQI_MIN = 57.0;
const float DATA_AQI_MAX = 185.4;
const float DATA_LUX_MAX = 85759.0;

// ════════════════════════════════════════════════
//  Dynamic Calibration Windows
// ════════════════════════════════════════════════
float gasMin   = 500.0;
float gasMax   = 3000.0;
float lightMin = 0.0;
float lightMax = 4095.0;

// ════════════════════════════════════════════════
//  Software Timestamp
// ════════════════════════════════════════════════
int ts_year  = 2026;
int ts_month = 5;
int ts_day   = 2;
int ts_hour  = 0;
int ts_min   = 0;
int ts_sec   = 0;


// ════════════════════════════════════════════════════════════════════
//  Helper: Increment Software Timestamp (+5 s per reading)
// ════════════════════════════════════════════════════════════════════
void incrementTimestamp() {
  ts_sec += 5;                              // +5 s — matches Code 2
  if (ts_sec  >= 60) { ts_sec  = 0; ts_min++;  }
  if (ts_min  >= 60) { ts_min  = 0; ts_hour++; }
  if (ts_hour >= 24) { ts_hour = 0; ts_day++;  }
}


// ════════════════════════════════════════════════════════════════════
//  Helper: EMA Filter
// ════════════════════════════════════════════════════════════════════
float applyEMA(float current, float prev) {
  return (alpha * current) + ((1.0f - alpha) * prev);
}


// ════════════════════════════════════════════════════════════════════
//  Servo: Window Open / Close
// ════════════════════════════════════════════════════════════════════
void moveForward() {
  for (int angle = 0; angle <= 25; angle++) {
    servo1.write(angle);
    client.loop();
    delay(15);
  }
}

void moveBackward() {
  for (int angle = 25; angle >= 0; angle--) {
    servo1.write(angle);
    client.loop();
    delay(15);
  }
  servo1.write(0);
}

void openWindow() {
  if (windowIsOpen) return;
  moveForward();
  windowIsOpen = true;
}

void closeWindow() {
  if (!windowIsOpen) return;
  moveBackward();
  servo1.write(0);
  delay(200);
  windowIsOpen = false;
}


// ════════════════════════════════════════════════════════════════════
//  RGB LED Helper
// ════════════════════════════════════════════════════════════════════
void setRGB1(bool r, bool g, bool b) {
  digitalWrite(RED1_PIN,   r);
  digitalWrite(GREEN1_PIN, g);
  digitalWrite(BLUE1_PIN,  b);
}


// ════════════════════════════════════════════════════════════════════
//  Mode Application
//  ┌─────────┬──────────┬───────────┬────────────┬────────┐
//  │  Mode   │  Fan     │  Buzzer   │  RGB       │ Window │
//  ├─────────┼──────────┼───────────┼────────────┼────────┤
//  │ 0 NORM  │  OFF     │  OFF      │  GREEN     │ Close  │
//  │ 1 CROWD │  ON      │  OFF      │  YELLOW    │ Close  │
//  │ 2 CHEM  │  ON      │  ON       │  RED       │ Open   │
//  │ 3 THEFT │  OFF     │  ON       │  RED       │ Close  │
//  └─────────┴──────────┴───────────┴────────────┴────────┘
// ════════════════════════════════════════════════════════════════════
void applyMode(int mode) {
  Serial.print("[MODE] ");
  Serial.print(lastMode);
  Serial.print(" → ");
  Serial.println(mode);

  switch (mode) {

    case 0: // ── NORMAL ──────────────────────────────────────────
      digitalWrite(FAN_PIN,    LOW);
      digitalWrite(BUZZER_PIN, LOW);
      setRGB1(0, 1, 0);            // 🟢 Green
      closeWindow();
      break;

    case 1: // ── CROWDED ─────────────────────────────────────────
      digitalWrite(FAN_PIN,    HIGH);
      digitalWrite(BUZZER_PIN, LOW);
      setRGB1(1, 1, 0);            // 🟡 Yellow
      closeWindow();
      break;

    case 2: // ── CHEMICAL HAZARD ──────────────────────────────────
      digitalWrite(FAN_PIN,    HIGH);
      digitalWrite(BUZZER_PIN, HIGH);
      setRGB1(1, 0, 0);            // 🔴 Red
      openWindow();
      break;

    case 3: // ── SECURITY / THEFT ─────────────────────────────────
      digitalWrite(FAN_PIN,    LOW);
      digitalWrite(BUZZER_PIN, HIGH);
      setRGB1(1, 0, 0);            // 🔴 Red
      closeWindow();
      break;

    default:
      Serial.println("[MODE] Unknown mode — ignored.");
      break;
  }
}


// ════════════════════════════════════════════════════════════════════
//  WiFi Connection
// ════════════════════════════════════════════════════════════════════
void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");

  int timeout = 0;
  while (WiFi.status() != WL_CONNECTED && timeout < 40) {
    delay(500);
    Serial.print(".");
    timeout++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n========================");
    Serial.println("WiFi Connected!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.print("Signal Strength (RSSI): ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
    Serial.println("========================");
  } else {
    Serial.println("\nWiFi Connection Failed!");
  }
}


// ════════════════════════════════════════════════════════════════════
//  MQTT Reconnect
// ════════════════════════════════════════════════════════════════════
void reconnect() {
  while (!client.connected()) {
    Serial.println("[MQTT] Connecting to broker...");
    if (client.connect("ESP32Client")) {
      Serial.println("[MQTT] Connected!");
      client.subscribe(action_topic);
    } else {
      Serial.print("[MQTT] Failed (rc=");
      Serial.print(client.state());
      Serial.println("). Retrying in 5s...");
      delay(5000);
    }
  }
}


// ════════════════════════════════════════════════════════════════════
//  Parse Action ID from JSON or plain integer
// ════════════════════════════════════════════════════════════════════
int parseActionId(String message) {
  message.trim();

  if (message.startsWith("{")) {
    int keyIdx   = message.indexOf("\"action_id\"");
    if (keyIdx == -1) return action_id;

    int colonIdx = message.indexOf(":", keyIdx);
    if (colonIdx == -1) return action_id;

    int commaIdx = message.indexOf(",", colonIdx);
    int braceIdx = message.indexOf("}", colonIdx);
    int endIdx   = (commaIdx == -1) ? braceIdx : min(commaIdx, braceIdx);
    if (endIdx == -1) return action_id;

    String rawValue = message.substring(colonIdx + 1, endIdx);
    rawValue.trim();
    rawValue.replace("\"", "");
    return constrain(rawValue.toInt(), 0, 3);
  }

  return constrain(message.toInt(), 0, 3);
}


// ════════════════════════════════════════════════════════════════════
//  MQTT Callback
// ════════════════════════════════════════════════════════════════════
void callback(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (unsigned int i = 0; i < length; i++) message += (char)payload[i];

  int nextActionId = parseActionId(message);

  // Debounce: skip if same action received within ACTION_DELAY
  if ((millis() - lastActionTime < ACTION_DELAY) && (nextActionId == lastActionId)) return;

  action_id      = nextActionId;
  lastActionId   = action_id;
  lastActionTime = millis();

  Serial.print("[MQTT] Action received: ");
  Serial.println(action_id);

  applyMode(action_id);
  lastMode = action_id;
}


// ════════════════════════════════════════════════════════════════════
//  Setup
// ════════════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(1000);

  // ── Pin modes ──────────────────────────────────────────────────
  pinMode(PIR_PIN,    INPUT);
  pinMode(FAN_PIN,    OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(RED1_PIN,   OUTPUT);
  pinMode(GREEN1_PIN, OUTPUT);
  pinMode(BLUE1_PIN,  OUTPUT);

  // ── I2C & ADS1115 ──────────────────────────────────────────────
  Wire.begin(21, 22);
  Wire.setTimeout(100);
  delay(100);

  if (!ads.begin()) {
    Serial.println("[ERROR] ADS1115 init failed! Check wiring.");
  } else {
    Serial.println("[OK] ADS1115 initialised.");
  }

  // ── DHT Sensor ─────────────────────────────────────────────────
  dht.begin();
  analogReadResolution(12);

  // ── Servo ──────────────────────────────────────────────────────
  servo1.attach(SERVO_PIN, 500, 2400);
  servo1.write(0);
  windowIsOpen = false;
  delay(1000);

  // ── WiFi & MQTT ────────────────────────────────────────────────
  connectWiFi();
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);

  // ── Initial State ──────────────────────────────────────────────
  applyMode(0);
  lastMode  = 0;
  action_id = 0;

  // ── CSV Header ─────────────────────────────────────────────────
  Serial.println();
  Serial.println("Timestamp,Temp_C,Humidity_pct,Gas_AQI,Light_Lux,Motion_Detected");
}


// ════════════════════════════════════════════════════════════════════
//  Main Loop
// ════════════════════════════════════════════════════════════════════
void loop() {

  // ── WiFi watchdog ──────────────────────────────────────────────
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Connection lost! Reconnecting...");
    connectWiFi();
    return;
  }

  // ── MQTT watchdog ──────────────────────────────────────────────
  if (!client.connected()) reconnect();
  client.loop();

  // ── Apply pending mode change ──────────────────────────────────
  if (action_id != lastMode) {
    applyMode(action_id);
    lastMode = action_id;
  }

  // ── Sensor publish at fixed interval ──────────────────────────
  if (millis() - lastSensorPublish < SENSOR_INTERVAL) return;
  lastSensorPublish = millis();

  // ── Raw sensor reads ───────────────────────────────────────────
  float rawGas = (float)ads.readADC_SingleEnded(0);
  float rawLDR = (float)ads.readADC_SingleEnded(1);
  float temp   = dht.readTemperature();
  float hum    = dht.readHumidity();
  int   motion = digitalRead(PIR_PIN);

  // ── DHT sanity check ───────────────────────────────────────────
  if (isnan(temp) || isnan(hum)) {
    Serial.println("[DHT] Read failed — skipping cycle.");
    return;
  }

  // ── EMA initialisation / update ───────────────────────────────
  if (firstRead) {
    smoothGas = rawGas;
    smoothLDR = rawLDR;
    firstRead = false;
  } else {
    smoothGas = applyEMA(rawGas, smoothGas);
    smoothLDR = applyEMA(rawLDR, smoothLDR);
  }

  // ════════════════════════════════════════
  //  Gas → AQI   (matches Code 2 formula)
  //  Normal direction: higher ADC = more gas
  // ════════════════════════════════════════
  if (smoothGas < gasMin) gasMin = smoothGas;
  if (smoothGas > gasMax) gasMax = smoothGas;

  float gasNorm = (smoothGas - gasMin) / (gasMax - gasMin + 1.0f);
  gasNorm = constrain(gasNorm, 0.0f, 1.0f);
  float aqi = DATA_AQI_MIN + (gasNorm * (DATA_AQI_MAX - DATA_AQI_MIN));

  // ════════════════════════════════════════
  //  LDR → Lux   (matches Code 2 formula)
  // ════════════════════════════════════════
  float invertedLDR = 4095.0f - smoothLDR;
  if (invertedLDR < lightMin) lightMin = invertedLDR;
  if (invertedLDR > lightMax) lightMax = invertedLDR;

  float luxNorm = (invertedLDR - lightMin) / (lightMax - lightMin + 1.0f);
  float lux     = luxNorm * DATA_LUX_MAX;

  // ── Constrain to valid dataset ranges ─────────────────────────
  temp = constrain(temp, 22.1f, 40.4f);
  hum  = constrain(hum,  38.1f, 65.0f);
  aqi  = constrain(aqi,  57.0f, 185.4f);

  // ── Build timestamp string ─────────────────────────────────────
  char buf[25];
  sprintf(buf, "%04d-%02d-%02d %02d:%02d:%02d",
          ts_year, ts_month, ts_day, ts_hour, ts_min, ts_sec);

  // ── CSV output on Serial ───────────────────────────────────────
  Serial.print(buf);     Serial.print(",");
  Serial.print(temp, 2); Serial.print(",");
  Serial.print(hum,  2); Serial.print(",");
  Serial.print(aqi,  2); Serial.print(",");
  Serial.print(lux,  2); Serial.print(",");
  Serial.println(motion);

  // ── Build & publish MQTT JSON payload ─────────────────────────
  String payload = "{";
  payload += "\"Timestamp\":\""     + String(buf)     + "\",";
  payload += "\"Temp_C\":"          + String(temp, 2) + ",";
  payload += "\"Humidity_pct\":"    + String(hum,  2) + ",";
  payload += "\"Gas_AQI\":"         + String(aqi,  2) + ",";
  payload += "\"Light_Lux\":"       + String(lux,  2) + ",";
  payload += "\"Motion_Detected\":" + String(motion);
  payload += "}";

  if (client.connected()) {
    client.publish(sensor_topic, payload.c_str());
    Serial.println("[MQTT] SENT: " + payload);
  } else {
    Serial.println("[MQTT] Not connected — payload not sent.");
  }

  // ── Advance timestamp ──────────────────────────────────────────
  incrementTimestamp();
}
