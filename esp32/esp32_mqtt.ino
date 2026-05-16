#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <DHT.h>
#include <ESP32Servo.h>
#include <ArduinoJson.h>

// ================= WiFi / MQTT =================
const char* ssid        = "Mena";
const char* password    = "12345678";
const char* mqtt_server = "10.35.93.69";

const char* sensor_topic = "alg1/sensors";
const char* action_topic = "alg1/actions";

WiFiClient espClient;
PubSubClient client(espClient);

// ================= Pins =================
#define PIR_PIN    15
#define DHT_PIN    14
#define DHT_TYPE   DHT11
#define FAN_PIN    33
#define BUZZER_PIN 23
#define SERVO1_PIN 18
#define SERVO2_PIN 19
#define RED_PIN    25
#define GREEN_PIN  26
#define BLUE_PIN   27

// ================= Objects =================
DHT dht(DHT_PIN, DHT_TYPE);
Adafruit_ADS1115 ads;
Servo servo1;
Servo servo2;

// ================= State =================
int action_id = 0;

// ================= EMA Smoothing =================
float alpha = 0.15;
float smoothGas = 0;
float smoothLDR = 0;
bool firstRead = true;

// ================= Dataset Stats =================
const float DATA_AQI_MIN = 57.0;
const float DATA_AQI_MAX = 185.4;
const float DATA_LUX_MAX = 85759.0;

// ================= Calibration =================
float gasMin = 500;
float gasMax = 3000;
float lightMin = 0;
float lightMax = 4095;

// ================= Timestamp =================
int ts_year  = 2026;
int ts_month = 5;
int ts_day   = 2;
int ts_hour  = 0;
int ts_min   = 0;
int ts_sec   = 0;

// ================= Servo Non-Blocking =================
int servoTarget1 = 0;
int servoTarget2 = 180;
int servoCurrent1 = 0;
int servoCurrent2 = 180;
unsigned long lastServoMove = 0;

// ================= Sensor Timing =================
unsigned long lastSensorPublish = 0;
const unsigned long SENSOR_INTERVAL = 5000;

// =================================================
// Timestamp Increment
// =================================================
void incrementTimestamp() {
  ts_sec += 5;
  if (ts_sec >= 60)  { ts_sec = 0;  ts_min++;  }
  if (ts_min >= 60)  { ts_min = 0;  ts_hour++; }
  if (ts_hour >= 24) { ts_hour = 0; ts_day++;  }
}

// =================================================
// Servo Update - called every loop(), moves 1 step per 10ms
// =================================================
void updateServos() {
  if (millis() - lastServoMove >= 10) {
    lastServoMove = millis();
    bool moved = false;

    if (servoCurrent1 < servoTarget1) { servoCurrent1++; moved = true; }
    else if (servoCurrent1 > servoTarget1) { servoCurrent1--; moved = true; }

    if (servoCurrent2 < servoTarget2) { servoCurrent2++; moved = true; }
    else if (servoCurrent2 > servoTarget2) { servoCurrent2--; moved = true; }

    if (moved) {
      servo1.write(servoCurrent1);
      servo2.write(servoCurrent2);
    }
  }
}

void setRGB(bool r, bool g, bool b) {
  digitalWrite(RED_PIN, r);
  digitalWrite(GREEN_PIN, g);
  digitalWrite(BLUE_PIN, b);
}

// =================================================
// Apply Mode
// 0 normal, 1 ventilation, 2 chemical, 3 security breach
// =================================================
void applyMode(int mode) {
  action_id = constrain(mode, 0, 3);

  switch (action_id) {
    case 0:
      digitalWrite(FAN_PIN, LOW);
      digitalWrite(BUZZER_PIN, LOW);
      setRGB(0, 1, 0);
      servoTarget1 = 0;
      servoTarget2 = 180;
      break;

    case 1:
      digitalWrite(FAN_PIN, HIGH);
      digitalWrite(BUZZER_PIN, LOW);
      setRGB(1, 1, 0);
      servoTarget1 = 90;
      servoTarget2 = 90;
      break;

    case 2:
      digitalWrite(FAN_PIN, HIGH);
      digitalWrite(BUZZER_PIN, HIGH);
      setRGB(1, 0, 1);
      servoTarget1 = 180;
      servoTarget2 = 0;
      break;

    case 3:
      digitalWrite(FAN_PIN, LOW);
      digitalWrite(BUZZER_PIN, HIGH);
      setRGB(1, 0, 0);
      servoTarget1 = 0;
      servoTarget2 = 180;
      break;
  }
}

float applyEMA(float current, float prev) {
  return (alpha * current) + ((1.0 - alpha) * prev);
}

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
    Serial.println();
    Serial.println("WiFi connected");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.print("RSSI: ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
  } else {
    Serial.println();
    Serial.println("WiFi connection failed");
  }
}

void reconnect() {
  while (!client.connected()) {
    Serial.println("Trying to connect to MQTT broker...");
    String clientId = "ESP32Client-" + String((uint32_t)ESP.getEfuseMac(), HEX);
    if (client.connect(clientId.c_str())) {
      Serial.println("Connected to broker");
      client.subscribe(action_topic);
    } else {
      Serial.print("Failed rc=");
      Serial.print(client.state());
      Serial.println(". Retrying in 5s...");
      delay(5000);
    }
  }
}

int modeFromActionJson(JsonDocument& doc) {
  if (doc["action_id"].is<int>()) {
    return doc["action_id"].as<int>();
  }

  const char* fan = doc["fan"] | "OFF";
  const char* alarm = doc["alarm"] | "OFF";
  const char* servo = doc["servo"] | "CLOSED";
  const char* buzzer = doc["buzzer"] | "OFF";
  const char* rgb = doc["rgb_led"] | "GREEN";

  bool fanOn = strcmp(fan, "ON") == 0;
  bool alarmOn = strcmp(alarm, "ON") == 0;
  bool servoOpen = strcmp(servo, "OPEN") == 0;
  bool buzzerOn = strcmp(buzzer, "ON") == 0;
  bool red = strcmp(rgb, "RED") == 0;
  bool yellow = strcmp(rgb, "YELLOW") == 0;

  if (buzzerOn && !fanOn) return 3;
  if (alarmOn || buzzerOn || red) return 2;
  if (fanOn || servoOpen || yellow) return 1;
  return 0;
}

void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message received on topic: ");
  Serial.println(topic);

  StaticJsonDocument<256> doc;
  DeserializationError err = deserializeJson(doc, payload, length);

  int nextMode = action_id;
  if (!err) {
    nextMode = modeFromActionJson(doc);
  } else {
    String message = "";
    for (unsigned int i = 0; i < length; i++) {
      message += (char)payload[i];
    }
    nextMode = message.toInt();
  }

  applyMode(nextMode);
  Serial.print("Applied action_id: ");
  Serial.println(action_id);
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
  client.setBufferSize(512);

  dht.begin();
  ads.begin();
  analogReadResolution(12);

  pinMode(PIR_PIN, INPUT);
  pinMode(FAN_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(RED_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);
  pinMode(BLUE_PIN, OUTPUT);

  servo1.attach(SERVO1_PIN);
  servo2.attach(SERVO2_PIN);
  servo1.write(servoCurrent1);
  servo2.write(servoCurrent2);

  applyMode(0);
  connectWiFi();

  Serial.println("Timestamp,Temp_C,Humidity_pct,Gas_AQI,Light_Lux,Motion_Detected");
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi lost. Reconnecting...");
    connectWiFi();
  }

  if (!client.connected()) reconnect();
  client.loop();
  updateServos();

  if (millis() - lastSensorPublish >= SENSOR_INTERVAL) {
    lastSensorPublish = millis();

    float rawGas = ads.readADC_SingleEnded(0);
    float rawLDR = ads.readADC_SingleEnded(1);
    float temp = dht.readTemperature();
    float hum = dht.readHumidity();
    int motion = digitalRead(PIR_PIN);

    if (isnan(temp) || isnan(hum)) {
      Serial.println("DHT read failed. Skipping this cycle.");
      return;
    }

    if (firstRead) {
      smoothGas = rawGas;
      smoothLDR = rawLDR;
      firstRead = false;
    } else {
      smoothGas = applyEMA(rawGas, smoothGas);
      smoothLDR = applyEMA(rawLDR, smoothLDR);
    }

    if (smoothGas < gasMin) gasMin = smoothGas;
    if (smoothGas > gasMax) gasMax = smoothGas;

    float gasNorm = (smoothGas - gasMin) / (gasMax - gasMin + 1.0);
    float aqi = DATA_AQI_MIN + (gasNorm * (DATA_AQI_MAX - DATA_AQI_MIN));

    float invertedLDR = 4095.0 - smoothLDR;
    if (invertedLDR < lightMin) lightMin = invertedLDR;
    if (invertedLDR > lightMax) lightMax = invertedLDR;
    float luxNorm = (invertedLDR - lightMin) / (lightMax - lightMin + 1.0);
    float lux = luxNorm * DATA_LUX_MAX;

    temp = constrain(temp, 22.1, 40.4);
    hum = constrain(hum, 38.1, 65.0);
    aqi = constrain(aqi, 57.0, 185.4);

    char timestamp[25];
    sprintf(timestamp, "%04d-%02d-%02d %02d:%02d:%02d",
            ts_year, ts_month, ts_day, ts_hour, ts_min, ts_sec);

    Serial.print(timestamp); Serial.print(",");
    Serial.print(temp, 2); Serial.print(",");
    Serial.print(hum, 2); Serial.print(",");
    Serial.print(aqi, 2); Serial.print(",");
    Serial.print(lux, 2); Serial.print(",");
    Serial.println(motion ? 1 : 0);

    StaticJsonDocument<256> doc;
    doc["Timestamp"] = timestamp;
    doc["Temp_C"] = temp;
    doc["Humidity_pct"] = hum;
    doc["Gas_AQI"] = aqi;
    doc["Light_Lux"] = lux;
    doc["Motion_Detected"] = motion ? 1 : 0;

    char payload[256];
    size_t n = serializeJson(doc, payload);
    client.publish(sensor_topic, payload, n);

    Serial.print("MQTT SENT: ");
    Serial.println(payload);

    incrementTimestamp();
  }
}
