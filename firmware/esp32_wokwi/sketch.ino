// ESP32 Smart Factory Digital Twin Sensor Firmware (Wokwi Compatible)
// Hardware:
//  - DHT22 on GPIO 15 (Temperature Sensor)
//  - MPU6050 on I2C (SDA=GPIO 21, SCL=GPIO 22) (Vibration / Accelerometer)
//  - Potentiometer on GPIO 34 (Analog Current/Power proxy)

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <DHT.h>
#include <ArduinoJson.h>

#define DHTPIN 15
#define DHTTYPE DHT22
#define POT_PIN 34

// Configuration
const char* WIFI_SSID = "Wokwi-GUEST";
const char* WIFI_PASS = "";
// Set to public tunnel URL (ngrok) when using Wokwi in browser,
// or your local host IP if running physical ESP32 on same LAN:
const char* API_URL = "http://192.168.1.100:5000/sensor-data";
const char* MACHINE_ID = "M1";

DHT dht(DHTPIN, DHTTYPE);
Adafruit_MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n[ESP32] Initializing Smart Factory Sensor Node...");

  // Init DHT22
  dht.begin();

  // Init MPU6050
  Wire.begin(21, 22);
  if (!mpu.begin()) {
    Serial.println("[ESP32 Warning] Could not find MPU6050 chip!");
  } else {
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
    Serial.println("[ESP32] MPU6050 Initialized.");
  }

  // Connect WiFi
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("[ESP32] Connecting to WiFi");
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[ESP32] WiFi Connected! IP: " + WiFi.localIP().toString());
  } else {
    Serial.println("\n[ESP32] WiFi not connected. Serial streaming continues.");
  }
}

void loop() {
  // 1. Read Temperature
  float temperature = dht.readTemperature();
  if (isnan(temperature)) {
    temperature = 60.0; // fallback
  }

  // 2. Read Vibration magnitude from Accelerometer
  float vibration = 0.35;
  sensors_event_t a, g, temp;
  if (mpu.getEvent(&a, &g, &temp)) {
    // Magnitude in g (divide m/s^2 by 9.81)
    float ax = a.acceleration.x / 9.80665;
    float ay = a.acceleration.y / 9.80665;
    float az = (a.acceleration.z - 9.80665) / 9.80665; // Remove gravity offset
    vibration = sqrt(ax * ax + ay * ay + az * az);
  }

  // 3. Read Potentiometer as Power proxy (scaled to 0.0 - 6.0 kW)
  int potRaw = analogRead(POT_PIN);
  float power = (potRaw / 4095.0) * 6.0;

  // Build JSON payload
  StaticJsonDocument<256> doc;
  doc["machine_id"] = MACHINE_ID;
  doc["temperature"] = serialized(String(temperature, 2));
  doc["vibration"] = serialized(String(vibration, 3));
  doc["power"] = serialized(String(power, 2));
  doc["source"] = "esp32-sim";

  String jsonString;
  serializeJson(doc, jsonString);

  // Print JSON line to Serial for USB Serial Bridge
  Serial.println(jsonString);

  // HTTP POST if WiFi connected
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(API_URL);
    http.addHeader("Content-Type", "application/json");
    int httpResponseCode = http.POST(jsonString);
    if (httpResponseCode > 0) {
      Serial.print(" [HTTP POST OK: ");
      Serial.print(httpResponseCode);
      Serial.println("]");
    } else {
      Serial.print(" [HTTP POST ERR: ");
      Serial.print(httpResponseCode);
      Serial.println("]");
    }
    http.end();
  }

  delay(1000);
}
