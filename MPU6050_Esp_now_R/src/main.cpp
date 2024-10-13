// Receiver Code

#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>
#include <ArduinoJson.h>

typedef struct __attribute__((packed)) struct_message {
  unsigned long timestamp;
  float acc_X;
  float acc_Y;
  float acc_Z;
  float w, i, j, k;
  float psi, theta, phi;
} struct_message;

struct_message incomingData;

// Function to convert MAC address to String
String macToString(const uint8_t *mac) {
  char macStr[18];
  snprintf(macStr, sizeof(macStr), "%02X:%02X:%02X:%02X:%02X:%02X",
           mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
  return String(macStr);
}

void OnDataRecv(const uint8_t * mac, const uint8_t *incomingDataBytes, int len) {
  memcpy(&incomingData, incomingDataBytes, sizeof(incomingData));

  String jsonPayload ="{";
  jsonPayload += "\"sender\":\"" + macToString(mac) + "\",";
  jsonPayload += "\"timestamp\":\"" + String(incomingData.timestamp) + "\",";
  jsonPayload += "\"acc_X\":\""+ String(incomingData.acc_X) + "\",";
  jsonPayload += "\"acc_Y\":\""+ String(incomingData.acc_Y) + "\",";
  jsonPayload += "\"acc_Z\":\""+ String(incomingData.acc_Z) + "\",";
  jsonPayload += "\"w\":\""+ String(incomingData.w) + "\",";
  jsonPayload += "\"i\":\""+ String(incomingData.i) + "\",";
  jsonPayload += "\"j\":\""+ String(incomingData.j) + "\",";
  jsonPayload += "\"k\":\""+ String(incomingData.k) + "\",";
  jsonPayload += "\"psi\":\""+ String(incomingData.psi) + "\",";
  jsonPayload += "\"theta\":\""+ String(incomingData.theta) + "\",";
  jsonPayload += "\"phi\":\""+ String(incomingData.phi) + "\"";
  jsonPayload += "}";

  Serial.println(jsonPayload);
}

void setup() {
  Serial.begin(9600);

  // Initialize Wi-Fi and ESP-NOW
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  // Register receive callback function
  esp_now_register_recv_cb(OnDataRecv);

  // Print MAC address of this device
  Serial.print("Receiver MAC Address: ");
  Serial.println(WiFi.macAddress());
}

void loop() {
  // Nothing needed here, data is received via callback
}
