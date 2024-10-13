// Sender Code

#include <Arduino.h>
#include <Wire.h>
#include <MPU6050_6Axis_MotionApps20.h>
#include <esp_now.h>
#include <WiFi.h>

// Replace with the MAC address of your receiver (central ESP32 module)
uint8_t receiverAddress[] = {0xE0, 0x5A, 0x1B, 0xA1, 0x6D, 0xE0}; 
// Fill in your receiver's MAC address module_02 : 30:AE:A4:28:3A:D4 / MAC address module_01 :30:AE:A4:28:38:D8 / MAC Address Epa_esp: E0:5A:1B:A1:6D:E0

MPU6050 mpu;

#define INTERRUPT_PIN 14  // Use pin 14 on ESP32
bool dmpReady = false;
uint8_t devStatus;
uint16_t packetSize;
uint8_t fifoBuffer[64];

Quaternion q;
VectorInt16 aaReal;
VectorFloat gravity;
float euler[3];
float acc_X, acc_Y, acc_Z;

typedef struct __attribute__((packed)) struct_message {
  unsigned long timestamp;
  float acc_X;
  float acc_Y;
  float acc_Z;
  float w, i, j, k;
  float psi, theta, phi;
} struct_message;

struct_message sensorData;

void initMPU6050() {
  Wire.begin();
  mpu.initialize();
  devStatus = mpu.dmpInitialize();

  if (devStatus == 0) {
    mpu.CalibrateAccel(6);
    mpu.CalibrateGyro(6);
    mpu.setDMPEnabled(true);
    packetSize = mpu.dmpGetFIFOPacketSize();
    dmpReady = true;
  } else {
    Serial.println("MPU6050 initialization failed");
  }
}

void getSensorData() {
  if (!dmpReady) return;

  if (mpu.dmpGetCurrentFIFOPacket(fifoBuffer)) {
    mpu.dmpGetQuaternion(&q, fifoBuffer);
    mpu.dmpGetAccel(&aaReal, fifoBuffer);
    mpu.dmpGetGravity(&gravity, &q);
    mpu.dmpGetLinearAccel(&aaReal, &aaReal, &gravity);
    mpu.dmpGetEuler(euler, &q);

    acc_X = aaReal.x / 835.066; // Adjust the scaling if needed
    acc_Y = aaReal.y / 835.066;
    acc_Z = aaReal.z / 835.066;

    sensorData.timestamp = millis();
    sensorData.acc_X = acc_X;
    sensorData.acc_Y = acc_Y;
    sensorData.acc_Z = acc_Z;
    sensorData.w = q.w;
    sensorData.i = q.x;
    sensorData.j = q.y;
    sensorData.k = q.z;
    sensorData.psi = euler[0];
    sensorData.theta = euler[1];
    sensorData.phi = euler[2];
  }
}

void setup() {
  Serial.begin(9600);

  // Initialize MPU6050
  initMPU6050();

  // Initialize Wi-Fi and ESP-NOW
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  // Register send callback function
  esp_now_register_send_cb([](const uint8_t *mac_addr, esp_now_send_status_t status) {
    // Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Delivery Success" : "Delivery Fail");
  });

  // Add receiver as peer
  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, receiverAddress, 6);
  peerInfo.channel = 0;  
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK){
    Serial.println("Failed to add peer");
    return;
  }
}

void loop() {
  getSensorData();

  // Send data via ESP-NOW
  esp_err_t result = esp_now_send(receiverAddress, (uint8_t *)&sensorData, sizeof(sensorData));

  if (result == ESP_OK) {
    Serial.println("Sent with success");
  } else {
    Serial.println("Error sending data");
  }

  delay(90); // Adjust delay as needed
}
