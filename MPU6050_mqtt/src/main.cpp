#include <Arduino.h>
#include <Wire.h>
#include <MPU6050_6Axis_MotionApps20.h>
#include <WiFi.h>
#include <PubSubClient.h>

// Replace with your network credentials and MQTT broker details
const char* ssid = "ZTE";//"Eng-Student";//"Tharinda_Lap";//"TIEC-2.4G";//"SLT-Fiber-37DB";
const char* password = "Agrovoltic";//"3nG5tuDt";//"Rashmi1213";//"nava@123";//"TIEC2954";
const char* mqtt_server = "192.168.1.102";//"192.168.1.105";//"192.168.1.2";//"192.168.1.10";//"ec2-54-219-34-11.us-west-1.compute.amazonaws.com";
const char* mqtt_topic = "sensor/mpu6050_in";
const char* client_id = "ESP32_MPU6050_03";

WiFiClient espClient;
PubSubClient client(espClient);

MPU6050 mpu;

#define INTERRUPT_PIN 14  // use pin 14 on ESP32
bool dmpReady = false;
uint8_t devStatus;
uint16_t packetSize;
uint8_t fifoBuffer[64];

Quaternion q;
VectorInt16 aa;
VectorInt16 aaReal;
VectorInt16 aaWorld;
VectorFloat gravity;
float euler[3];
float ypr[3];
float acc_X, acc_Y, acc_Z;

volatile bool mpuInterrupt = false;
void dmpDataReady() {
  mpuInterrupt = true;
}

SemaphoreHandle_t dataSemaphore;
String jsonPayload;

// MQTT reconnect function
void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    client.setServer(mqtt_server, 1883);
    if (client.connect(client_id)) {
      Serial.println("connected");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      delay(5000);
    }
  }
}

void getSensorDataTask(void* parameter) {
  while (true) {
    if (!dmpReady) continue;

    if (mpu.dmpGetCurrentFIFOPacket(fifoBuffer)) {
      mpu.dmpGetQuaternion(&q, fifoBuffer);
      mpu.dmpGetAccel(&aa, fifoBuffer);
      mpu.dmpGetGravity(&gravity, &q);
      mpu.dmpGetLinearAccel(&aaReal, &aa, &gravity);
      mpu.dmpConvertToWorldFrame(&aaWorld, &aaReal, &q);
      mpu.dmpGetEuler(euler, &q);

      acc_X = aaReal.x / 835.066;
      acc_Y = aaReal.y / 835.066;
      acc_Z = aaReal.z / 835.066;

      unsigned long timestamp = millis(); // Get timestamp in milliseconds

      xSemaphoreTake(dataSemaphore, portMAX_DELAY);
      
      xSemaphoreGive(dataSemaphore);
    }

    delay(5);  // Minimize task blocking
  }
}

void publishSensorDataTask(void* parameter) {
  while (true) {
    if (!client.connected()) {
      reconnect();
    }
    client.loop();

    xSemaphoreTake(dataSemaphore, portMAX_DELAY);
    unsigned long timestamp = millis(); // Get timestamp in milliseconds

    jsonPayload = "{";
    jsonPayload += "\"timestamp\":\"" + String(timestamp) + "\",";
    jsonPayload += "\"acc_X\":\"" + String(acc_X) + "\",";
    jsonPayload += "\"acc_Y\":\"" + String(acc_Y) + "\",";
    jsonPayload += "\"acc_Z\":\"" + String(acc_Z) + "\",";
    jsonPayload += "\"w\":\"" + String(q.w) + "\",";
    jsonPayload += "\"i\":\"" + String(q.x) + "\",";
    jsonPayload += "\"j\":\"" + String(q.y) + "\",";
    jsonPayload += "\"k\":\"" + String(q.z) + "\",";
    jsonPayload += "\"psi\":\"" + String(euler[0]) + "\",";
    jsonPayload += "\"theta\":\"" + String(euler[1]) + "\",";
    jsonPayload += "\"phi\":\"" + String(euler[2]) + "\"";
    jsonPayload += "}";
    client.publish(mqtt_topic, jsonPayload.c_str());
    xSemaphoreGive(dataSemaphore);

    delay(15);  // Adjust as needed for data frequency
  }
}

void setup() {
  Serial.begin(9600);

  // Setup WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");

  // Setup MQTT
  client.setServer(mqtt_server, 1883);

  // Initialize MPU6050
  Wire.begin();
  mpu.initialize();
  pinMode(INTERRUPT_PIN, INPUT);

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

  // Create semaphore
  dataSemaphore = xSemaphoreCreateMutex();

  // Create tasks for each core
  xTaskCreatePinnedToCore(getSensorDataTask, "GetSensorDataTask", 4096, NULL, 1, NULL, 0);
  xTaskCreatePinnedToCore(publishSensorDataTask, "PublishSensorDataTask", 4096, NULL, 1, NULL, 1);
}

void loop() {
  // Loop is empty since tasks handle the workload
}
