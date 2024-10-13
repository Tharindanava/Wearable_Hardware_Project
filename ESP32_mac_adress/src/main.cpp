#include <Arduino.h>
#include "esp_mac.h"  // required - exposes esp_mac_type_t values

// Function declarations (forward declarations)
String getDefaultMacAddress();
String getInterfaceMacAddress(esp_mac_type_t interface);

void setup() {
  Serial.begin(9600);
  while (!Serial) {
    delay(100);
  }

  Serial.println("Interface\t\t\t\t\t\tMAC address (6 bytes, 4 universally administered, default)");

  Serial.print("Wi-Fi Station (using 'esp_efuse_mac_get_default')\t");
  Serial.println(getDefaultMacAddress());

  Serial.print("WiFi Station (using 'esp_read_mac')\t\t\t");
  Serial.println(getInterfaceMacAddress(ESP_MAC_WIFI_STA));

  Serial.print("WiFi Soft-AP (using 'esp_read_mac')\t\t\t");
  Serial.println(getInterfaceMacAddress(ESP_MAC_WIFI_SOFTAP));

  Serial.print("Bluetooth (using 'esp_read_mac')\t\t\t");
  Serial.println(getInterfaceMacAddress(ESP_MAC_BT));

  Serial.print("Ethernet (using 'esp_read_mac')\t\t\t\t");
  Serial.println(getInterfaceMacAddress(ESP_MAC_ETH));
}

String getDefaultMacAddress() {
  String mac = "";
  unsigned char mac_base[6] = {0};
  if (esp_efuse_mac_get_default(mac_base) == ESP_OK) {
    char buffer[18];
    sprintf(buffer, "%02X:%02X:%02X:%02X:%02X:%02X", mac_base[0], mac_base[1], mac_base[2], mac_base[3], mac_base[4], mac_base[5]);
    mac = buffer;
  }
  return mac;
}

String getInterfaceMacAddress(esp_mac_type_t interface) {
  String mac = "";
  unsigned char mac_base[6] = {0};
  if (esp_read_mac(mac_base, interface) == ESP_OK) {
    char buffer[18];
    sprintf(buffer, "%02X:%02X:%02X:%02X:%02X:%02X", mac_base[0], mac_base[1], mac_base[2], mac_base[3], mac_base[4], mac_base[5]);
    mac = buffer;
  }
  return mac;
}

void loop() {
  // Nothing in the loop
}
