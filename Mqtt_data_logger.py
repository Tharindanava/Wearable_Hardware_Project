import sys
import json
import time
import pandas as pd
from paho.mqtt import client as mqtt_client
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QWidget, QLabel, QComboBox, QFileDialog,
                            QTextEdit, QGroupBox, QSpinBox, QMessageBox)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject, QDateTime
from collections import defaultdict
from typing import Dict, List, Set, Optional, Any

class MQTTCommunicator(QObject):
    message_received = pyqtSignal(str, dict)
    connection_status = pyqtSignal(str)
    
    def __init__(self, broker: str, port: int, topic_pattern: str):
        super().__init__()
        self.broker = broker
        self.port = port
        self.topic_pattern = topic_pattern
        self.client = mqtt_client.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        
    def start(self) -> None:
        """Start the MQTT client connection"""
        try:
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            self.connection_status.emit(f"Connection failed: {str(e)}")
            
    def stop(self) -> None:
        """Stop the MQTT client connection"""
        self.client.loop_stop()
        self.client.disconnect()
        
    def on_connect(self, client: mqtt_client.Client, userdata: Any, flags: Dict, rc: int) -> None:
        """Callback for when the client connects to the broker"""
        if rc == 0:
            self.connection_status.emit("Connected to MQTT Broker")
            self.client.subscribe(self.topic_pattern)
        else:
            self.connection_status.emit(f"Failed to connect, return code {rc}")
            
    def on_disconnect(self, client: mqtt_client.Client, userdata: Any, rc: int) -> None:
        """Callback for when the client disconnects from the broker"""
        if rc != 0:
            self.connection_status.emit(f"Unexpected disconnection (rc={rc}), reconnecting...")
            self.start()
            
    def on_message(self, client: mqtt_client.Client, userdata: Any, msg: mqtt_client.MQTTMessage) -> None:
        """Callback for when a message is received"""
        try:
            sensor_id = msg.topic.split('/')[-1]
            payload = json.loads(msg.payload.decode())
            
            # Ensure timestamp is an integer
            payload['timestamp'] = int(payload.get('timestamp', time.time() * 1000))
                
            self.message_received.emit(sensor_id, payload)
        except json.JSONDecodeError:
            print(f"Failed to decode JSON payload from {msg.topic}")
        except Exception as e:
            print(f"Error processing message: {e}")

class MQTTDataLogger(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MPU6050 Data Logger")
        self.setGeometry(100, 100, 1000, 800)
        
        # MQTT Configuration
        self.broker = '192.168.1.2'
        self.port = 1883
        self.topic_pattern = "sensor/+"
        
        # Data storage
        self.recording = False
        self.recording_start_time = 0
        self.data: Dict[str, List[Dict]] = defaultdict(list)
        self.connected_sensors: Set[str] = set()
        self.current_sensor: Optional[str] = None
        self.sampling_rate = 50
        self.last_recorded_time: Dict[str, int] = {}
        self.last_values: Dict[str, Dict] = {}  # Store last values for each sensor
        
        # Initialize UI
        self.init_ui()
        
        # Setup MQTT communicator
        self.mqtt_communicator = MQTTCommunicator(self.broker, self.port, self.topic_pattern)
        self.mqtt_communicator.message_received.connect(self.handle_mqtt_message)
        self.mqtt_communicator.connection_status.connect(self.update_connection_status)
        self.mqtt_communicator.start()
        
        # Timer for updating the UI
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(100)  # Update UI every 100ms
        
    def init_ui(self) -> None:
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Status Group
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.connection_status = QLabel("Connecting to MQTT broker...")
        self.recording_status = QLabel("Recording: OFF")
        self.sensors_connected = QLabel("Connected Sensors: None")
        self.recording_duration = QLabel("Duration: 0.00s")
        self.data_points_label = QLabel("Data Points: 0")
        
        status_layout.addWidget(self.connection_status)
        status_layout.addWidget(self.recording_status)
        status_layout.addWidget(self.sensors_connected)
        status_layout.addWidget(self.recording_duration)
        status_layout.addWidget(self.data_points_label)
        status_group.setLayout(status_layout)
        
        # Control Group
        control_group = QGroupBox("Controls")
        control_layout = QHBoxLayout()
        
        # Sampling rate control
        sampling_layout = QHBoxLayout()
        sampling_layout.addWidget(QLabel("Sampling Rate (Hz):"))
        self.sampling_rate_spin = QSpinBox()
        self.sampling_rate_spin.setRange(1, 200)
        self.sampling_rate_spin.setValue(50)
        self.sampling_rate_spin.valueChanged.connect(self.set_sampling_rate)
        sampling_layout.addWidget(self.sampling_rate_spin)
        
        # Recording controls
        self.start_button = QPushButton("Start Recording (Space)")
        self.start_button.clicked.connect(self.toggle_recording)
        self.stop_button = QPushButton("Stop Recording (Space)")
        self.stop_button.clicked.connect(self.toggle_recording)
        self.stop_button.setEnabled(False)
        
        control_layout.addLayout(sampling_layout)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_group.setLayout(control_layout)
        
        # Sensor Selection Group
        sensor_group = QGroupBox("Sensor Monitoring")
        sensor_layout = QVBoxLayout()
        
        self.sensor_selector = QComboBox()
        self.sensor_selector.currentIndexChanged.connect(self.change_monitored_sensor)
        sensor_layout.addWidget(QLabel("Select Sensor to Monitor:"))
        sensor_layout.addWidget(self.sensor_selector)
        
        self.sensor_display = QTextEdit()
        self.sensor_display.setReadOnly(True)
        self.sensor_display.setFontFamily("Courier New")  # Monospace font for better alignment
        sensor_layout.addWidget(self.sensor_display)
        
        sensor_group.setLayout(sensor_layout)
        
        # Add groups to main layout
        main_layout.addWidget(status_group)
        main_layout.addWidget(control_group)
        main_layout.addWidget(sensor_group)
        
    def set_sampling_rate(self, rate: int) -> None:
        """Set the target sampling rate in Hz"""
        self.sampling_rate = rate
        
    def keyPressEvent(self, event) -> None:
        """Handle keyboard events"""
        if event.key() == Qt.Key_Space:
            self.toggle_recording()
        super().keyPressEvent(event)
            
    def update_connection_status(self, status: str) -> None:
        """Update the connection status display"""
        self.connection_status.setText(status)
            
    def handle_mqtt_message(self, sensor_id: str, payload: Dict) -> None:
        """Handle incoming MQTT messages"""
        self.connected_sensors.add(sensor_id)
        
        # Store the latest values for this sensor
        self.last_values[sensor_id] = payload
        
        if self.recording:
            timestamp = int(payload['timestamp'])
            
            # Initialize if first time seeing this sensor
            if sensor_id not in self.last_recorded_time:
                self.last_recorded_time[sensor_id] = timestamp
                self.data[sensor_id].append(payload)
                return
            
            # Calculate time since last recorded sample
            time_since_last = timestamp - self.last_recorded_time[sensor_id]
            target_interval = 1000 / self.sampling_rate  # ms between samples
            
            # If enough time has passed, record the current values
            if time_since_last >= target_interval:
                self.data[sensor_id].append(payload)
                self.last_recorded_time[sensor_id] = timestamp
            
        # Always update display with latest values
        if sensor_id == self.current_sensor:
            self.update_sensor_display(payload)
            
    def toggle_recording(self) -> None:
        """Toggle recording state"""
        if not self.recording:
            # Start recording
            self.start_recording()
        else:
            # Stop recording
            self.stop_recording()
            
    def start_recording(self) -> None:
        """Start a new recording session"""
        self.recording = True
        self.data = defaultdict(list)
        self.last_recorded_time = {}
        self.recording_start_time = int(time.time() * 1000)
        self.recording_status.setText("Recording: ON")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # Initialize with any values we already have
        for sensor_id, payload in self.last_values.items():
            self.data[sensor_id].append(payload)
            self.last_recorded_time[sensor_id] = int(payload['timestamp'])
            
    def stop_recording(self) -> None:
        """Stop the current recording session"""
        self.recording = False
        self.recording_status.setText("Recording: OFF")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        if not self.data:
            QMessageBox.information(self, "No Data", "No data was recorded during this session.")
            return
            
        self.save_data()
            
    def save_data(self) -> None:
        """Save the recorded data to a file"""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Save Data")
        if not folder:
            return
            
        try:
            combined_df = self.combine_sensor_data()
            
            if combined_df is not None and not combined_df.empty:
                timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
                filename = f"{folder}/mpu6050_data_{timestamp}.xlsx"
                combined_df.to_excel(filename, index=False)
                QMessageBox.information(self, "Success", f"Data successfully saved to:\n{filename}")
            else:
                QMessageBox.warning(self, "No Data", "No valid data to save.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save data: {str(e)}")
            
    def combine_sensor_data(self) -> Optional[pd.DataFrame]:
        """Combine data from all sensors into a single DataFrame"""
        if not self.data:
            return None
            
        # Create a list of all timestamps from all sensors
        all_timestamps = set()
        for sensor_data in self.data.values():
            for point in sensor_data:
                all_timestamps.add(point['timestamp'])
        all_timestamps = sorted(all_timestamps)
        
        # Create a DataFrame with all timestamps
        df = pd.DataFrame({'timestamp': all_timestamps})
        
        # Normalize timestamps to start from 0
        if len(all_timestamps) > 0:
            df['time_s'] = (df['timestamp'] - self.recording_start_time) / 1000.0
        
        # Add data from each sensor
        for sensor_id, sensor_data in self.data.items():
            if not sensor_data:
                continue
                
            # Create a temporary DataFrame for this sensor
            sensor_df = pd.DataFrame(sensor_data)
            
            # Rename columns to include sensor ID
            column_mapping = {
                'acc_X': f'ax_{sensor_id}',
                'acc_Y': f'ay_{sensor_id}',
                'acc_Z': f'az_{sensor_id}',
                'gyro_X': f'gx_{sensor_id}',
                'gyro_Y': f'gy_{sensor_id}',
                'gyro_Z': f'gz_{sensor_id}',
                'w': f'w_{sensor_id}',
                'i': f'i_{sensor_id}',
                'j': f'j_{sensor_id}',
                'k': f'k_{sensor_id}'
            }
            sensor_df = sensor_df.rename(columns=column_mapping)
            
            # Merge with main DataFrame
            df = pd.merge(df, sensor_df[['timestamp'] + list(column_mapping.values())], 
                         on='timestamp', how='left')
        
        # Sort by timestamp and reset index
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
                
    def change_monitored_sensor(self, index: int) -> None:
        """Change which sensor is being monitored in the display"""
        if 0 <= index < self.sensor_selector.count():
            self.current_sensor = self.sensor_selector.itemText(index)
            if self.current_sensor in self.last_values:
                self.update_sensor_display(self.last_values[self.current_sensor])
            else:
                self.sensor_display.setText(f"Waiting for data from sensor {self.current_sensor}...")
            
    def update_sensor_display(self, data: Dict) -> None:
        """Update the sensor display with new data"""
        display_text = f"Sensor {self.current_sensor} Data:\n\n"
        for key, value in data.items():
            display_text += f"{key:>10}: {value}\n"
        self.sensor_display.setPlainText(display_text)
        
    def update_ui(self) -> None:
        """Update the user interface"""
        # Update connected sensors list
        if self.connected_sensors:
            self.sensors_connected.setText(f"Connected Sensors: {', '.join(sorted(self.connected_sensors))}")
            
            current_items = {self.sensor_selector.itemText(i) for i in range(self.sensor_selector.count())}
            new_sensors = sorted(self.connected_sensors - current_items)
            
            for sensor in new_sensors:
                self.sensor_selector.addItem(sensor)
                
            if not self.current_sensor and self.sensor_selector.count() > 0:
                self.current_sensor = self.sensor_selector.itemText(0)
        else:
            self.sensors_connected.setText("Connected Sensors: None")
            
        # Update recording duration
        if self.recording:
            duration = (time.time() * 1000 - self.recording_start_time) / 1000.0
            self.recording_duration.setText(f"Duration: {duration:.2f}s")
            
            # Update data points count
            total_points = sum(len(data) for data in self.data.values())
            self.data_points_label.setText(f"Data Points: {total_points}")
            
    def closeEvent(self, event) -> None:
        """Handle window close event"""
        self.mqtt_communicator.stop()
        if self.recording:
            reply = QMessageBox.question(
                self, 'Recording in Progress',
                'A recording is in progress. Do you want to stop and save it before exiting?',
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Yes:
                self.stop_recording()
                event.accept()
            elif reply == QMessageBox.No:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MQTTDataLogger()
    window.show()
    sys.exit(app.exec_())