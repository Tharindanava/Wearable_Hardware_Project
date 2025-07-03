import sys
import time
import pandas as pd
from paho.mqtt import client as mqtt_client
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QWidget, QLabel, QComboBox, QFileDialog,
                            QTextEdit, QGroupBox, QMessageBox)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject, QDateTime, QThread
from collections import defaultdict, deque
from typing import Dict, List, Set, Optional, Any
import threading
import numpy as np

class MQTTWorker(QObject):
    message_received = pyqtSignal(str, dict)  # device_id, data
    device_discovered = pyqtSignal(str)
    device_lost = pyqtSignal(str)
    connection_status = pyqtSignal(str)
    
    def __init__(self, broker: str, port: int, sampling_rate: int):
        super().__init__()
        self.broker = broker
        self.port = port
        self.sampling_rate = sampling_rate
        self.active_devices = set()
        self.device_last_seen = {}  # Stores last timestamp from sensor data (in ms)
        self.device_timeout = 5  # seconds
        self.ping_sequence = deque()
        self.client = None
        self.last_values = {}

    def start(self):
        try:
            self.client = mqtt_client.Client()
            self.client.on_connect = self.on_connect
            self.client.on_message = self.on_message
            self.client.connect(self.broker, self.port)
            self.client.loop_start()
            self.connection_status.emit("Connected to MQTT Broker")
        except Exception as e:
            self.connection_status.emit(f"Connection failed: {str(e)}")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.subscribe("sensor/announce/#")
            client.subscribe("sensor/data/#")
            self.connection_status.emit("Connected to MQTT Broker")
        else:
            self.connection_status.emit(f"Connection failed with code {rc}")

    def on_message(self, client, userdata, msg):
        try:
            topic_parts = msg.topic.split('/')
            
            # Handle device announcements (only for registration)
            if len(topic_parts) == 3 and topic_parts[0] == "sensor" and topic_parts[1] == "announce":
                device_id = topic_parts[2]
                alive_status = msg.payload.decode()
                if alive_status in ["alive", "connected"]:
                    self.register_device(device_id)
                return
                
            # Handle sensor data (for both data and keepalive)
            if len(topic_parts) == 3 and topic_parts[0] == "sensor" and topic_parts[1] == "data":
                device_id = topic_parts[2]
                payload = msg.payload.decode()
                
                try:
                    # Parse the comma-separated values from the device
                    parts = payload.split(',')
                    if len(parts) != 11:
                        raise ValueError("Expected 11 values")
                        
                    data = {
                        'timestamp': int(parts[0]),  # Assuming timestamp is in milliseconds
                        'acc_X': int(parts[1]),
                        'acc_Y': int(parts[2]),
                        'acc_Z': int(parts[3]),
                        'w': float(parts[4]),
                        'x': float(parts[5]),
                        'y': float(parts[6]),
                        'z': float(parts[7]),
                        'gyro_X': int(parts[8]),
                        'gyro_Y': int(parts[9]),
                        'gyro_Z': int(parts[10]),
                        'device_id': device_id
                    }
                    
                    # Register device if not already registered
                    if device_id not in self.active_devices:
                        self.register_device(device_id)
                    
                    # Update last seen timestamp from sensor data
                    self.device_last_seen[device_id] = data['timestamp']
                    
                    self.last_values[device_id] = data
                    self.message_received.emit(device_id, data)
                    
                except (ValueError, IndexError) as e:
                    print(f"Error parsing data from {device_id}: {e}")
                    print(f"Raw payload: {payload}")
                    
        except Exception as e:
            print(f"Error processing message: {e}")

    def register_device(self, device_id: str):
        """Register a new device or update an existing one"""
        if device_id not in self.active_devices:
            self.active_devices.add(device_id)
            self.ping_sequence.append(device_id)
            self.device_discovered.emit(device_id)
            print(f"New device discovered: {device_id}")
            
        # Note: We don't update last_seen here - that's done when we receive sensor data

    def update_ping_timer(self):
        if not self.active_devices:
            return
            
        # Calculate desired ping interval (ms between pings to same device)
        total_cycle_time = 1000 / self.sampling_rate  # Total time for one complete cycle (ms)
        ping_interval = total_cycle_time / len(self.active_devices)
        
        # Ensure we don't go too fast (minimum 1ms between pings)
        ping_interval = max(ping_interval, 1)
        
        # This will be handled by the main thread's timer
        return ping_interval

    def send_next_ping(self):
        if self.ping_sequence and self.client:
            device = self.ping_sequence.popleft()
            topic = f"sensor/ping/{device}"
            self.client.publish(topic, "1")  # Send simple "1" as ping
            self.ping_sequence.append(device)
            print(f"Sent ping to {device}")  # Debug output

    def check_device_timeouts(self):
        """Check if any devices have stopped sending data by comparing timestamps"""
        timeout_ms = self.device_timeout * 1000  # Convert timeout to milliseconds
        
        # We'll need to track initial timestamps for comparison
        if not hasattr(self, 'device_initial_timestamps'):
            self.device_initial_timestamps = {}
        
        current_time = time.time() * 1000  # Current time in milliseconds
        
        for device_id, last_timestamp in list(self.device_last_seen.items()):
            # If we don't have an initial timestamp for this device yet, set it
            if device_id not in self.device_initial_timestamps:
                self.device_initial_timestamps[device_id] = (last_timestamp, current_time)
                continue
                
            initial_timestamp, check_time = self.device_initial_timestamps[device_id]
            
            # Check if timeout period has elapsed since we set the initial timestamp
            if current_time - check_time >= timeout_ms:
                # Compare the initial timestamp with the current last_timestamp
                if last_timestamp == initial_timestamp:
                    # Timestamp hasn't changed - remove the device
                    self.remove_device(device_id)
                else:
                    # Timestamp has changed - update the reference for next check
                    self.device_initial_timestamps[device_id] = (last_timestamp, current_time)

    def remove_device(self, device_id: str):
        if device_id in self.active_devices:
            self.active_devices.remove(device_id)
            self.ping_sequence = deque(d for d in self.ping_sequence if d != device_id)
            if device_id in self.device_last_seen:
                del self.device_last_seen[device_id]
            if hasattr(self, 'device_initial_timestamps') and device_id in self.device_initial_timestamps:
                del self.device_initial_timestamps[device_id]
            self.device_lost.emit(device_id)
            print(f"Device lost: {device_id}")

    def parse_sensor_data(self, payload: str) -> Dict[str, Any]:
        try:
            parts = payload.split(',')
            if len(parts) != 11:
                raise ValueError(f"Expected 11 values, got {len(parts)}")
                
            return {
                'timestamp': int(parts[0]),
                'acc_X': int(parts[1]),
                'acc_Y': int(parts[2]),
                'acc_Z': int(parts[3]),
                'w': float(parts[4]),
                'x': float(parts[5]),
                'y': float(parts[6]),
                'z': float(parts[7]),
                'gyro_X': int(parts[8]),
                'gyro_Y': int(parts[9]),
                'gyro_Z': int(parts[10])
            }
        except ValueError as e:
            raise ValueError(f"Invalid data format: {str(e)}. Payload: {payload}")

class MQTTDataLogger(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MPU6050 Data Logger")
        self.setGeometry(100, 100, 1000, 800)
        
        self.broker = '192.168.1.2'
        self.port = 1883
        self.sampling_rate = 64
        self.recording = False
        self.recording_start_time = 0
        self.recording_stop_time = 0
        self.data = defaultdict(list)
        self.data_lock = threading.Lock()
        self.actual_data_points = 0
        
        self.init_ui()
        self.setup_mqtt()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Status Group
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.connection_status_label = QLabel("Connecting to MQTT broker...")
        self.recording_status_label = QLabel("Recording: OFF")
        self.sensors_connected_label = QLabel("Connected Sensors: 0")
        self.recording_duration_label = QLabel("Duration: 0.00s")
        self.data_points_label = QLabel("Data Points: 0")
        
        status_layout.addWidget(self.connection_status_label)
        status_layout.addWidget(self.recording_status_label)
        status_layout.addWidget(self.sensors_connected_label)
        status_layout.addWidget(self.recording_duration_label)
        status_layout.addWidget(self.data_points_label)
        status_group.setLayout(status_layout)
        
        # Control Group
        control_group = QGroupBox("Controls")
        control_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Recording (Space)")
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button = QPushButton("Stop Recording (Space)")
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_group.setLayout(control_layout)
        
        # Sensor Selection Group
        sensor_group = QGroupBox("Sensor Monitoring")
        sensor_layout = QVBoxLayout()
        
        self.device_list = QComboBox()
        self.device_list.currentIndexChanged.connect(self.device_selected)
        sensor_layout.addWidget(QLabel("Select Sensor to Monitor:"))
        sensor_layout.addWidget(self.device_list)
        
        self.sensor_display = QTextEdit()
        self.sensor_display.setReadOnly(True)
        self.sensor_display.setFontFamily("Courier New")
        sensor_layout.addWidget(self.sensor_display)
        
        sensor_group.setLayout(sensor_layout)
        
        main_layout.addWidget(status_group)
        main_layout.addWidget(control_group)
        main_layout.addWidget(sensor_group)
        
        # Setup timers in main thread
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(100)  # 10 Hz UI update
        
        self.monitor_timer = QTimer(self)
        self.monitor_timer.timeout.connect(self.check_device_timeouts)
        self.monitor_timer.start(1000)  # Check every second

    def setup_mqtt(self):
        self.mqtt_thread = QThread()
        self.mqtt_worker = MQTTWorker(self.broker, self.port, self.sampling_rate)
        self.mqtt_worker.moveToThread(self.mqtt_thread)
        
        # Connect signals
        self.mqtt_worker.message_received.connect(
            self.handle_sensor_data, 
            Qt.QueuedConnection
        )
        self.mqtt_worker.device_discovered.connect(self.device_discovered)
        self.mqtt_worker.device_lost.connect(self.device_lost)
        self.mqtt_worker.connection_status.connect(self.update_connection_status)
        
        # Calculate initial ping interval (will update as devices connect)
        self.ping_interval = 1000 / self.sampling_rate  # Start with single device rate
        self.ping_timer = QTimer(self)
        self.ping_timer.timeout.connect(self.send_pings)
        self.update_ping_timer()
        
        # Start the thread
        self.mqtt_thread.started.connect(self.mqtt_worker.start)
        self.mqtt_thread.start()

    def update_ping_timer(self):
        if hasattr(self, 'mqtt_worker'):
            # Get recommended interval from worker
            interval = self.mqtt_worker.update_ping_timer()
            if interval:
                if self.ping_timer.isActive():
                    self.ping_timer.stop()
                self.ping_timer.start(int(interval))
                print(f"Updated ping interval: {interval}ms")

    def update_connection_status(self, status):
        self.connection_status_label.setText(status)

    def device_discovered(self, device_id):
        self.device_list.addItem(device_id)
        if self.device_list.count() == 1:
            self.device_list.setCurrentIndex(0)
        self.update_device_count()
        self.update_ping_timer()

    def device_lost(self, device_id):
        index = self.device_list.findText(device_id)
        if index >= 0:
            self.device_list.removeItem(index)
        self.update_device_count()
        self.update_ping_timer()

    def update_device_count(self):
        self.sensors_connected_label.setText(f"Connected Sensors: {self.device_list.count()}")

    def device_selected(self, index):
        if index >= 0:
            device_id = self.device_list.itemText(index)
            if device_id in self.mqtt_worker.last_values:
                self.update_device_display(self.mqtt_worker.last_values[device_id])

    def update_device_display(self, data):
        display_text = f"Sensor {data['device_id']} Data:\n\n"
        display_text += f"{'Timestamp':>12}: {data['timestamp']}\n"
        display_text += f"{'Acc X':>12}: {data['acc_X']:>6} (raw)\n"
        display_text += f"{'Acc Y':>12}: {data['acc_Y']:>6} (raw)\n"
        display_text += f"{'Acc Z':>12}: {data['acc_Z']:>6} (raw)\n"
        display_text += f"{'Quat W':>12}: {data['w']:>8.4f}\n"
        display_text += f"{'Quat X':>12}: {data['x']:>8.4f}\n"
        display_text += f"{'Quat Y':>12}: {data['y']:>8.4f}\n"
        display_text += f"{'Quat Z':>12}: {data['z']:>8.4f}\n"
        display_text += f"{'Gyro X':>12}: {data['gyro_X']:>6} (raw)\n"
        display_text += f"{'Gyro Y':>12}: {data['gyro_Y']:>6} (raw)\n"
        display_text += f"{'Gyro Z':>12}: {data['gyro_Z']:>6} (raw)\n"
        self.sensor_display.setPlainText(display_text)

    def handle_sensor_data(self, device_id, data):
        # Debug output
        print(f"Received data from {device_id}:")
        print(f"  Timestamp: {data['timestamp']}")
        print(f"  Acc: X={data['acc_X']}, Y={data['acc_Y']}, Z={data['acc_Z']}")
        print(f"  Quat: w={data['w']:.4f}, x={data['x']:.4f}, y={data['y']:.4f}, z={data['z']:.4f}")
        print(f"  Gyro: X={data['gyro_X']}, Y={data['gyro_Y']}, Z={data['gyro_Z']}")
        
        # Update display if this is the selected device
        if device_id == self.device_list.currentText():
            self.update_device_display(data)
        
        # Store data if recording
        if self.recording:
            try:
                with self.data_lock:
                    record_data = data.copy()
                    record_data['server_time'] = time.time() * 1000  # Add server timestamp
                    self.data[device_id].append(record_data)
                    self.actual_data_points += 1
                print(f"Stored data point #{self.actual_data_points} from {device_id}")
            except Exception as e:
                print(f"Error recording data: {e}")

    def start_recording(self):
        if not self.recording:
            with self.data_lock:
                self.recording = True
                self.recording_start_time = time.time()
                self.data = defaultdict(list)
                self.actual_data_points = 0
            self.recording_status_label.setText("Recording: ON")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            print("Recording STARTED - ready to receive data")
            print(f"Recording state: {self.recording}")

    def stop_recording(self):
        if self.recording:
            with self.data_lock:
                self.recording = False
                self.recording_stop_time = time.time()
                if self.recording_stop_time < self.recording_start_time:
                    print("Warning: Stop time before start time, resetting")
                    self.recording_stop_time = self.recording_start_time + 1
                print(f"Stop time set: {self.recording_stop_time}")
            self.recording_status_label.setText("Recording: OFF")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            print(f"Recording STOPPED - collected {self.actual_data_points} points")
            self.save_data()

    def save_data(self):
        if not any(len(data) > 0 for data in self.data.values()):
            QMessageBox.information(self, "No Data", "No data was recorded during this session.")
            return
            
        # Debug print to verify data
        for device_id, device_data in self.data.items():
            print(f"Device {device_id} has {len(device_data)} data points")
            
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Save Data")
        if not folder:
            return
            
        try:
            combined_df = self.combine_sensor_data()
            print("Combined DataFrame shape:", combined_df.shape)
            if combined_df.empty:
                print("Warning: Combined DataFrame is empty")
                QMessageBox.warning(self, "No Data", "The combined DataFrame is empty. No file saved.")
                return
            timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
            filename = f"{folder}/mpu6050_data_{timestamp}.xlsx"
            print(f"Saving to: {filename}")
            combined_df.to_excel(filename, index=False)
            print(f"File saved successfully")
            QMessageBox.information(self, "Success", f"Data successfully saved to:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save data: {str(e)}")
            print("Error during save:", e)

    def combine_sensor_data(self):
        if not self.data:
            print("No data in self.data")
            return pd.DataFrame()
        
        # Calculate recording duration in milliseconds
        duration_ms = (self.recording_stop_time - self.recording_start_time) * 1000
        num_samples = int(duration_ms * self.sampling_rate / 1000)
        print(f"Duration: {duration_ms}ms, Samples: {num_samples}")
        
        # Create regular time intervals based on sampling rate
        base_time = self.recording_start_time * 1000  # Convert to milliseconds
        time_step = 1000 / self.sampling_rate
        time_points = [base_time + i * time_step for i in range(num_samples)]
        
        # Create the base DataFrame - ensure unique index
        df = pd.DataFrame({
            'timestamp': time_points,
            'time_s': [(t - base_time)/1000.0 for t in time_points]
        }).set_index('timestamp')
        print(f"Base DataFrame shape: {df.shape}")
        
        # Process each device's data
        for device_id, device_data in self.data.items():
            if not device_data:
                print(f"No data for device {device_id}")
                continue
                
            # Create DataFrame from device data
            device_df = pd.DataFrame(device_data)
            print(f"Device {device_id} DataFrame shape: {device_df.shape}")
            print(f"Device {device_id} server_time range: {device_df['server_time'].min()} to {device_df['server_time'].max()}")
            
            # Ensure we have unique timestamps by adding small increments to duplicates
            device_df['server_time'] = self.make_timestamps_unique(device_df['server_time'])
            
            # Snap server_time to nearest time_point to preserve new values
            def snap_to_nearest_timepoint(t):
                # Find the nearest time point in time_points
                idx = np.argmin(np.abs(np.array(time_points) - t))
                return time_points[idx]
            
            device_df['server_time'] = device_df['server_time'].apply(snap_to_nearest_timepoint)
            print(f"Device {device_id} after snapping timestamps: {device_df['server_time'].head().tolist()}")
            
            # Set server_time as index for alignment
            device_df.set_index('server_time', inplace=True)
            
            # Aggregate duplicate timestamps by taking the most recent value (last)
            device_df = device_df.groupby(device_df.index).last()
            print(f"Device {device_id} after deduplication shape: {device_df.shape}")
            
            # Reindex to our regular time points
            temp_df = pd.DataFrame(index=time_points)
            device_df = temp_df.join(device_df, how='left')
            print(f"Device {device_id} after join shape: {device_df.shape}")
            
            # Forward fill to copy previous values for gaps, then backfill for leading NaNs
            device_df = device_df.ffill().bfill()
            print(f"Device {device_id} after fill shape: {device_df.shape}")
            
            # Rename columns with device prefix
            device_df = device_df.drop(columns=['device_id', 'timestamp'], errors='ignore')
            device_df = device_df.rename(columns={
                'acc_X': f'acc_X_{device_id}',
                'acc_Y': f'acc_Y_{device_id}',
                'acc_Z': f'acc_Z_{device_id}',
                'w': f'w_{device_id}',
                'x': f'x_{device_id}',
                'y': f'y_{device_id}',
                'z': f'z_{device_id}',
                'gyro_X': f'gyro_X_{device_id}',
                'gyro_Y': f'gyro_Y_{device_id}',
                'gyro_Z': f'gyro_Z_{device_id}'
            })
            
            # Merge with main DataFrame
            df = df.join(device_df, how='left')
            print(f"Merged DataFrame shape: {df.shape}")
        
        return df.reset_index()

    def make_timestamps_unique(self, timestamps):
        """Ensure all timestamps are unique by adding small increments to duplicates"""
        if len(timestamps) == len(set(timestamps)):
            return timestamps  # Already unique
        
        # Create a Series to track duplicates
        ts_series = pd.Series(timestamps)
        
        # Find duplicates and add small increments
        duplicates = ts_series.duplicated(keep=False)
        if duplicates.any():
            # Group by timestamp and add incrementing microseconds to duplicates
            groups = ts_series.groupby(ts_series).cumcount()
            increment = groups * 0.001  # Add 1ms to each duplicate
            ts_series = ts_series + increment
        
        return ts_series.values

    def update_ui(self):
        if self.recording:
            duration = time.time() - self.recording_start_time
            self.recording_duration_label.setText(f"Duration: {duration:.2f}s")
            self.data_points_label.setText(f"Data Points: {self.actual_data_points}")

    def send_pings(self):
        if hasattr(self, 'mqtt_worker'):
            self.mqtt_worker.send_next_ping()

    def check_device_timeouts(self):
        if hasattr(self, 'mqtt_worker'):
            self.mqtt_worker.check_device_timeouts()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            if self.recording:
                self.stop_recording()
            else:
                self.start_recording()
            print(f"Spacebar pressed, recording: {self.recording}")
        super().keyPressEvent(event)

    def closeEvent(self, event):
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
        
        # Clean up threads
        if hasattr(self, 'mqtt_thread'):
            self.mqtt_thread.quit()
            self.mqtt_thread.wait()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MQTTDataLogger()
    window.show()
    sys.exit(app.exec_())