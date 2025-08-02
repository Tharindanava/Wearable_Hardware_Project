import sys
import time
import socket
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                           QPushButton, QWidget, QLabel, QComboBox, QFileDialog,
                           QTextEdit, QGroupBox, QMessageBox, QGridLayout)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject, QThread, QDateTime
from collections import defaultdict
import threading
import subprocess

class UDPWorker(QObject):
    message_received = pyqtSignal(str, dict)  # device_id, data
    device_discovered = pyqtSignal(str)
    device_lost = pyqtSignal(str)
    connection_status = pyqtSignal(str)
    wifi_status_updated = pyqtSignal(str, bool)
    
    def __init__(self, port: int, sampling_rate: int):
        super().__init__()
        self.port = port
        self.sampling_rate = sampling_rate
        self.active_devices = set()
        self.last_values = {}
        self.device_last_seen = {}  # Stores last timestamp from sensor data (in ms)
        self.device_timeout = 0.5  # seconds
        self.wifi_status = {}  # device_id: bool (True if connected to WiFi)
        self.wifi_check_timer = QTimer()
        self.wifi_check_timer.timeout.connect(self.check_wifi_connections)
        self.wifi_check_timer.start(1000)  # Check WiFi every 5s
        self.running = True
        self.sock = None

    def start(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(('0.0.0.0', self.port))
            self.sock.settimeout(0.1)  # Small timeout to allow checking for thread exit
            self.connection_status.emit(f"UDP server listening on port {self.port}")
            
            while self.running:
                try:
                    data, addr = self.sock.recvfrom(1024)  # Buffer size
                    self.process_message(data.decode('utf-8'))
                except socket.timeout:
                    continue
                except Exception as e:
                    self.connection_status.emit(f"Error receiving data: {str(e)}")
        except Exception as e:
            self.connection_status.emit(f"Failed to start UDP server: {str(e)}")
        finally:
            if self.sock:
                self.sock.close()

    def stop(self):
        self.running = False

    def check_wifi_connections(self):
        """Check WiFi connectivity for all known devices"""
        for device_id in list(self.active_devices) + list(self.wifi_status.keys()):
            if device_id not in self.active_devices:
                # Only check devices that aren't currently active
                ip_address = f"192.168.1.{100 + int(device_id)}"
                is_connected = self.ping_device(ip_address)
                self.wifi_status[device_id] = is_connected
                self.wifi_status_updated.emit(device_id, is_connected)

    def ping_device(self, ip_address):
        """Ping a device to check WiFi connectivity"""
        try:
            # Use system ping command (works on both Windows and Linux)
            response = subprocess.run(
                ['ping', '-n', '1', '-w', '1000', ip_address],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return response.returncode == 0
        except Exception:
            return False

    def process_message(self, message):
        try:
            parts = message.split(',')
            
            # Handle heartbeat messages
            if len(parts) == 2 and parts[1] == "HEARTBEAT":
                device_id = parts[0]
                self.register_device(device_id)
                return
                
            # Handle sensor data messages
            if len(parts) == 12:
                device_id = parts[0]
                data = {
                    'timestamp': int(parts[1]),  # Assuming timestamp is in milliseconds
                    'acc_X': int(parts[2]),
                    'acc_Y': int(parts[3]),
                    'acc_Z': int(parts[4]),
                    'w': float(parts[5]),
                    'x': float(parts[6]),
                    'y': float(parts[7]),
                    'z': float(parts[8]),
                    'gyro_X': int(parts[9]),
                    'gyro_Y': int(parts[10]),
                    'gyro_Z': int(parts[11]),
                    'device_id': device_id,
                    'server_time': time.time() * 1000  # Server timestamp in ms
                }
                
                # Register device if not already registered
                if device_id not in self.active_devices:
                    self.register_device(device_id)
                
                # Update last seen timestamp from sensor data
                self.device_last_seen[device_id] = data['timestamp']
                
                self.last_values[device_id] = data
                self.message_received.emit(device_id, data)
                
                # Update WiFi status (if we're getting data, WiFi must be connected)
                self.wifi_status[device_id] = True
                self.wifi_status_updated.emit(device_id, True)
                
        except (ValueError, IndexError) as e:
            print(f"Error parsing data: {e}")
            print(f"Raw message: {message}")

    def register_device(self, device_id: str):
        """Register a new device or update an existing one"""
        if device_id not in self.active_devices:
            self.active_devices.add(device_id)
            self.device_discovered.emit(device_id)
            self.wifi_status[device_id] = True
            self.wifi_status_updated.emit(device_id, True)
            
            # Track when device first connected
            if not hasattr(self, 'device_connection_times'):
                self.device_connection_times = {}
            self.device_connection_times[device_id] = time.time() * 1000
            
            print(f"New device discovered: {device_id}")

    def check_device_timeouts(self):
        """Check if any devices have stopped sending data"""
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
        
        # Check devices that connected but never sent data
        for device_id in list(self.active_devices):
            if device_id not in self.device_last_seen:
                # If device never sent any data, check connection time
                if hasattr(self, 'device_connection_times'):
                    if device_id in self.device_connection_times:
                        if current_time - self.device_connection_times[device_id] > timeout_ms:
                            print(f"Device {device_id} connected but never sent data")
                            self.remove_device(device_id)
                else:
                    # Initialize connection times tracking
                    self.device_connection_times = {}
                    for dev in self.active_devices:
                        self.device_connection_times[dev] = current_time

    def remove_device(self, device_id: str):
        if device_id in self.active_devices:
            self.active_devices.remove(device_id)
            if device_id in self.device_last_seen:
                del self.device_last_seen[device_id]
            if hasattr(self, 'device_connection_times') and device_id in self.device_connection_times:
                del self.device_connection_times[device_id]
            self.device_lost.emit(device_id)
            print(f"Device lost: {device_id}")

class UDPDataLogger(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MPU6050 Data Logger (UDP)")
        self.setGeometry(100, 100, 1000, 800)
        
        self.port = 12345  # Must match ESP32 code
        self.sampling_rate = 50  # Hz
        self.recording = False
        self.recording_start_time = 0
        self.recording_stop_time = 0
        self.data = defaultdict(list)
        self.data_lock = threading.Lock()
        self.sensor_status_labels = {}
        self.actual_data_points = 0
        
        self.init_ui()
        self.setup_udp()

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
        
        # Sensor Status Grid
        sensor_grid_group = QGroupBox("Sensor Status (Green: Publishing, Blue: WiFi Only, Red: Disconnected)")
        sensor_grid_layout = QGridLayout()
        
        # Create a 5x5 grid (for 25 sensors)
        for i in range(25):
            sensor_id = f"{i+1}"
            label = QLabel(f"{i+1}")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("background-color: red; color: white; border: 1px solid black;")
            label.setMinimumSize(40, 40)
            row = i // 5
            col = i % 5
            sensor_grid_layout.addWidget(label, row, col)
            self.sensor_status_labels[sensor_id] = label
        
        sensor_grid_group.setLayout(sensor_grid_layout)
        
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
        main_layout.addWidget(sensor_grid_group)
        main_layout.addWidget(sensor_group)
        
        # Setup timers in main thread
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(100)  # 10 Hz UI update
        
        self.monitor_timer = QTimer(self)
        self.monitor_timer.timeout.connect(self.check_device_timeouts)
        self.monitor_timer.start(1000)  # Check every second

    def setup_udp(self):
        self.udp_thread = QThread()
        self.udp_worker = UDPWorker(self.port, self.sampling_rate)
        self.udp_worker.moveToThread(self.udp_thread)
        
        # Connect signals (same as MQTT version)
        self.udp_worker.message_received.connect(self.handle_sensor_data)
        self.udp_worker.device_discovered.connect(self.device_discovered)
        self.udp_worker.device_lost.connect(self.device_lost)
        self.udp_worker.connection_status.connect(self.update_connection_status)
        self.udp_worker.wifi_status_updated.connect(self.update_wifi_status)
        
        # Start the thread
        self.udp_thread.started.connect(self.udp_worker.start)
        self.udp_thread.start()

    def update_wifi_status(self, device_id: str, is_connected: bool):
        """Update the status indicator for a device based on WiFi and data status"""
        if device_id in self.sensor_status_labels:
            label = self.sensor_status_labels[device_id]
            if device_id in self.udp_worker.active_devices:
                # Device is publishing data - green
                label.setStyleSheet("background-color: green; color: white; border: 1px solid black;")
            elif is_connected:
                # Device is on WiFi but not publishing - blue
                label.setStyleSheet("background-color: blue; color: white; border: 1px solid black;")
            else:
                # Device not connected to WiFi - red
                label.setStyleSheet("background-color: red; color: white; border: 1px solid black;")

    def handle_sensor_data(self, device_id, data):
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

    
    def update_connection_status(self, status):
        self.connection_status_label.setText(status)

    def device_discovered(self, device_id):
        self.device_list.addItem(device_id)
        if self.device_list.count() == 1:
            self.device_list.setCurrentIndex(0)
        self.update_device_count()
        
        # Update status indicator
        if device_id in self.sensor_status_labels:
            self.sensor_status_labels[device_id].setStyleSheet(
                "background-color: green; color: white; border: 1px solid black;"
            )

    def device_lost(self, device_id):
        index = self.device_list.findText(device_id)
        if index >= 0:
            self.device_list.removeItem(index)
        self.update_device_count()
        
        # Update status indicator - check WiFi status
        if device_id in self.sensor_status_labels:
            is_connected = self.udp_worker.wifi_status.get(device_id, False)
            if is_connected:
                self.sensor_status_labels[device_id].setStyleSheet(
                    "background-color: blue; color: white; border: 1px solid black;"
                )
            else:
                self.sensor_status_labels[device_id].setStyleSheet(
                    "background-color: red; color: white; border: 1px solid black;"
                )

    def update_device_count(self):
        self.sensors_connected_label.setText(f"Connected Sensors: {self.device_list.count()}")

    def device_selected(self, index):
        if index >= 0:
            device_id = self.device_list.itemText(index)
            if device_id in self.udp_worker.last_values:
                self.update_device_display(self.udp_worker.last_values[device_id])

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
        
        # Create the base DataFrame with regular time intervals
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
                
            # Create DataFrame from device data and drop the device timestamp
            device_df = pd.DataFrame(device_data).drop(columns=['timestamp'], errors='ignore')
            print(f"Device {device_id} DataFrame shape: {device_df.shape}")
            
            # Use server_time as our reference
            # Ensure timestamps are unique by adding small increments if needed
            device_df['server_time'] = self.make_timestamps_unique(device_df['server_time'])
            
            # Set server_time as index
            device_df.set_index('server_time', inplace=True)
            
            # Resample to our regular time grid using nearest neighbor interpolation
            resampled_df = device_df.reindex(time_points, method='nearest')
            
            # Rename columns with device prefix
            resampled_df = resampled_df.drop(columns=['device_id'], errors='ignore')
            resampled_df = resampled_df.rename(columns={
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
            df = df.join(resampled_df, how='left')
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

    def check_device_timeouts(self):
        if hasattr(self, 'udp_worker'):
            self.udp_worker.check_device_timeouts()

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
        if hasattr(self, 'udp_thread'):
            self.udp_worker.stop()
            self.udp_thread.quit()
            self.udp_thread.wait()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UDPDataLogger()
    window.show()
    sys.exit(app.exec_())