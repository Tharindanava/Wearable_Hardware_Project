import sys
import socket
import time
import threading
import json
from collections import defaultdict
from datetime import datetime
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                           QPushButton, QWidget, QLabel, QComboBox, QFileDialog,
                           QTextEdit, QGroupBox, QMessageBox, QGridLayout)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject, QThread, QDateTime

class TCPServer(QObject):
    data_received = pyqtSignal(str, dict)  # device_id, data
    device_connected = pyqtSignal(str)
    device_disconnected = pyqtSignal(str)
    connection_status = pyqtSignal(str)
    
    def __init__(self, host='0.0.0.0', port=5000):
        super().__init__()
        self.host = host
        self.port = port
        self.server_socket = None
        self.active_connections = {}
        self.running = False
        self.data_buffer = defaultdict(list)
        self.lock = threading.Lock()
        self.device_last_seen = {}
        
    def start_server(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(25)
            self.running = True
            self.connection_status.emit(f"Server started on {self.host}:{self.port}")
            
            accept_thread = threading.Thread(target=self.accept_connections, daemon=True)
            accept_thread.start()
        except Exception as e:
            self.connection_status.emit(f"Failed to start server: {str(e)}")
    
    def accept_connections(self):
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                device_id = client_socket.recv(32).decode().strip()
                if device_id:
                    with self.lock:
                        self.active_connections[device_id] = client_socket
                        self.device_last_seen[device_id] = time.time()
                    self.device_connected.emit(device_id)
                    
                    current_time = int(time.time() * 1000)
                    client_socket.sendall(str(current_time).encode())
                    
                    handler_thread = threading.Thread(
                        target=self.handle_client,
                        args=(device_id, client_socket),
                        daemon=True
                    )
                    handler_thread.start()
            except Exception as e:
                if self.running:
                    self.connection_status.emit(f"Accept error: {str(e)}")
    
    def handle_client(self, device_id, client_socket):
        try:
            while self.running:
                raw_length = client_socket.recv(4)
                if not raw_length:
                    break
                
                length = int.from_bytes(raw_length, byteorder='big')
                data = client_socket.recv(length)
                
                if not data:
                    break
                
                try:
                    data_dict = json.loads(data.decode())
                    data_dict['device_id'] = device_id
                    data_dict['server_time'] = int(time.time() * 1000)
                    
                    with self.lock:
                        self.device_last_seen[device_id] = time.time()
                    
                    self.data_received.emit(device_id, data_dict)
                    
                    if device_id in self.data_buffer:
                        with self.lock:
                            self.data_buffer[device_id].append(data_dict)
                except json.JSONDecodeError:
                    print(f"Invalid JSON from {device_id}")
        except ConnectionResetError:
            pass
        finally:
            with self.lock:
                if device_id in self.active_connections:
                    del self.active_connections[device_id]
                if device_id in self.device_last_seen:
                    del self.device_last_seen[device_id]
                client_socket.close()
            self.device_disconnected.emit(device_id)
    
    def start_recording(self, device_id):
        with self.lock:
            self.data_buffer[device_id] = []
    
    def stop_recording(self, device_id):
        with self.lock:
            data = self.data_buffer.get(device_id, [])
            if device_id in self.data_buffer:
                del self.data_buffer[device_id]
            return data
    
    def check_device_timeouts(self, timeout=5):
        current_time = time.time()
        with self.lock:
            to_remove = [dev_id for dev_id, last_seen in self.device_last_seen.items() 
                        if current_time - last_seen > timeout]
            
            for dev_id in to_remove:
                if dev_id in self.active_connections:
                    self.active_connections[dev_id].close()
                    del self.active_connections[dev_id]
                if dev_id in self.device_last_seen:
                    del self.device_last_seen[dev_id]
                self.device_disconnected.emit(dev_id)
    
    def stop_server(self):
        self.running = False
        with self.lock:
            for device_id, sock in list(self.active_connections.items()):
                sock.close()
                del self.active_connections[device_id]
            self.device_last_seen.clear()
        if self.server_socket:
            self.server_socket.close()
        self.connection_status.emit("Server stopped")

class TCPDataLogger(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MPU6050 TCP Data Logger")
        self.setGeometry(100, 100, 1000, 800)
        
        self.host = '192.168.1.2'
        self.port = 5000
        self.sampling_rate = 64
        self.recording = False
        self.recording_start_time = 0
        self.recording_stop_time = 0
        self.data = defaultdict(list)
        self.data_lock = threading.Lock()
        self.actual_data_points = 0
        self.sensor_status_labels = {}
        self.last_values = {}  # Store last received values for each device
        
        self.init_ui()
        self.setup_server()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Status Group
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.connection_status_label = QLabel("Server not started")
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
        sensor_grid_group = QGroupBox("Sensor Status (Green: Connected, Red: Disconnected)")
        sensor_grid_layout = QGridLayout()
        
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
        
        # Setup timers
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(100)
        
        self.timeout_timer = QTimer(self)
        self.timeout_timer.timeout.connect(self.check_timeouts)
        self.timeout_timer.start(1000)

    def setup_server(self):
        self.server_thread = QThread()
        self.tcp_server = TCPServer(self.host, self.port)
        self.tcp_server.moveToThread(self.server_thread)
        
        self.tcp_server.data_received.connect(self.handle_sensor_data)
        self.tcp_server.device_connected.connect(self.device_connected)
        self.tcp_server.device_disconnected.connect(self.device_disconnected)
        self.tcp_server.connection_status.connect(self.update_connection_status)
        
        self.server_thread.started.connect(self.tcp_server.start_server)
        self.server_thread.start()

    def device_connected(self, device_id):
        self.device_list.addItem(device_id)
        if device_id in self.sensor_status_labels:
            self.sensor_status_labels[device_id].setStyleSheet(
                "background-color: green; color: white; border: 1px solid black;"
            )
        self.update_device_count()

    def device_disconnected(self, device_id):
        index = self.device_list.findText(device_id)
        if index >= 0:
            self.device_list.removeItem(index)
        if device_id in self.sensor_status_labels:
            self.sensor_status_labels[device_id].setStyleSheet(
                "background-color: red; color: white; border: 1px solid black;"
            )
        self.update_device_count()

    def check_timeouts(self):
        self.tcp_server.check_device_timeouts()

    def update_connection_status(self, status):
        self.connection_status_label.setText(status)

    def update_device_count(self):
        self.sensors_connected_label.setText(f"Connected Sensors: {self.device_list.count()}")

    def device_selected(self, index):
        if index >= 0:
            device_id = self.device_list.itemText(index)
            if device_id in self.last_values:
                self.update_device_display(self.last_values[device_id])

    def update_device_display(self, data):
        display_text = f"Sensor {data['device_id']} Data:\n\n"
        display_text += f"{'Timestamp':>12}: {data['timestamp']}\n"
        display_text += f"{'Acc X':>12}: {data['ax']:>6}\n"
        display_text += f"{'Acc Y':>12}: {data['ay']:>6}\n"
        display_text += f"{'Acc Z':>12}: {data['az']:>6}\n"
        display_text += f"{'Quat W':>12}: {data['qw']:>8.4f}\n"
        display_text += f"{'Quat X':>12}: {data['qx']:>8.4f}\n"
        display_text += f"{'Quat Y':>12}: {data['qy']:>8.4f}\n"
        display_text += f"{'Quat Z':>12}: {data['qz']:>8.4f}\n"
        display_text += f"{'Gyro X':>12}: {data['gx']:>6}\n"
        display_text += f"{'Gyro Y':>12}: {data['gy']:>6}\n"
        display_text += f"{'Gyro Z':>12}: {data['gz']:>6}\n"
        self.sensor_display.setPlainText(display_text)

    def handle_sensor_data(self, device_id, data):
        # Store last value for display
        self.last_values[device_id] = data
        
        # Update display if this is the selected device
        if device_id == self.device_list.currentText():
            self.update_device_display(data)
        
        # Store data if recording
        if self.recording:
            try:
                with self.data_lock:
                    record_data = data.copy()
                    record_data['server_time'] = time.time() * 1000
                    self.data[device_id].append(record_data)
                    self.actual_data_points += 1
            except Exception as e:
                print(f"Error recording data: {e}")

    def start_recording(self):
        if not self.recording:
            with self.data_lock:
                self.recording = True
                self.recording_start_time = time.time()
                self.data = defaultdict(list)
                self.actual_data_points = 0
            
            # Start recording on all connected devices
            for device_id in self.tcp_server.active_connections:
                self.tcp_server.start_recording(device_id)
            
            self.recording_status_label.setText("Recording: ON")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)

    def stop_recording(self):
        if self.recording:
            with self.data_lock:
                self.recording = False
                self.recording_stop_time = time.time()
            
            # Stop recording on all devices
            for device_id in self.tcp_server.active_connections:
                self.tcp_server.stop_recording(device_id)
            
            self.recording_status_label.setText("Recording: OFF")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.save_data()

    def save_data(self):
        if not any(len(data) > 0 for data in self.data.values()):
            QMessageBox.information(self, "No Data", "No data was recorded during this session.")
            return
            
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Save Data")
        if not folder:
            return
            
        try:
            combined_df = self.combine_sensor_data()
            if combined_df.empty:
                QMessageBox.warning(self, "No Data", "The combined DataFrame is empty.")
                return
                
            timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
            filename = f"{folder}/mpu6050_data_{timestamp}.xlsx"
            combined_df.to_excel(filename, index=False)
            QMessageBox.information(self, "Success", f"Data saved to:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save data: {str(e)}")

    def combine_sensor_data(self):
        if not self.data:
            return pd.DataFrame()
        
        # Create base DataFrame with regular timestamps
        duration = self.recording_stop_time - self.recording_start_time
        num_samples = int(duration * self.sampling_rate)
        base_time = self.recording_start_time * 1000
        time_step = 1000 / self.sampling_rate
        time_points = [base_time + i * time_step for i in range(num_samples)]
        
        df = pd.DataFrame({
            'timestamp': time_points,
            'time_s': [(t - base_time)/1000.0 for t in time_points]
        })
        
        # Process each device's data
        for device_id, device_data in self.data.items():
            if not device_data:
                continue
                
            device_df = pd.DataFrame(device_data)
            
            # Ensure unique timestamps
            device_df['server_time'] = self.make_timestamps_unique(device_df['server_time'])
            
            # Rename columns
            device_df = device_df.rename(columns={
                'ax': f'acc_X_{device_id}',
                'ay': f'acc_Y_{device_id}',
                'az': f'acc_Z_{device_id}',
                'qw': f'w_{device_id}',
                'qx': f'x_{device_id}',
                'qy': f'y_{device_id}',
                'qz': f'z_{device_id}',
                'gx': f'gyro_X_{device_id}',
                'gy': f'gyro_Y_{device_id}',
                'gz': f'gyro_Z_{device_id}'
            })
            
            # Drop unused columns
            device_df = device_df.drop(columns=['device_id', 'timestamp'], errors='ignore')
            
            # Merge with main DataFrame
            df = df.merge(device_df, left_on='timestamp', right_on='server_time', how='left')
            df = df.drop(columns=['server_time'], errors='ignore')
        
        return df

    def make_timestamps_unique(self, timestamps):
        if len(timestamps) == len(set(timestamps)):
            return timestamps
            
        ts_series = pd.Series(timestamps)
        duplicates = ts_series.duplicated(keep=False)
        if duplicates.any():
            groups = ts_series.groupby(ts_series).cumcount()
            increment = groups * 0.001
            ts_series = ts_series + increment
        
        return ts_series.values

    def update_ui(self):
        if self.recording:
            duration = time.time() - self.recording_start_time
            self.recording_duration_label.setText(f"Duration: {duration:.2f}s")
            self.data_points_label.setText(f"Data Points: {self.actual_data_points}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            if self.recording:
                self.stop_recording()
            else:
                self.start_recording()

    def closeEvent(self, event):
        if self.recording:
            reply = QMessageBox.question(
                self, 'Recording in Progress',
                'Stop recording and save before exiting?',
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
        
        self.tcp_server.stop_server()
        self.server_thread.quit()
        self.server_thread.wait()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TCPDataLogger()
    window.show()
    sys.exit(app.exec_())