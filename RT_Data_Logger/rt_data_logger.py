import socket
import threading
import time
import queue
from collections import deque

class IMUDataLogger:
    """
    IMUDataLogger manages real-time data collection from multiple IMU (Inertial Measurement Unit) devices over UDP.
    This class is designed to:
    - Listen to multiple IMU devices on configurable IPs and ports.
    - Buffer incoming sensor data for each device.
    - Track device connectivity and health status.
    - Parse and process sensor data in CSV format.
    - Provide access to buffered data and device status.
    - Send real-time sample-by-sample data to another UDP socket.

    Attributes:
        devices (dict): Information about each device (IP, port, connection status, last seen).
        data_buffers (dict): Buffers for storing parsed sensor data per device.
        active_devices (set): Set of device IDs currently sending data.
        data_queues (dict): Queues for incoming raw data per device.
        health_check_interval (float): Interval (seconds) for device health checks.
        last_received_time (dict): Timestamps of last data received per device.
        running (bool): Indicates if logging is active.
        threads (list): List of threads for device listening, health checking, and data processing.
        realtime_socket (socket): UDP socket for sending real-time data.
        realtime_target (tuple): Target (IP, port) for real-time data streaming.

    Methods:
        start_logging():
            Start threads for listening to devices, health checking, and data processing.
        stop_logging():
            Stop all threads and end logging.
        _listen_to_device(device_id):
            Internal method to listen for UDP packets from a specific device.
        _parse_sensor_data(data_str, device_id):
            Parse CSV-formatted sensor data string from a device.
        _process_data():
            Internal method to process and buffer incoming data from all devices.
        _health_check():
            Internal method to periodically check device connectivity.
        set_health_check_interval(interval):
            Set the health check interval in seconds.
        get_active_devices():
            Return a list of device IDs currently sending data.
        get_device_status(device_id):
            Get status and statistics for a specific device.
        _calculate_data_rate(device_id):
            Calculate the approximate data rate for a device.
        get_all_data():
            Return all buffered data for all devices.
        get_device_data(device_id):
            Return buffered data for a specific device.
        print_latest_data(device_id=None):
            Print the latest data from one or all devices.
        set_realtime_target(ip, port):
            Set the target for real-time data streaming.
        send_realtime_data(parsed_data):
            Send parsed data to the real-time UDP socket.
    """

    def __init__(self):
        self.devices = {}
        self.data_buffers = {}
        self.active_devices = set()
        self.data_queues = {}
        self.health_check_interval = 1.0  # seconds
        self.last_received_time = {}
        self.running = False
        
        # Device configuration
        self.DEVICE_COUNT = 25
        self.BASE_IP = '192.168.110.'
        self.START_IP = 100
        self.START_PORT = 12300
        self.SAMPLE_RATE = 100  # Hz
        self.BUFFER_DURATION = 300  # 5 minutes in seconds

        # Real-time streaming configuration
        self.realtime_socket = 12345
        self.realtime_target = '192.168.110.2'
        self.realtime_enabled = True

        # Calculate buffer size needed for 5 minutes at 100Hz
        BUFFER_SIZE = self.BUFFER_DURATION * self.SAMPLE_RATE
        
        # Initialize device information
        for i in range(self.DEVICE_COUNT):
            device_id = i + 1
            ip = f"{self.BASE_IP}{self.START_IP + i}"
            port = self.START_PORT + i
            self.devices[device_id] = {
                'ip': ip,
                'port': port,
                'id': device_id,
                'connected': False,
                'last_seen': None
            }
            self.data_buffers[device_id] = deque(maxlen=BUFFER_SIZE)
            self.data_queues[device_id] = queue.Queue()
            self.last_received_time[device_id] = time.time()
    
    def set_realtime_target(self, ip, port):
        """Set the target for real-time data streaming"""
        try:
            self.realtime_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.realtime_target = (ip, port)
            self.realtime_enabled = True
            print(f"Real-time streaming enabled to {ip}:{port}")
        except Exception as e:
            print(f"Error setting up real-time socket: {e}")
            self.realtime_enabled = False
    
    def send_realtime_data(self, parsed_data):
        """Send parsed data to the real-time UDP socket"""
        if not self.realtime_enabled or not self.realtime_socket:
            return
        
        try:
            # Convert parsed data back to CSV format for transmission
            data_str = f"{parsed_data['device_id']},{parsed_data['timestamp']},{parsed_data['ax']},{parsed_data['ay']},{parsed_data['az']},{parsed_data['gx']},{parsed_data['gy']},{parsed_data['gz']}"
            self.realtime_socket.sendto(data_str.encode('utf-8'), self.realtime_target)
        except Exception as e:
            print(f"Error sending real-time data: {e}")
    
    def start_logging(self):
        self.running = True
        
        # Create and start a thread for each device
        self.threads = []
        for device_id in self.devices:
            thread = threading.Thread(
                target=self._listen_to_device, 
                args=(device_id,),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
        
        # Start health check thread
        health_thread = threading.Thread(target=self._health_check, daemon=True)
        health_thread.start()
        self.threads.append(health_thread)
        
        # Start data processing thread
        process_thread = threading.Thread(target=self._process_data, daemon=True)
        process_thread.start()
        self.threads.append(process_thread)
        
        print(f"Started logging for {self.DEVICE_COUNT} devices")
    
    def stop_logging(self):
        self.running = False
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        # Close real-time socket if it exists
        if self.realtime_socket:
            self.realtime_socket.close()
            
        print("Stopped logging")
    
    def _listen_to_device(self, device_id):
        device = self.devices[device_id]
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
        
        try:
            sock.bind(('', device['port']))
            print(f"Listening to device {device_id} at {device['ip']}:{device['port']}")
            
            while self.running:
                try:
                    data, addr = sock.recvfrom(1024)
                    if addr[0] == device['ip']:
                        self.data_queues[device_id].put(data)
                        self.last_received_time[device_id] = time.time()
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Error receiving from device {device_id}: {e}")
                    break
        finally:
            sock.close()
    
    def _parse_sensor_data(self, data_str, device_id):
        """Parse the CSV format: device_id,timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z"""
        try:
            parts = data_str.split(',')
            if len(parts) != 8:
                print(f"Invalid data format from device {device_id}: {data_str}")
                return None
            
            # Convert parts to appropriate data types
            return {
                'device_id': parts[0],
                'timestamp': int(parts[1]),
                'ax': int(parts[2]),
                'ay': int(parts[3]),
                'az': int(parts[4]),
                'gx': int(parts[5]),
                'gy': int(parts[6]),
                'gz': int(parts[7]),
                'received_timestamp': time.time()
            }
        except (ValueError, IndexError) as e:
            print(f"Error parsing data from device {device_id}: {e}, Data: {data_str}")
            return None
    
    def _process_data(self):
        while self.running:
            for device_id in self.devices:
                try:
                    # Process all available data in the queue without blocking too long
                    while True:
                        try:
                            data = self.data_queues[device_id].get_nowait()
                            try:
                                data_str = data.decode('utf-8').strip()
                                parsed_data = self._parse_sensor_data(data_str, device_id)
                                
                                if parsed_data:
                                    self.data_buffers[device_id].append(parsed_data)
                                    
                                    # Send real-time data if enabled
                                    if self.realtime_enabled:
                                        self.send_realtime_data(parsed_data)
                                    
                                    # Mark device as active
                                    self.active_devices.add(device_id)
                                    self.devices[device_id]['connected'] = True
                                    self.devices[device_id]['last_seen'] = time.time()
                                
                            except UnicodeDecodeError:
                                print(f"Encoding error from device {device_id}: {data}")
                        except queue.Empty:
                            break
                except Exception as e:
                    print(f"Error processing data for device {device_id}: {e}")
            
            time.sleep(0.01)  # Small sleep to prevent CPU overload
    
    def _health_check(self):
        while self.running:
            current_time = time.time()
            for device_id in self.devices:
                time_since_last = current_time - self.last_received_time[device_id]
                
                # If no data received for twice the health check interval, mark as disconnected
                if time_since_last > self.health_check_interval * 2:
                    if device_id in self.active_devices:
                        print(f"Device {device_id} stopped sending data")
                        self.active_devices.remove(device_id)
                        self.devices[device_id]['connected'] = False
            
            time.sleep(self.health_check_interval)
    
    def set_health_check_interval(self, interval):
        """Set the health check interval in seconds"""
        self.health_check_interval = interval
        print(f"Health check interval set to {interval} seconds")
    
    def get_active_devices(self):
        """Return list of actively sending devices"""
        return list(self.active_devices)
    
    def get_device_status(self, device_id):
        """Get status information for a specific device"""
        device = self.devices.get(device_id)
        if device:
            status = device.copy()
            status['buffer_size'] = len(self.data_buffers[device_id])
            status['data_rate'] = self._calculate_data_rate(device_id)
            return status
        return None
    
    def _calculate_data_rate(self, device_id):
        """Calculate approximate data rate for a device"""
        buffer = self.data_buffers[device_id]
        if len(buffer) < 2:
            return 0
        
        # Calculate based on timestamps of first and last data points
        first_time = buffer[0].get('received_timestamp', 0)
        last_time = buffer[-1].get('received_timestamp', 0)
        
        if last_time <= first_time or len(buffer) < 2:
            return 0
        
        time_span = last_time - first_time
        return len(buffer) / time_span if time_span > 0 else 0
    
    def get_all_data(self):
        """Return all buffered data"""
        return {device_id: list(self.data_buffers[device_id]) for device_id in self.devices}
    
    def get_device_data(self, device_id):
        """Return buffered data for a specific device"""
        return list(self.data_buffers.get(device_id, []))
    
    def print_latest_data(self, device_id=None):
        """Print the latest data from one or all devices"""
        if device_id:
            if device_id in self.data_buffers and self.data_buffers[device_id]:
                latest = self.data_buffers[device_id][-1]
                print(f"Device {device_id}: {latest}")
        else:
            for dev_id in self.active_devices:
                if self.data_buffers[dev_id]:
                    latest = self.data_buffers[dev_id][-1]
                    print(f"Device {dev_id}: {latest}")