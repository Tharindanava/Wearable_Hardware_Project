import sys
import os
import time
import socket
import json

# Add the package folder itself to sys.path
sys.path.append(r"C:/Users/acer/Documents/GitHub/kalman-universe/ekf-quat-py")

import rt_data_logger as rtdl
import rt_kalman as rtk


# Main application
def main():
    logger = rtdl.IMUDataLogger()

    # Configuration - adjust these values as needed
    SENSOR_PAIRS = [(22, 23)]  # Define which sensor pairs to track
    
    filter = rtk.RealTimeKalmanFilter(
        input_udp_ip="192.168.1.2",   # Where IMU data is coming from
        input_udp_port=5000,          # Port where IMU data is sent
        output_udp_ip="192.168.1.2",  # Where to send quaternion results
        output_udp_port=5001,         # Port for quaternion output
        sensor_pairs=SENSOR_PAIRS,
        sampling_rate=100.0
    )
    
    try:
        logger.start_logging()
        
        # Example: Set health check interval to 1 second
        logger.set_health_check_interval(1.0)
        
        filter.start()
        
        # Keep main thread alive
        while True:
            # Print status periodically
            active_devices = logger.get_active_devices()
            print(f"Active devices: {len(active_devices)}/{logger.DEVICE_COUNT}")
            
            # Print latest data from each active device
            logger.print_latest_data()
            
            for device_id in active_devices:
                status = logger.get_device_status(device_id)
                if status and status['connected']:
                    buffer_size = status['buffer_size']
                    data_rate = status['data_rate']
                    print(f"Device {device_id}: {buffer_size} samples, {data_rate:.1f} Hz")
            
            print("---")

            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        logger.stop_logging()
        
        # Get all buffered data before exiting
        all_data = logger.get_all_data()
        print(f"Retrieved {sum(len(data) for data in all_data.values())} data points")
        
        # You can process the data here or return it as needed
        UDP_IP = "192.168.1.2"  # Change as needed
        UDP_PORT = 5000         # Change as needed

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Combine all device data into a single payload
        payload = {
            "all_devices_data": all_data
        }
        message = json.dumps(payload).encode('utf-8')
        sock.sendto(message, (UDP_IP, UDP_PORT))

        sock.close()

        filter.stop()

if __name__ == "__main__":
    main()