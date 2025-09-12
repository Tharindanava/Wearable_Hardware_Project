import time
import socket
import json

import rt_data_logger as rtdl

def main():
    logger = rtdl.IMUDataLogger()
    
    try:
        logger.start_logging()
        
        # Example: Set health check interval to 1 second
        logger.set_health_check_interval(1.0)
        
        # Main loop
        while True:
            time.sleep(5)
            
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
        return all_data


if __name__ == "__main__":
    main()