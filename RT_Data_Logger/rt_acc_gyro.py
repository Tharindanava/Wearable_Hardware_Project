import time
import socket
import json

import rt_data_logger as rtdl

"""
This script logs real-time IMU (accelerometer and gyroscope) data from multiple devices using the rt_data_logger module.
It periodically prints device status and latest data, and on shutdown, retrieves all buffered data.
The collected data is sent over UDP in manageable chunks to avoid packet size limitations.
Usage: Configure the target IP and port for real-time and bulk data transfer as needed.
"""

def main():
    logger = rtdl.IMUDataLogger()
    logger.set_realtime_target('192.168.1.2', 12345)  # Set target for real-time data
    
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
                #endif
            #endfor
            
            print("---")
        
    except KeyboardInterrupt:
        print("\nShutting down...")
        
    finally:
        logger.stop_logging()
        
        # Get all buffered data before exiting
        all_data = logger.get_all_data()
        total_points = sum(len(data) for data in all_data.values())
        print(f"Retrieved {total_points} data points")
        
        # Send data in smaller chunks to avoid UDP packet size limitations
        UDP_IP = "192.168.1.2"  # Change as needed
        UDP_PORT = 12346         # Change as needed

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Calculate maximum payload size (conservative estimate for UDP)
        MAX_UDP_PAYLOAD = 1400  # bytes (safe for most networks)
        
        # Convert data to JSON and split into chunks
        all_data_json = json.dumps(all_data)
        chunk_size = MAX_UDP_PAYLOAD
        
        # Split the data into chunks
        chunks = [all_data_json[i:i+chunk_size] for i in range(0, len(all_data_json), chunk_size)]
        
        # Send metadata first (number of chunks, total size)
        metadata = {
            "total_chunks": len(chunks),
            "total_size": len(all_data_json),
            "data_points": total_points
        }
        sock.sendto(json.dumps(metadata).encode('utf-8'), (UDP_IP, UDP_PORT))
        
        # Send each chunk with sequence number
        for i, chunk in enumerate(chunks):
            chunk_payload = {
                "sequence": i,
                "total": len(chunks),
                "data": chunk
            }
            message = json.dumps(chunk_payload).encode('utf-8')
            sock.sendto(message, (UDP_IP, UDP_PORT))
            time.sleep(0.001)  # Small delay to prevent packet loss
        
        # Send completion message
        completion = {"status": "complete"}
        sock.sendto(json.dumps(completion).encode('utf-8'), (UDP_IP, UDP_PORT))
        
        print(f"Sent {len(chunks)} chunks of data to {UDP_IP}:{UDP_PORT}")
        
        sock.close()
        return all_data


if __name__ == "__main__":
    main()