import sys
import os
import time
import socket
import json

# Add the package folder itself to sys.path
sys.path.append(r"C:/Users/acer/Documents/GitHub/kalman-universe/ekf-quat-py")

import rt_data_logger as rtdl
import rt_kalman as rtk

"""
Main function for real-time IMU data logging and relative angle computation.
This function initializes and manages the data logging and processing pipeline for IMU sensors.
It configures sensor pairs, sets up UDP communication for both input (IMU data) and output (quaternion results),
and starts a real-time Kalman filter for sensor fusion. The function periodically prints device status,
handles graceful shutdown on interruption, retrieves all buffered data, and sends it over UDP in manageable chunks
to avoid packet size limitations. Finally, it stops the filter and prints its final state.
# Description:
# This code sets up a real-time IMU data logger and Kalman filter for tracking relative angles between sensor pairs.
# It manages UDP communication for both incoming sensor data and outgoing processed results, monitors device status,
# and ensures safe shutdown by sending all collected data in chunks over UDP.
"""

# Main application
def main():

    logger = rtdl.IMUDataLogger()

    # Configuration - adjust these values as needed
    SENSOR_PAIRS = [(22, 23)]  # Define which sensor pairs to track
    
    filter = rtk.RealTimeKalmanFilter(
        input_udp_ip="192.168.1.2",   # Where IMU data is coming from
        input_udp_port=12345,          # Port where IMU data is sent
        output_udp_ip="192.168.1.2",  # Where to send quaternion results
        output_udp_port=12347,         # Port for quaternion output
        sensor_pairs=SENSOR_PAIRS,
        sampling_rate=100.0
    )
    
    try:
        logger.start_logging()
        
        # Example: Set health check interval to 1 second
        logger.set_health_check_interval(1.0)
        
        filter.start()
        
        # Keep main thread alive - monitor status
        status_counter = 0
        while True:
            time.sleep(0.1)  # Small sleep to prevent CPU overload
            status_counter += 1
            
            # Print status every 10 iterations (approx 1 second)
            if status_counter % 10 == 0:
                # Print status periodically
                active_devices = logger.get_active_devices()
                print(f"Active IMU devices: {len(active_devices)}/{logger.DEVICE_COUNT}")
                
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

        # Stop the filter and get final state
        filter_state = filter.stop()
        print(f"Filter final state: {filter_state}")


if __name__ == "__main__":
    main()