import socket
import threading
import argparse
from datetime import datetime

class UDPReceiver:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.running = False
        self.socket = None
        self.receive_count = 0
        
    def start(self):
        """Start listening for UDP packets"""
        try:
            # Create UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # Allow socket reuse
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind to the specified IP and port
            self.socket.bind((self.ip, self.port))
            
            print(f"Listening for UDP packets on {self.ip}:{self.port}")
            print("Press Ctrl+C to stop...")
            
            self.running = True
            self.receive_count = 0
            
            # Start receiving data
            while self.running:
                try:
                    data, addr = self.socket.recvfrom(4096)  # Buffer size is 4096 bytes
                    self.receive_count += 1
                    
                    # Display information about received data
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    print(f"\n[{timestamp}] Received {len(data)} bytes from {addr[0]}:{addr[1]}")
                    
                    # Try to decode as UTF-8 text
                    try:
                        decoded_data = data.decode('utf-8').strip()
                        print(f"Text data: {decoded_data}")
                    except UnicodeDecodeError:
                        print("Binary data (hex):", data.hex())
                        
                    print(f"Total packets received: {self.receive_count}")
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:  # Only print errors if we're supposed to be running
                        print(f"Error receiving data: {e}")
                    
        except Exception as e:
            print(f"Error setting up UDP listener: {e}")
        finally:
            if self.socket:
                self.socket.close()
                
    def stop(self):
        """Stop listening for UDP packets"""
        self.running = False
        if self.socket:
            self.socket.close()

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='UDP Data Receiver')
    parser.add_argument('-i', '--ip', default='0.0.0.0', 
                       help='IP address to listen on (default: 0.0.0.0 - all interfaces)')
    parser.add_argument('-p', '--port', type=int, required=True,
                       help='Port number to listen on')
    
    args = parser.parse_args()
    
    # Create and start the receiver
    receiver = UDPReceiver(args.ip, args.port)
    
    try:
        receiver.start()
    except KeyboardInterrupt:
        print("\nStopping UDP receiver...")
        receiver.stop()

if __name__ == "__main__":
    main()