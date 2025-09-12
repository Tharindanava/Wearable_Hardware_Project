import socket
import json


ESP32_IP = '192.168.8.76'
DATA_PORT = 5001


def main():
    try:
        # TCP socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((ESP32_IP, DATA_PORT))
            socket_file = sock.makefile('r')
            for line in socket_file:
                try:
                    data = json.loads(line)

                    print(
                        f"TS: {data.get('timestamp', 0):<10} | Device: {data.get('id', 0)} "
                        f"Accel: [{data.get('ax', 0):>6.2f}, {data.get('ay', 0):>6.2f}, {data.get('az', 0):>6.2f}] | "
                        f"Gyro: [{data.get('gx', 0):>7.2f}, {data.get('gy', 0):>7.2f}, {data.get('gz', 0):>7.2f}]"
                    )

                except json.JSONDecodeError:
                    print(f"Received a non-JSON line: {line.strip()}")
                except KeyboardInterrupt:

                    break

    except ConnectionRefusedError:
        print(f"Connection refused. Make sure a recording is active on {ESP32_IP}.")
    except socket.timeout:
        print("Connection timed out.")
    except KeyboardInterrupt:
        print("\nExiting.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Stream disconnected.")


if __name__ == "__main__":
    main()