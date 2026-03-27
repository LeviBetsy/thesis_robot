import serial
import time
from enum import Enum

class Instruction_t(Enum):
    IDLE = '0'
    FORWARD = '1'
    BACKWARD = '2'
    LEFT = '3'
    RIGHT = '4'


class SerialManager:
    def __init__(self, port='/dev/ttyAMA0', baudrate=115200, timeout=1):
        """Initializes the serial connection parameters."""
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None

    def connect(self):
        """Attempts to open the serial port."""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=self.timeout
            )
            print(f"Serial port {self.port} opened successfully.")
        except Exception as e:
            print(f"Error opening serial port: {e}")
            self.ser = None
            raise # Re-raise the exception to handle it in the calling code

    def send_string(self, message):
        """Encodes a string to bytes and sends it over the serial port."""
        if self.ser and self.ser.is_open:
            try:
                # Strings must be encoded to bytes before sending
                self.ser.write(message.encode('utf-8'))
                print(f"Sent: {message}")
            except Exception as e:
                print(f"Failed to send data: {e}")
        else:
            print("Serial port is not open. Cannot send data.")

    def close(self):
        """Closes the serial port connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"Serial port {self.port} closed.")

# --- Example Usage ---
if __name__ == "__main__":
    # Create an instance of the class
    communicator = SerialManager(port='/dev/ttyAMA0', baudrate=115200)

    #note that exception does not need to be of type KeyboardInterrupt, it can be any exception that goes to finally
    try:
        communicator.connect()
        # # Input the string you want to send here
        # communicator.send_string("Hello World\n")
        while True:
            cmd = input("Enter a command to send (or 'exit' to quit): ")
            communicator.send_string(cmd)
            time.sleep(1)  # Keep the program running to maintain the serial connection
        
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        communicator.close()