import serial
import threading
import time
import struct #for unpacking big-Endian message
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

    def send_command(self, inst_t, leftduty, rightduty):
        if self.ser and self.ser.is_open:
            try:
                header = 0xAA
                l_high, l_low = (leftduty >> 8) & 0xFF, leftduty & 0xFF
                r_high, r_low = (rightduty >> 8) & 0xFF, rightduty & 0xFF
                inst = inst_t & 0xFF
                checksum = inst ^ l_high ^ l_low ^ r_high ^ r_low
                packet = bytearray([header, inst, l_high, l_low, r_high, r_low, checksum])
                self.ser.write(packet)
                print(f"Sent")
            except Exception as e:
                print(f"Failed to send data: {e}")
        else:
            print("Serial port is not open. Cannot send data.")

    def poll_receive(self):
        """Continuously polls the serial port for the 4-byte packet."""
        if not (self.ser and self.ser.is_open):
            print("Serial port is not open. Cannot receive data.")
            return

        print("Listening for incoming data...")
        
        while self.ser.is_open:
            try:
                # Block until at least one byte is in the buffer
                if self.ser.in_waiting > 0:
                    # Read one byte at a time looking for the start byte
                    start_byte = self.ser.read(1)
                    
                    if start_byte == b'\xAA':
                        # Read the remaining 5 bytes: LC_MSB, LC_LSB, RC_MSB, RC_LSB, checksum
                        payload = self.ser.read(5)
                        
                        if len(payload) == 5:
                            # '>' means Big-Endian (High Byte first)
                            # 'h' means signed 16-bit integer (takes 2 bytes each)
                            # 'B' means unsigned 8-bit integer (checksum)
                            data1_signed, data2_signed, received_checksum = struct.unpack('>hhB', payload)
                            
                            # Reconstruct the raw bytes to calculate the expected checksum
                            lc_high = (data1_signed >> 8) & 0xFF
                            lc_low  = data1_signed & 0xFF
                            rc_high = (data2_signed >> 8) & 0xFF
                            rc_low  = data2_signed & 0xFF
                            
                            # Note: The C code XORs 0xAA in its checksum calculation!
                            expected_checksum = lc_high ^ lc_low ^ rc_high ^ rc_low
                            
                            with open("uart_log.txt", "a") as log_file:
                                if received_checksum == expected_checksum:
                                    log_file.write(f"Valid Packet -> LCount: {data1_signed}, RCount: {data2_signed}\n")
                                else:
                                    print("Incomplete packet received. Waiting for next start byte.")
            
            except serial.SerialException:
                # Exit cleanly if the port is closed while reading
                break
            except Exception as e:
                print(f"Receive error: {e}")

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
        receive_thread = threading.Thread(target=communicator.poll_receive, daemon=True)
        receive_thread.start()

        # # Input the string you want to send here
        # communicator.send_string("Hello World\n")
        while True:
            # communicator.poll_receive()

            print("Enter a command to send (enter in the format INST,LEFTDUTY,RIGHTCYCLE ): ")
            print("INST: 0-4 for STOP, FORWARD, BACKWARD, lEFT, RIGHT")
            print("DUTYCYCLE from 0-14998")
            cmd = input()
            parts = cmd.split(',')
            communicator.send_command(int(parts[0]), int(parts[1]), int(parts[2]))

            communicator.send_string(cmd)
            time.sleep(1)  # Keep the program running to maintain the serial connection
        
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        communicator.close()