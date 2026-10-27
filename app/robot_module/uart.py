import serial
import threading
import time
import struct #for unpacking big-Endian message
from enum import Enum
import queue

class Instruction_t(Enum):
    IDLE = '0'
    FORWARD = '1'
    BACKWARD = '2'
    LEFT = '3'
    RIGHT = '4'


class MSP432Uart:
    def __init__(self, port='/dev/ttyAMA0', baudrate=115200, timeout=1):
        """Initializes the serial connection parameters."""
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.data_queue = queue.Queue(maxsize=1) # Counts are cumulative, so only the newest packet is kept
        self._is_logging = False
        self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=self.timeout
            )

    def start_receiving(self): #start the receiving thread to fill up data
        """Attempts to open the serial port."""
        try:
            receive_thread = threading.Thread(target=self.poll_receive, daemon=True)
            receive_thread.start()

            print(f"Serial port {self.port} opened successfully.")
        except Exception as e:
            print(f"Error opening serial port: {e}")
            raise # Re-raise the exception to handle it in the calling code
    
    def stop_logging(self):
        """Call this method to break the loop and safely close the thread."""
        self._is_logging = False

    def send_string(self, message):
        """Encodes a string to bytes and sends it over the serial port."""
        if self.ser and self.ser.is_open:
            try:
                # Strings must be encoded to bytes before sending
                self.ser.write(message.encode('utf-8'))
            except Exception as e:
                print(f"Failed to send data: {e}")
        else:
            print("Serial port is not open. Cannot send data.")

    def send_command(self, left_cps, right_cps):
        """Send signed wheel speeds (encoder counts/sec) as big-endian int16s."""
        if self.ser and self.ser.is_open:
            try:
                header = 0xAA
                # Clamp to int16 so out-of-range values can't wrap and flip direction
                left = max(-32768, min(32767, int(left_cps))) & 0xFFFF
                right = max(-32768, min(32767, int(right_cps))) & 0xFFFF
                l_high, l_low = (left >> 8) & 0xFF, left & 0xFF
                r_high, r_low = (right >> 8) & 0xFF, right & 0xFF
                checksum = l_high ^ l_low ^ r_high ^ r_low
                packet = bytearray([header, l_high, l_low, r_high, r_low, checksum])
                self.ser.write(packet)
            except Exception as e:
                print(f"Failed to send data: {e}")
        else:
            print("Serial port is not open. Cannot send data.")
        

    def poll_receive(self):
        """Continuously polls the serial port for the 10-byte packet."""
        if not (self.ser and self.ser.is_open):
            print("Serial port is not open. Cannot receive data.")
            return

        print("Listening for incoming data...")

        while self.ser.is_open:
            try:
                # Read one byte at a time looking for the start byte; this blocks
                # for up to self.timeout and returns b'' if nothing arrives
                start_byte = self.ser.read(1)

                if start_byte == b'\xAA':
                    # Read the remaining 9 bytes: LCount[31:0], RCount[31:0], checksum
                    payload = self.ser.read(9)

                    if len(payload) < 9:
                        print(f"Short read: got {len(payload)}/9 payload bytes. Waiting for next start byte.")
                        continue

                    # '>' means Big-Endian (High Byte first)
                    # 'i' means signed 32-bit integer (takes 4 bytes each)
                    # 'B' means unsigned 8-bit integer (checksum)
                    l_count, r_count, received_checksum = struct.unpack('>iiB', payload)

                    # The MSP432 XORs the 8 data bytes, so XOR them back the same way
                    expected_checksum = 0
                    for data_byte in payload[:8]:
                        expected_checksum ^= data_byte

                    if received_checksum == expected_checksum:
                        # Drop the stale packet (if any) so the queue never grows
                        try:
                            self.data_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self.data_queue.put_nowait((l_count, r_count))
                    else:
                        print(f"Checksum mismatch (got 0x{received_checksum:02X}, expected 0x{expected_checksum:02X}). Resyncing.")

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