import serial
import time
from enum import IntEnum

class MotorInstruction(IntEnum):
    IDLE = 0
    FORWARD = 1
    BACKWARD = 2
    LEFT = 3
    RIGHT = 4



SERIAL_PORT = '/dev/ttyAMA0' 
BAUD_RATE = 115200

try:
    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUD_RATE,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1
    )
    print(f"Serial port {SERIAL_PORT} opened successfully.")

    while True:
        user_input = input("Enter motor instruction (0: IDLE, 1: FORWARD, 2: BACKWARD, 3: LEFT, 4: RIGHT): ")
        if user_input == 'exit':
            print("Exiting...")
            break
        try:
            instruction = MotorInstruction(int(user_input))
            ser.write(instruction.to_bytes(1, byteorder='big')) # Send instruction as a single byte
            print(f"Sent instruction: {instruction.name}")
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 4, or 'exit' to quit.")

except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
except KeyboardInterrupt:
    print("Exiting...")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print(f"Serial port {SERIAL_PORT} closed.")