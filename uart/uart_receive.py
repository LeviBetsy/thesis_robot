import serial
import time


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
        if ser.in_waiting > 0:
            # data = ser.read(1).decode('utf-8') #this is for receiving string data
            data = ser.read(1) #Read one byte at a time
            number = data[0] # Get the integer value of the byte
            print(number, end='', flush=True) #Print received data immediately
        time.sleep(0.01)

except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
except KeyboardInterrupt:
    print("Exiting...")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print(f"Serial port {SERIAL_PORT} closed.")