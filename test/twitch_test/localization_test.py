from app.uart.uart_exchange import *
from app.localization.localization import *
import math

msp432_uart = MSP432Uart()
msp432_uart.connect()

loc = Localization(width=214, length=356)
loc.stream_occupancy_grid()

loc.init_odometry_thread(msp432_uart)


try:
    print("Main program is running. Doing other tasks...")
    while True:
        # The main thread handles other robot operations here
        # The UART logging is happening simultaneously in the background
        print("Enter a command to send (enter in the format INST,LEFTDUTY,RIGHTCYCLE ): ")
        print("INST: 0-4 for STOP, FORWARD, BACKWARD, lEFT, RIGHT")
        print("DUTYCYCLE from 0-14998")
        cmd = input()
        parts = cmd.split(',')
        msp432_uart.send_command(int(parts[0]), int(parts[1]), int(parts[2]))

        time.sleep(2)  # Keep the program running to maintain the serial connection
        msp432_uart.send_command(0, 0, 0)
        

except KeyboardInterrupt:
    print("\nCtrl+C detected. Shutting down cleanly...")
    msp432_uart.close()
    print("Program terminated.")