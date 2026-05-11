import sys
import os

# Adds the project root (two levels up from this file) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.uart.uart_exchange import *
from app.localization.localization import *
from app.control.keyboard_controller_ssh import *
import math

msp432_uart = MSP432Uart()

msp432_uart.start_receiving() #start thread to listen to odometry data from MSP432 and fill buffer
loc = Localization(width=23, length=37, cell_size=50) #23 columns, 37 rows, cell size is 50x50mm
loc.stream_occupancy_grid() #start thread to stream odometry data to Flask Server
loc.init_odometry_thread(msp432_uart) #start thread to change localization using UART buffer

controller = RobotController(msp432_uart.send_command)
controller.start() #start thread to listen for keyboard

if __name__ == "__main__":
    print("Main program is running. Initializing keyboard controller...")
    try:
        while True:
            time.sleep(0.1) # Small sleep to prevent the while loop from maxing out the Pi's CPU
            
    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C
        print("\nShutting down...")
        msp432_uart.send_command(0, 0, 0)