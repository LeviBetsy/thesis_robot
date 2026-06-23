import sys
import os

# Adds the project root (two levels up from this file) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from app.module.uart import *
from app.control.keyboard_controller_ssh import *

msp432_uart = MSP432Uart()

msp432_uart.start_receiving() #start thread to listen to odometry data from MSP432 and fill buffer

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