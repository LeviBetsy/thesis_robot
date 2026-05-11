from app.uart.uart_exchange import *
from app.localization.localization import *
from app.control.keyboard_controller_ssh import *
import math

msp432_uart = MSP432Uart()

# msp432_uart.start_receiving() #start thread to listen to odometry data from MSP432 and fill buffer
# loc = Localization(width=214, length=356)
# loc.stream_occupancy_grid() #start thread to stream odometry data to Flask Server
# loc.init_odometry_thread(msp432_uart) #start thread to change localization using UART buffer


if __name__ == "__main__":
    print("Main program is running. Initializing keyboard controller...")
    
    # Initialize the controller and pass in the function it should call
    controller = RobotController(msp432_uart.send_command)
    
    # Run the keyboard listener in a background thread
    # daemon=True ensures the thread dies when you exit the main program
    keyboard_thread = threading.Thread(target=controller.start, daemon=True)
    keyboard_thread.start()

    try:
        while True:
            # The main thread handles other robot operations here
            # The UART logging and keyboard control are happening simultaneously in the background
            
            time.sleep(0.1) # Small sleep to prevent the while loop from maxing out the Pi's CPU
            
    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C
        print("\nShutting down...")
        msp432_uart.send_command(0, 0, 0)