from app.uart.uart_exchange import *


msp432_uart = MSP432Uart()
msp432_uart.connect()

logging_thread = threading.Thread(target=msp432_uart.log_tach, daemon=True)

# 3. Start the thread
logging_thread.start()

# 4. Main Program Loop
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

        msp432_uart.send_string(cmd)
        time.sleep(1)  # Keep the program running to maintain the serial connection
        

except KeyboardInterrupt:
    print("\nCtrl+C detected. Shutting down cleanly...")
    
    # 5. Safely stop the logging loop
    msp432_uart.stop_logging()
    
    # 6. Wait for the logging thread to finish its final write cycle
    logging_thread.join(timeout=2)
    msp432_uart.close()
    print("Program terminated.")