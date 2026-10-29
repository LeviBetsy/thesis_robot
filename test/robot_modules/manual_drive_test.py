import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_path not in sys.path:
    sys.path.append(root_path)

import time
import queue

from app.robot_module.robot import Robot
from app.robot_module.uart import MSP432Uart
from app.control.keyboard_controller_local import LocalKeyboardController
from app.util.config import Config
#********************************************** IMPORTS **********************************************

'''
Bare keyboard drive test: wireless keyboard on the Pi -> UART -> MSP432, nothing else running.
Every command sent is printed with a timestamp, and the measured wheel speeds from the encoders
are printed alongside. A key press that prints a cmd line but leaves the measured cps at 0 means
the packet was lost between the Pi and the MSP432, not in the keyboard.
'''

PRINT_PERIOD = 0.2 #seconds between measured speed prints

#Robot
config = Config.load()
robot = Robot(config)

#UART
msp432_uart = MSP432Uart(port=config.uart.port, baudrate=config.uart.baudrate, timeout=config.uart.timeout)
msp432_uart.start_receiving() #THREAD 1: listen to cumulative encoder counts from MSP432 and fill data_queue

t0 = time.monotonic()
last_logged = [None]

def send_and_log(left_cps, right_cps):
    if (left_cps, right_cps) != last_logged[0]: # the controller re-sends every RESEND_PERIOD, only log changes
        print(f"[{time.monotonic() - t0:7.3f}] cmd   left={left_cps:5d} right={right_cps:5d}")
        last_logged[0] = (left_cps, right_cps)
    msp432_uart.send_command(left_cps, right_cps)

#Keyboard Controller
controller = LocalKeyboardController(send_and_log)
controller.start() #THREAD 2: read arrow keys from the wireless keyboard and forward left_cps,right_cps to msp432

def count_delta(curr, prev):
    # Same int32 wrap handling as pid_test.py
    return ((curr - prev + 32768) % 65536) - 32768

if __name__ == "__main__":
    print("Manual drive test running. Drive with the arrow keys on the Pi's wireless keyboard. Ctrl+C to stop.")
    prev_counts = None
    prev_time = None
    try:
        while True:
            time.sleep(PRINT_PERIOD)
            latest = None
            while True:
                try:
                    latest = msp432_uart.data_queue.get_nowait()
                except queue.Empty:
                    break
            if latest is None:
                print(f"[{time.monotonic() - t0:7.3f}] no encoder data from MSP432")
                continue

            now = time.monotonic()
            if prev_counts is not None:
                dt = now - prev_time
                l_cps = count_delta(latest[0], prev_counts[0]) / dt
                r_cps = count_delta(latest[1], prev_counts[1]) / dt
                print(f"[{now - t0:7.3f}] meas  left={l_cps:7.1f} right={r_cps:7.1f}  (cmd {controller.current_cmd})")
            prev_counts, prev_time = latest, now

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        msp432_uart.send_command(0, 0) # Safety stop
        controller.stop()
        msp432_uart.close()
