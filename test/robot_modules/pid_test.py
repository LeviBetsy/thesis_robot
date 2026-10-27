import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_path not in sys.path:
    sys.path.append(root_path)

import time
import queue

from app.robot_module.robot import Robot
from app.robot_module.uart import MSP432Uart
from app.control.keyboard_controller_ssh import RobotController
from app.util.config import Config
#********************************************** IMPORTS **********************************************

#Robot
config = Config.load()
robot = Robot(config)

#UART
msp432_uart = MSP432Uart(port=config.uart.port, baudrate=config.uart.baudrate, timeout=config.uart.timeout)
msp432_uart.start_receiving() #THREAD 1: listen to encoder counts from MSP432 and fill data_queue

#Keyboard Controller
controller = RobotController(msp432_uart.send_command)
controller.start() #THREAD 2: listen for left_cps,right_cps from the laptop and forward to msp432

def count_delta(curr, prev):
    # Cumulative counts are int32 from the MSP432. Folding the difference into +/-32768 is exact
    # for any real per-sample delta, and also absorbs the counter's int32 wrap (2^32 is a multiple of 65536)
    return ((curr - prev + 32768) % 65536) - 32768

if __name__ == "__main__":
    print("PID test running. Drive with keyboard_stream.py on the laptop. Ctrl+C to stop.")
    prev_counts = None
    prev_time = None
    max_l_cps = 0.0
    max_r_cps = 0.0
    try:
        while True:
            time.sleep(1.0)

            # Counts are already accumulated by the MSP432, so only the newest packet matters
            latest = None
            while True:
                try:
                    latest = msp432_uart.data_queue.get_nowait()
                except queue.Empty:
                    break

            if latest is None:
                print("No encoder data received in the last second")
                continue

            now = time.monotonic()
            if prev_counts is not None:
                dt = now - prev_time
                l_cps = count_delta(latest[0], prev_counts[0]) / dt
                r_cps = count_delta(latest[1], prev_counts[1]) / dt
                max_l_cps = max(max_l_cps, abs(l_cps))
                max_r_cps = max(max_r_cps, abs(r_cps))
                print(f"LCount: {latest[0]:6d}  RCount: {latest[1]:6d}  "
                      f"L_CPS: {l_cps:7.1f}  R_CPS: {r_cps:7.1f}  "
                      f"(max L: {max_l_cps:.1f}  R: {max_r_cps:.1f})")
            prev_counts = latest
            prev_time = now

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        msp432_uart.send_command(0, 0) # Safety stop
        controller.stop()
        msp432_uart.close()
