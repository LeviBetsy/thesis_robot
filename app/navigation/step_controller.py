import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import math
import time

from app.robot_module.uart import MSP432Uart
from app.robot_module.odometry import WheelOdometry
from app.util.config import Config

'''
Executes one nav step on the encoders: turn in place by dtheta, or drive straight by d.

MCL decides the step (once, from the mean pose), this carries it out. MCL is too slow and
stale to close a turn on (~35 deg of spin per scan at turn_cps), the encoders are not:
each call clears odometry.segment and polls it until the target is reached. The main
accumulator MCL consume()s is never touched.

Sign convention matches world theta and the keyboard controllers: positive dtheta is CCW,
i.e. left wheel back, right wheel forward.
'''

POLL_PERIOD = 0.005 #seconds between segment reads, encoder packets arrive faster than this


class StepController:
    def __init__(self, msp432_uart: MSP432Uart, odometry: WheelOdometry, config: Config):
        self.uart = msp432_uart
        self.segment = odometry.segment
        self.nav = config.nav
        self.current_cmd = None

    def send(self, left_cps, right_cps):
        """Writes to the MSP432 only when the command changes, the loops below poll far
        faster than the UART needs to hear about it."""
        if (left_cps, right_cps) != self.current_cmd:
            self.uart.send_command(left_cps, right_cps)
            self.current_cmd = (left_cps, right_cps)

    def stop(self):
        self.send(0, 0)

    def rotate_by(self, dtheta):
        """
        Turns in place by dtheta, the short way round, closed on the encoders.

        Args:
            dtheta (float): radians, CCW positive. Wrapped to [-pi, pi] first.
        """
        dtheta = math.atan2(math.sin(dtheta), math.cos(dtheta))
        sign = 1 if dtheta > 0 else -1
        target = abs(dtheta) - self.nav.turn_brake #send stop early, the robot coasts the rest

        self.segment.clear()
        while True:
            turned = sign * self.segment.peek()[2] #progress in the commanded direction
            remaining = target - turned
            if remaining <= 0:
                break
            cps = self.nav.turn_cps if remaining > self.nav.turn_slow_window else self.nav.turn_slow_cps
            self.send(-sign * cps, sign * cps)
            time.sleep(POLL_PERIOD)
        self.stop()

    def drive_by(self, d):
        """
        Drives straight forward by d, closed on the encoders.

        Args:
            d (float): meters, forward only.
        """
        target = d - self.nav.drive_brake #send stop early, the robot coasts the rest

        self.segment.clear()
        while self.segment.peek()[0] < target: #dx, along the heading the drive started on
            self.send(self.nav.drive_cps, self.nav.drive_cps)
            time.sleep(POLL_PERIOD)
        self.stop()
