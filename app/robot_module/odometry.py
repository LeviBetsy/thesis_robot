import threading
import math
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.robot_module.uart import MSP432Uart
from app.robot_module.robot import Robot
from app.util.config import Config

'''
    Pose-increment accumulators for the motion stage of MCL.

    Odometry is the base: it holds the running increment, the lock guarding it, and the
    body-frame composition that folds each small arc into that increment. A subclass only
    has to turn ITS OWN input into (d, delta_theta) and hand it to accumulate() -
    tachometer packets here, drive commands in a simulated rover, replayed log lines in a
    playback harness. The composition, which is the part that is easy to get wrong, is
    written once.

    The base is instantiable and inert on its own: never fed, it consumes zeros. That is
    what MCLLocalization uses when no real source is wired up, so there is no special case
    for "no odometry" anywhere in the filter.

    FRAME: the accumulated (dx, dy, dtheta) is expressed in the robot frame as it was at
    the LAST consume() - x along the heading the robot had then, y to its left. That is
    what MCLLocalization.motion_update expects (Thrun's barred deltas with theta_bar
    already taken out), so no absolute pose is needed in here.
'''


class Odometry:
    """Accumulates small arcs into one body-frame pose increment, read-and-clear."""

    def __init__(self):
        #increment since the last consume(), in the body frame of that consume()
        self.dx = 0.0
        self.dy = 0.0
        self.dtheta = 0.0

        # A producer thread writes these while the consumer (MCL, on the ZMQ receiver
        # thread) reads them, and a read is read-then-zero, so it has to be atomic against
        # a concurrent arc or that arc's motion is silently dropped.
        self.lock = threading.Lock()

    # ****** ACCUMULATION ******

    def accumulate(self, d, delta_theta):
        """
        Folds one small arc into the running increment.

        Args:
            d (float): distance travelled by the robot's midpoint over this arc, meters.
            delta_theta (float): heading change over this arc, radians.
        """
        with self.lock:
            # The arc is laid down along the heading the robot has reached SO FAR within
            # this window (self.dtheta), not along the window's starting heading,
            # otherwise a turn-then-drive would come out pointing the wrong way. The
            # +delta_theta/2 is the midpoint rule: over this arc the robot sweeps
            # delta_theta, so its average heading is half way through it. Exact for the
            # chord's DIRECTION; the chord/arc length mismatch is what is approximated,
            # and at ~0.3 deg per packet that is a relative error of 1e-6.
            heading = self.dtheta + delta_theta / 2
            self.dx += d * math.cos(heading)
            self.dy += d * math.sin(heading)
            self.dtheta += delta_theta #left unwrapped, it is a relative change not a pose

    # ****** READOUT ******

    def consume(self) -> tuple:
        """
        Returns the increment accumulated since the last call AND clears it, in one
        atomic step, so the caller owns that motion and no arc is applied twice.

        Returns:
            (dx, dy, dtheta): meters and radians, in the body frame of the previous call.
        """
        with self.lock:
            increment = (self.dx, self.dy, self.dtheta)
            self.dx = self.dy = self.dtheta = 0.0
            return increment

    def peek(self) -> tuple:
        """Same reading as consume() but WITHOUT clearing, for logging or display. Never
        feed this to the filter: the same motion would be applied again on the next
        consume()."""
        with self.lock:
            return self.dx, self.dy, self.dtheta

    def clear(self):
        """Throws the accumulated increment away. MCLLocalization calls this after
        re-seeding the particle cloud, where motion from before the reset means nothing."""
        with self.lock:
            self.dx = self.dy = self.dtheta = 0.0


class WheelOdometry(Odometry):
    """
    Odometry fed by the MSP432's tachometer packets.

    The MSP432 pushes encoder deltas over UART far faster than the camera produces scans
    (3 fps), so every packet is folded into the running increment and the whole thing is
    handed over when MCL asks.
    """

    def __init__(self, robot: Robot):
        super().__init__()
        self.robot = robot
        self.odometry_thread = None
        self._running = False

    def init_odometry_thread(self, msp432_uart: MSP432Uart):
        """Starts the thread that drains the UART queue into the accumulator."""
        def odom_loop():
            while self._running:
                #blocks until the MSP432 sends a packet
                Lcount, Rcount = msp432_uart.get_data()
                self.update_odom_coordinate(Lcount, Rcount)

        self._running = True
        self.odometry_thread = threading.Thread(target=odom_loop, daemon=True)
        self.odometry_thread.start()

    def stop(self):
        """Asks the thread to exit. It is parked in a blocking get_data(), so it only
        notices on the next packet; daemon=True means it dies with the process anyway."""
        self._running = False

    def update_odom_coordinate(self, LCount, RCount):
        """
        Converts one tachometer packet into an arc and accumulates it.

        Args:
            LCount (int): left encoder slots since the last packet (signed).
            RCount (int): right encoder slots, same.
        """
        dl = LCount * self.robot.c / self.robot.n
        dr = RCount * self.robot.c / self.robot.n
        d = (dl + dr) / 2 #distance travelled by the midpoint
        delta_theta = (dr - dl) / self.robot.w
        self.accumulate(d, delta_theta)


if __name__ == "__main__":
    config = Config.load()
    robot = Robot(config)
    odom = WheelOdometry(robot)

    uart = MSP432Uart(port=config.uart.port,
                      baudrate=config.uart.baudrate,
                      timeout=config.uart.timeout)
    uart.start_receiving()
    odom.init_odometry_thread(uart)

    print("Drive the robot. Printing the increment once a second, consuming it each time.")
    try:
        while True:
            threading.Event().wait(1.0)
            dx, dy, dtheta = odom.consume()
            print(f"dx={dx:+.4f} m  dy={dy:+.4f} m  dtheta={math.degrees(dtheta):+.2f} deg")
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        odom.stop()
        uart.close()
