# """
# keyboard_controller_local.py

# Overview:
# Local counterpart of keyboard_controller_ssh.py. Instead of receiving
# "left_cps,right_cps" over a TCP tunnel from the laptop, it reads the arrow keys
# of a wireless keyboard plugged into the Pi (USB receiver) directly through evdev,
# applies the same arrow key -> wheel speed mapping as laptop/keyboard_control/keyboard_stream.py,
# and hands the speeds to a callback.

# Technical Details:
# - Input: Linux evdev (/dev/input/event*). Works headless, no X server or ssh session needed.
#   The user running this needs read access to /dev/input (be in the 'input' group, or sudo).
# - Device: auto-detected as the first input device exposing the arrow keys, or pass device_path.
# - Concurrency: the read loop runs in a daemonized background thread.
# - Data Format: callback(left_cps, right_cps), signed encoder counts per second (negative = reverse).
#   Only called when the command changes.

# Usage:
# 1. controller = LocalKeyboardController(command_callback=msp432_uart.send_command)
# 2. controller.start()
# 3. If the receiver disconnects or a read fails, the controller triggers a safety stop (0,0).
# """


import select
import threading
import time
from evdev import InputDevice, list_devices, ecodes

# Target wheel speeds in encoder counts per second, same as keyboard_stream.py
FULL_CPS = 360
HALF_CPS = 180

# The current command is re-sent this often (seconds) even when no key changes, so a single
# packet the MSP432 drops or fails the checksum on doesn't leave the robot ignoring a press
RESEND_PERIOD = 0.1

# How often (seconds) the keyboard nodes are checked for still existing. An undervoltage USB
# reset removes them, and a removed node can go quiet instead of raising on read
DEVICE_CHECK_PERIOD = 0.5

ARROW_KEYS = {ecodes.KEY_UP, ecodes.KEY_DOWN, ecodes.KEY_LEFT, ecodes.KEY_RIGHT}

# evdev key event values
KEY_UP_EVENT = 0
KEY_DOWN_EVENT = 1
KEY_HOLD_EVENT = 2


def find_keyboards():
    """Return every input device that has all four arrow keys. A wireless receiver exposes
    several nodes (keyboard, mouse, consumer control) and more than one can claim arrow keys,
    so the controller listens on all of them instead of guessing."""
    devices = []
    for path in list_devices():
        dev = InputDevice(path)
        keys = dev.capabilities().get(ecodes.EV_KEY, [])
        if ARROW_KEYS.issubset(keys):
            devices.append(dev)
        else:
            dev.close()
    return devices


class LocalKeyboardController:
    def __init__(self, command_callback, device_path=None, grab=True):
        self.command_callback = command_callback
        self.device_path = device_path
        self.grab_devices = grab
        self.devices = self.open_devices()
        if not self.devices:
            raise RuntimeError("No keyboard with arrow keys found in /dev/input, is the USB receiver plugged in?")
        self.pressed_keys = set()
        self.current_cmd = (0, 0)
        self.last_send_time = 0.0
        self.running = True

    def start(self):
        keyboard_thread = threading.Thread(target=self.listen, daemon=True)
        keyboard_thread.start()

    def open_devices(self):
        """Open (and grab) the keyboard nodes, [] if none are present right now."""
        try:
            devices = [InputDevice(self.device_path)] if self.device_path else find_keyboards()
            if self.grab_devices:
                for dev in devices:
                    dev.grab() # exclusive, so arrow presses don't also reach the Pi's console
        except OSError:
            return [] # node vanished between listing and opening, mid re-enumeration
        return devices

    def listen(self):
        while self.running:
            try:
                self.read_events()
            except OSError as e:
                if self.running:
                    print(f"Keyboard lost ({e}), robot stopped, waiting for the receiver to come back...")
            self.pressed_keys.clear()
            self.send(0, 0) # Safety stop
            if self.running:
                self.reconnect()

    def reconnect(self):
        """Hold the robot stopped and rescan until the keyboard nodes are back."""
        self.close_devices()
        found_before = False
        while self.running:
            time.sleep(DEVICE_CHECK_PERIOD)
            self.command_callback(0, 0) # heartbeat stop while there is no keyboard
            self.last_send_time = time.monotonic()
            devices = self.open_devices()
            if devices and found_before:
                self.devices = devices
                print("Keyboard reconnected")
                return
            # The receiver brings its nodes up over a few hundred ms, so take the second scan that sees them
            found_before = bool(devices)
            for dev in devices:
                dev.close()

    def read_events(self):
        """Handle arrow keys until stop(). Raises OSError when a keyboard node is lost."""
        names = ", ".join(f"{dev.path} ({dev.name})" for dev in self.devices)
        print(f"Listening for arrow keys on {names}...")
        fd_to_dev = {dev.fd: dev for dev in self.devices}
        last_check = time.monotonic()
        while self.running:
            # timeout drives the resend and device check, and lets stop() be noticed
            ready, _, _ = select.select(fd_to_dev, [], [], min(RESEND_PERIOD, DEVICE_CHECK_PERIOD))
            now = time.monotonic()
            if now - self.last_send_time >= RESEND_PERIOD:
                self.command_callback(*self.current_cmd) # heartbeat, same command again
                self.last_send_time = now
            if now - last_check >= DEVICE_CHECK_PERIOD:
                for dev in self.devices:
                    dev.active_keys() # ioctl, raises ENODEV on a removed node even when read() stays quiet
                last_check = now
            for fd in ready:
                for event in fd_to_dev[fd].read():
                    if event.type != ecodes.EV_KEY or event.code not in ARROW_KEYS:
                        continue
                    if event.value == KEY_DOWN_EVENT:
                        self.pressed_keys.add(event.code)
                    elif event.value == KEY_UP_EVENT:
                        self.pressed_keys.discard(event.code)
                    else:
                        continue # KEY_HOLD_EVENT auto-repeat, command already sent
                    self.update_logic()

    def send(self, left_cps, right_cps):
        if (left_cps, right_cps) != self.current_cmd:
            self.command_callback(left_cps, right_cps)
            self.current_cmd = (left_cps, right_cps)
            self.last_send_time = time.monotonic()

    def update_logic(self):
        up = ecodes.KEY_UP in self.pressed_keys
        down = ecodes.KEY_DOWN in self.pressed_keys
        left = ecodes.KEY_LEFT in self.pressed_keys
        right = ecodes.KEY_RIGHT in self.pressed_keys

        if up:
            if left: self.send(HALF_CPS, FULL_CPS)
            elif right: self.send(FULL_CPS, HALF_CPS)
            else: self.send(FULL_CPS, FULL_CPS)
        elif down:
            if left: self.send(-HALF_CPS, -FULL_CPS)
            elif right: self.send(-FULL_CPS, -HALF_CPS)
            else: self.send(-FULL_CPS, -FULL_CPS)
        elif left: self.send(-HALF_CPS, HALF_CPS)
        elif right: self.send(HALF_CPS, -HALF_CPS)
        else: self.send(0, 0)

    def stop(self):
        self.running = False
        self.close_devices()

    def close_devices(self):
        for dev in self.devices:
            try:
                dev.ungrab()
            except OSError:
                pass
            dev.close()


if __name__ == "__main__":
    # Key presses are printed by the [DEBUG] line in listen(), the callback prints the resulting wheel speeds
    controller = LocalKeyboardController(command_callback=lambda l, r: print(f"cmd left_cps={l} right_cps={r}"))
    controller.start()
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("Stopping keyboard controller")
    finally:
        controller.stop()
