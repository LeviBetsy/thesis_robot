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


import threading
from evdev import InputDevice, list_devices, ecodes

# Target wheel speeds in encoder counts per second, same as keyboard_stream.py
FULL_CPS = 360
HALF_CPS = 180

ARROW_KEYS = {ecodes.KEY_UP, ecodes.KEY_DOWN, ecodes.KEY_LEFT, ecodes.KEY_RIGHT}

# evdev key event values
KEY_UP_EVENT = 0
KEY_DOWN_EVENT = 1
KEY_HOLD_EVENT = 2


def find_keyboard():
    """Return the first input device that has all four arrow keys, or None."""
    for path in list_devices():
        dev = InputDevice(path)
        keys = dev.capabilities().get(ecodes.EV_KEY, [])
        if ARROW_KEYS.issubset(keys):
            return dev
        dev.close()
    return None


class LocalKeyboardController:
    def __init__(self, command_callback, device_path=None, grab=True):
        self.command_callback = command_callback
        self.device = InputDevice(device_path) if device_path else find_keyboard()
        if self.device is None:
            raise RuntimeError("No keyboard with arrow keys found in /dev/input, is the USB receiver plugged in?")
        if grab:
            self.device.grab() # exclusive, so arrow presses don't also reach the Pi's console
        self.pressed_keys = set()
        self.current_cmd = (0, 0)
        self.running = True

    def start(self):
        keyboard_thread = threading.Thread(target=self.listen, daemon=True)
        keyboard_thread.start()

    def listen(self):
        print(f"Listening for arrow keys on {self.device.path} ({self.device.name})...")
        try:
            for event in self.device.read_loop():
                if not self.running:
                    break
                if event.type != ecodes.EV_KEY or event.code not in ARROW_KEYS:
                    continue
                if event.value == KEY_DOWN_EVENT:
                    self.pressed_keys.add(event.code)
                elif event.value == KEY_UP_EVENT:
                    self.pressed_keys.discard(event.code)
                else:
                    continue # KEY_HOLD_EVENT auto-repeat, command already sent
                self.update_logic()
        except OSError as e:
            if self.running:
                print(f"Keyboard lost or error: {e}")
        self.pressed_keys.clear()
        self.send(0, 0) # Safety stop

    def send(self, left_cps, right_cps):
        if (left_cps, right_cps) != self.current_cmd:
            self.command_callback(left_cps, right_cps)
            self.current_cmd = (left_cps, right_cps)

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
        try:
            self.device.ungrab()
        except OSError:
            pass
        self.device.close()
