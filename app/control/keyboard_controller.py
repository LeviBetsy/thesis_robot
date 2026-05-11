from pynput import keyboard
import threading

class RobotController:
    def __init__(self, command_callback):
        self.command_callback = command_callback
        self.pressed_keys = set()
        self.current_command = (0, 0, 0)  # (INST, LEFT, RIGHT)
        self.lock = threading.Lock()

        # Instruction mapping based on your 0-4 scheme
        self.STOP = 0
        self.FORWARD = 1
        self.BACKWARD = 2
        self.LEFT = 3
        self.RIGHT = 4

    def update_command(self):
        with self.lock:
            # Default to STOP if no recognized keys are pressed
            new_command = (self.STOP, 0, 0)

            # Determine command based on currently pressed keys
            if keyboard.Key.up in self.pressed_keys:
                if keyboard.Key.left in self.pressed_keys:
                    new_command = (self.FORWARD, 2500, 5000)
                elif keyboard.Key.right in self.pressed_keys:
                    new_command = (self.FORWARD, 5000, 2500)
                else:
                    new_command = (self.FORWARD, 5000, 5000)
            elif keyboard.Key.down in self.pressed_keys:
                new_command = (self.BACKWARD, 5000, 5000)
            elif keyboard.Key.left in self.pressed_keys:
                new_command = (self.LEFT, 5000, 5000)
            elif keyboard.Key.right in self.pressed_keys:
                new_command = (self.RIGHT, 5000, 5000)

            # Only send the command to UART if it actually changed
            # This prevents spamming the MSP432 with identical commands
            if new_command != self.current_command:
                self.current_command = new_command
                self.command_callback(*new_command)

    def on_press(self, key):
        if key in [keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right]:
            self.pressed_keys.add(key)
            self.update_command()

    def on_release(self, key):
        if key in self.pressed_keys:
            self.pressed_keys.remove(key)
            self.update_command()

    def start(self):
        print("Keyboard listener started. Use Arrow Keys to move.")
        # This listener blocks, which is why we will call start() in a thread
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()