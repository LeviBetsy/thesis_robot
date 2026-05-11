import keyboard
import threading
import time

class RobotController:
    def __init__(self, command_callback):
        self.command_callback = command_callback
        self.current_command = (0, 0, 0)
        
        # Instruction mapping
        self.STOP = 0
        self.FORWARD = 1
        self.BACKWARD = 2
        self.LEFT = 3
        self.RIGHT = 4

    def get_current_command(self):
        # Default to stop
        inst, left, right = self.STOP, 0, 0

        # Check key states
        up = keyboard.is_pressed('up')
        down = keyboard.is_pressed('down')
        left_key = keyboard.is_pressed('left')
        right_key = keyboard.is_pressed('right')

        if up:
            inst = self.FORWARD
            if left_key:
                left, right = 2500, 5000
            elif right_key:
                left, right = 5000, 2500
            else:
                left, right = 5000, 5000
        elif down:
            inst = self.BACKWARD
            left, right = 5000, 5000
        elif left_key:
            inst = self.LEFT
            left, right = 5000, 5000
        elif right_key:
            inst = self.RIGHT
            left, right = 5000, 5000

        return (inst, left, right)

    def start(self):
        print("Keyboard listener started. Use Arrow Keys to move.")
        try:
            while True:
                new_command = self.get_current_command()
                
                # Only call the UART method if the command state changes
                if new_command != self.current_command:
                    self.current_command = new_command
                    self.command_callback(*new_command)
                
                # Polling rate - 50ms is plenty responsive
                time.sleep(0.05)
        except Exception as e:
            print(f"Controller Error: {e}")
            self.command_callback(0, 0, 0)