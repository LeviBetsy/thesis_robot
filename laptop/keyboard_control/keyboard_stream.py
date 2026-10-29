# This file is meant to be ran on the ssh-ing device onto the Pi
# Its purpose is to create a listener on the ssh-ing device's keyboard
# And translate those keyboard input into movement command for the Pi
# The script is meant to send the commands over an ssh-tunnel on the 8080 port to the Pi's localhost
# So don't forget to set up that tunnel when you SSH


import socket
import os
import time
import threading
from dotenv import load_dotenv
from pynput import keyboard

load_dotenv()
PI_IP_ADDRESS = os.getenv("PI_IP_ADDRESS")
PORT = int(os.getenv("TCP_PORT", 8080)) # We'll reuse the env var name or change to TCP_PORT

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def connect_to_pi():
    try:
        sock.connect((PI_IP_ADDRESS, PORT))
        print(f"Successfully connected to {PI_IP_ADDRESS}")
        return True
    except Exception as e:
        print(f"Failed to connect: {e}")
        return False

# Target wheel speeds in encoder counts per second (signed: negative = reverse)
FULL_CPS = 360
HALF_CPS = 180

# The current command is re-sent this often (seconds) as a heartbeat. The Pi's RobotController
# stops the robot when it hears nothing for COMMAND_TIMEOUT, so this must stay well below that
RESEND_PERIOD = 0.1

pressed_keys = set()
current_cmd = (0, 0)
send_lock = threading.Lock() # the key listener and the heartbeat thread share the socket

def send_to_pi(left_cps, right_cps):
    global current_cmd
    current_cmd = (left_cps, right_cps)
    # We add a newline \n so the Pi knows where one command ends and the next begins
    msg = f"{left_cps},{right_cps}\n"
    with send_lock:
        try:
            sock.sendall(msg.encode('utf-8'))
            # print(f"Sent: {msg.strip()}")
        except Exception as e:
            print(f"Send failed: {e}")

def update_logic():
    up = keyboard.Key.up in pressed_keys
    down = keyboard.Key.down in pressed_keys
    left = keyboard.Key.left in pressed_keys
    right = keyboard.Key.right in pressed_keys

    if up:
        if left: send_to_pi(HALF_CPS, FULL_CPS)
        elif right: send_to_pi(FULL_CPS, HALF_CPS)
        else: send_to_pi(FULL_CPS, FULL_CPS)
    elif down:
        if left: send_to_pi(-HALF_CPS, -FULL_CPS)
        elif right: send_to_pi(-FULL_CPS, -HALF_CPS)
        else: send_to_pi(-FULL_CPS, -FULL_CPS)
    elif left: send_to_pi(-HALF_CPS, HALF_CPS)
    elif right: send_to_pi(HALF_CPS, -HALF_CPS)
    else: send_to_pi(0, 0)

def on_press(key):
    if key in [keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right]:
        pressed_keys.add(key)
        update_logic()

def on_release(key):
    if key in pressed_keys:
        pressed_keys.remove(key)
        update_logic()

def heartbeat():
    while True:
        time.sleep(RESEND_PERIOD)
        send_to_pi(*current_cmd)

if __name__ == "__main__":
    if connect_to_pi():
        threading.Thread(target=heartbeat, daemon=True).start()
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()