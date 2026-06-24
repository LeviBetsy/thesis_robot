# This file is meant to be ran on the ssh-ing device onto the Pi
# Its purpose is to create a listener on the ssh-ing device's keyboard
# And translate those keyboard input into movement command for the Pi
# The script is meant to send the commands over an ssh-tunnel on the 8080 port to the Pi's localhost
# So don't forget to set up that tunnel when you SSH


import socket
import os
import time
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

pressed_keys = set()
current_cmd = (0, 0, 0)

def send_to_pi(inst, left, right):
    global current_cmd
    if (inst, left, right) != current_cmd:
        # We add a newline \n so the Pi knows where one command ends and the next begins
        msg = f"{inst},{left},{right}\n"
        try:
            sock.sendall(msg.encode('utf-8'))
            current_cmd = (inst, left, right)
            print(f"Sent: {msg.strip()}")
        except Exception as e:
            print(f"Send failed: {e}")

def update_logic():
    up = keyboard.Key.up in pressed_keys
    down = keyboard.Key.down in pressed_keys
    left = keyboard.Key.left in pressed_keys
    right = keyboard.Key.right in pressed_keys

    if up:
        if left: send_to_pi(1, 2500, 5000)
        elif right: send_to_pi(1, 5000, 2500)
        else: send_to_pi(1, 5000, 5000)
    elif down:
        if left: send_to_pi(2, 2500, 5000)
        elif right: send_to_pi(2, 5000, 2500)
        else: send_to_pi(2, 5000, 5000)
    elif left: send_to_pi(3, 2500, 2500)
    elif right: send_to_pi(4, 2500, 2500)
    else: send_to_pi(0, 0, 0)

def on_press(key):
    if key in [keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right]:
        pressed_keys.add(key)
        update_logic()

def on_release(key):
    if key in pressed_keys:
        pressed_keys.remove(key)
        update_logic()

if __name__ == "__main__":
    if connect_to_pi():
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()