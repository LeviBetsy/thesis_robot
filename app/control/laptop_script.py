import socket
from pynput import keyboard

# CONFIGURATION
PI_IP_ADDRESS = "192.168.1.XX" # <--- Put your Pi's IP here
PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
pressed_keys = set()
current_cmd = (0, 0, 0)

def send_to_pi(inst, left, right):
    global current_cmd
    if (inst, left, right) != current_cmd:
        msg = f"{inst},{left},{right}"
        sock.sendto(msg.encode('utf-8'), (PI_IP_ADDRESS, PORT))
        current_cmd = (inst, left, right)
        print(f"Sent: {msg}")

def update():
    # Instructions: 0-STOP, 1-FORWARD, 2-BACKWARD, 3-LEFT, 4-RIGHT
    up = keyboard.Key.up in pressed_keys
    down = keyboard.Key.down in pressed_keys
    left = keyboard.Key.left in pressed_keys
    right = keyboard.Key.right in pressed_keys

    if up:
        if left: send_to_pi(1, 2500, 5000)
        elif right: send_to_pi(1, 5000, 2500)
        else: send_to_pi(1, 5000, 5000)
    elif down: send_to_pi(2, 5000, 5000)
    elif left: send_to_pi(3, 5000, 5000)
    elif right: send_to_pi(4, 5000, 5000)
    else: send_to_pi(0, 0, 0) # Release all = STOP

def on_press(key):
    if key in [keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right]:
        pressed_keys.add(key)
        update()

def on_release(key):
    if key in pressed_keys:
        pressed_keys.remove(key)
        update()

print(f"Controller active. Sending commands to {PI_IP_ADDRESS}")
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()