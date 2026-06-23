import cv2
from flask import Flask, render_template, Response, jsonify
import os

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# --- HARDCODED CONFIGURATION ---
SAVE_DIR = 'captures'
current_img_index = 51  # Change this number whenever you want to start from a new offset
# -------------------------------

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    global current_img_index
    success, frame = camera.read()
    if success:
        filename = f"{SAVE_DIR}/{current_img_index}.jpg"
        cv2.imwrite(filename, frame)
        
        print(f"Captured: {filename}")
        current_img_index += 1
        
        return jsonify({"status": "success", "filename": filename})
    
    return jsonify({"status": "error"}), 500

if __name__ == '__main__':
    # host='0.0.0.0' allows access from your laptop on the same network
    app.run(host='0.0.0.0', port=5000, debug=False)