import cv2
from ultralytics import YOLO

# 1. Load the ONNX model directly
# Providing the 'task' argument is good practice when using exported formats
model = YOLO("./app/models/best.onnx", task="detect")
# model = YOLO("./app/models/best.pt")

# 2. Initialize USB camera
cam = cv2.VideoCapture(0)

# Optional: Set resolution to improve FPS if needed
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, image = cam.read()
    if not ret:
        break

    # 3. Run inference 
    # The library automatically uses the ONNX backend here
    results = model(image, conf=0.25) 
    
    # 4. Visualize and Display
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv26 ONNX Inference", annotated_frame)

    if cv2.waitKey(1) == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()