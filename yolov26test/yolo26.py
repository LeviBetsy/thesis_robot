import cv2
from ultralytics import YOLO

# Load a YOLO26n PyTorch model
model = YOLO("yolo26n.pt")

# Export the model to NCNN format
model.export(format="ncnn")  # creates 'yolo26n_ncnn_model'

# Load the exported NCNN model
ncnn_model = YOLO("yolo26n_ncnn_model")

# Initialize USB camera
cam = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, image = cam.read()

    if not ret:
        print("Failed to grab frame")
        break  # or continue / handle error

    # Run YOLO26 inference on the frame
    results = ncnn_model(image)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow("Camera", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release resources and close windows
cv2.destroyAllWindows()