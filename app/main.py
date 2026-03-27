import cv2
from ultralytics import YOLO
from detection_filter import *
import ultralytics

#TODO: there has to be a better way to load models
# Load a YOLO26n PyTorch model
model = YOLO("yolo26n.pt")
model.export(format="ncnn")  # creates 'yolo26n_ncnn_model'
ncnn_model = YOLO("yolo26n_ncnn_model", task = "detect") # Load the exported NCNN model for embedded devices
class_id_mouse = find_class_id(ncnn_model, "mouse") #mouse should be 64
print(class_id_mouse)
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

    #test find location of mouse
    box_mouse = get_bounds(results, class_id_mouse) 

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow("Camera", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# cam.release()  # Release the camera resource
# # Release resources and close windows
# cv2.destroyAllWindows()