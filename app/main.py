import cv2
from ultralytics import YOLO
from detection_filter import *
from uart_exchange import SerialManager, Instruction_t
import ultralytics
import math

#TODO: there has to be a better way to load models
# Load a YOLO26n PyTorch model
# model = YOLO("yolo26n.pt")
# model.export(format="ncnn")  # creates 'yolo26n_ncnn_model'
ncnn_model = YOLO("yolo26n_ncnn_model", task = "detect") # Load the exported NCNN model for embedded devices
class_id_mouse = find_class_id(ncnn_model, "mouse") #mouse should be 64
print(class_id_mouse)
# Initialize USB camera
cam = cv2.VideoCapture(0)


#UART
cmd = Instruction_t.IDLE
communicator = SerialManager(port='/dev/ttyAMA0', baudrate=115200)
communicator.connect()
while True:
    # Capture frame-by-frame
    ret, image = cam.read()

    if not ret:
        print("Failed to grab frame")
        break  # or continue / handle error

    # Run YOLO26 inference on the frame
    results = ncnn_model(image)

    #test find location of mouse
    pixel_width = 0
    box_mouse = get_bounds(results, class_id_mouse) 
    if (box_mouse):
        pixel_width = abs((box_mouse[0][0] - box_mouse[0][2]))
        # print(f"width of mouse in pixel {pixel_width}")
        # dist = (391* 11)/ pixel_width

        dist = -0.227*pixel_width + 64.7
        # print(f"distance using ratio = {dist}cm")

        #if you see a mouse and it's far away go forward
        if(dist > 20):
            #dont resend UART if already going forward
            if (cmd != Instruction_t.FORWARD):
                cmd = Instruction_t.FORWARD
                communicator.send_string(cmd.value)
                print("send forward command")
        #else if you see a mouse and it's within distance, stop
        elif (dist <=20):
            if (cmd != Instruction_t.IDLE):
                cmd = Instruction_t.IDLE
                communicator.send_string(cmd.value)
                print("send idle command")
    #if you don't see the mouse, stop
    elif (cmd != Instruction_t.IDLE):
        cmd = Instruction_t.IDLE
        communicator.send_string(cmd.value)
        print("send idle command")

    # #***********************visual******************************
    # # Visualize the results on the frame
    # annotated_frame = results[0].plot()

    # # Display the resulting frame
    # cv2.imshow("Camera", annotated_frame)

    # Break the loop if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    # elif key == ord("s"):
    #     output_file = "./app/data/captured_image40"
    #     cv2.imwrite(f"{output_file}.jpg", annotated_frame)
    #     with open(f"{output_file}.txt", "w") as f:
    #         f.write(f"pixel width: {pixel_width}\n")


# cam.release()  # Release the camera resource
# # Release resources and close windows
# cv2.destroyAllWindows()