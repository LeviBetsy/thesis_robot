import cv2

cam = cv2.VideoCapture(0)


while True:
    ret, image = cam.read()
    cv2.imshow('test', image)
    k = cv2.waitKey(1)
    if k != -1: #if any key is pressed, exit the loop
        break

cam.release()
cv2.destroyAllWindows()