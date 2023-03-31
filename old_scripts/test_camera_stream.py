import cv2
import time
import tkinter as tk

# cap = cv2.VideoCapture("v4l2-ctl device=/dev/video2 extra-controls=\"c,exposure_auto-3\" ! video/x-raw, width=960, height=540 ! videoconvert ! video/x-raw,format=BGR ! appsink")
cap = cv2.VideoCapture(2)
window = tk.Tk()
button = tk.Button(window, text="Capture Image")
entry = tk.Entry(window)

# Define a function to capture an image from the webcam
def capture_image():
    filename = entry.get()
    cv2.imwrite(filename, img)

# Attach the capture_image function to the button
button.config(command=capture_image)

# Display the button in the window
button.pack()
entry.pack()
window.mainloop()

time_old = time.time()

if cap.isOpened():
    cv2.namedWindow("demo", cv2.WINDOW_AUTOSIZE)
    while True:
        time_now = time.time()
        global img
        ret_val, img = cap.read()
        print(1/(time_now-time_old), 'Hz')
        time_old = time_now
        cv2.imshow('demo', img)
        cv2.waitKey(1)

    else:
        print("camera open failed")

cv2.destroyAllWindows()








