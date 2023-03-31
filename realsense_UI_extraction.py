import tkinter as tk
import cv2
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pyrealsense2 as rs

class CameraApp:
    def __init__(self, master):
        self.master = master
        master.title("Camera App")

        # Create a slider to adjust the color range
        self.slider_label = tk.Label(master, text="H Range")
        self.slider_label.pack()
        self.slider = tk.Scale(master, from_=0, to=255, orient=tk.HORIZONTAL, length=400)
        self.slider.pack()

        # Create a slider to adjust the color range
        self.slider_label = tk.Label(master, text="S Range")
        self.slider_label.pack()
        self.slider = tk.Scale(master, from_=0, to=255, orient=tk.HORIZONTAL, length=400)
        self.slider.pack()

        # Create a slider to adjust the color range
        self.slider_label = tk.Label(master, text="V Range")
        self.slider_label.pack()
        self.slider = tk.Scale(master, from_=0, to=255, orient=tk.HORIZONTAL, length=400)
        self.slider.pack()

        # Create a text box for filename input
        self.filename_label = tk.Label(master, text="Filename:")
        self.filename_label.pack()
        self.filename_entry = tk.Entry(master)
        self.filename_entry.pack()

        # Create a text box for parentfolder input
        self.foldername_label = tk.Label(master, text="Parent Folder:")
        self.foldername_label.pack()
        self.foldername_entry = tk.Entry(master)
        self.foldername_entry.pack()

        # Create a text box for lighting input
        self.lighting_label = tk.Label(master, text="lighting Folder:")
        self.lighting_label.pack()
        self.lighting_entry = tk.Entry(master)
        self.lighting_entry.pack()

        # Create a text box for distance input
        self.distance_label = tk.Label(master, text="distance Folder:")
        self.distance_label.pack()
        self.distance_entry = tk.Entry(master)
        self.distance_entry.pack()

        # Create a button to capture image
        self.capture_button = tk.Button(master, text="Capture", command=self.capture_image)
        self.capture_button.pack()

        # Create a window to show camera stream
        self.video_frame = tk.Frame(master, width=1280, height=720)
        self.video_frame.pack()
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        # Open camera stream
        self.pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if self.device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.profile = self.pipeline.start(self.config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        print("Depth Scale is: " , self.depth_scale)

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        self.update()

    def capture_image(self):
        frames = self.pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_image = depth_image * self.depth_scale

        filename1 = self.filename_entry.get() + ".jpg"
        filename2 = self.filename_entry.get() + ".png"
        parent_folder = self.foldername_entry.get()
        distance = self.distance_entry.get()
        lighting = self.lighting_entry.get()
        pathh = os.path.join(parent_folder,distance, lighting)
        if not os.path.exists(pathh):
            os.makedirs(pathh, exist_ok=True)
        cv2.imwrite(os.path.join(pathh,filename1), color_image)
        cv2.imwrite(os.path.join(pathh,filename2), color_image)

        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        # define range of blue color in HSV

        H_range = self.slider.get() #128
        S_range = self.slider.get() #39
        V_range = self.slider.get() #255
        lower_neon = np.array([60-H_range,50,50])
        upper_neon = np.array([60+H_range,S_range,V_range])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_neon, upper_neon)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(color_image,color_image, mask= mask)
        plt.imshow(res)
        plt.show()

        # Convert the image to grayscale
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        plt.imshow(gray, 'gray')
        plt.show()

        # Threshold the grayscale image to extract the black object
        _, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
        plt.imshow(thresh, 'gray')
        plt.show()

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through the contours and find the contour with the largest area, which corresponds to the black object
        max_area = 0
        max_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        with_contours = cv2.drawContours(thresh,contours,-1,(0,255,0),3) 
        plt.imshow(with_contours)
        plt.show()
        depth_map = depth_image
        print(depth_map)
        print(type(depth_map))
        if max_contour is not None:
            x, y, w, h = cv2.boundingRect(max_contour)
            center_x = x + w//2
            center_y = y + h//2
            depth = depth_map[center_y, center_x]
            print("Depth of black object: ", depth)
        else:
            print("Black object not found in the image.")


#conversion depth estimate - 13cm for test 

    def update(self):
        frames = self.pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
       
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        # define range of blue color in HSV

        H_range = self.slider.get() #128
        S_range = self.slider.get() #39
        V_range = self.slider.get() #255
        lower_neon = np.array([60-H_range,50,50])
        upper_neon = np.array([60+H_range,S_range,V_range])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_neon, upper_neon)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(color_image,color_image, mask= mask)
        img = tk.PhotoImage(data=cv2.imencode('.png', res)[1].tobytes())
        self.video_label.config(image=img)
        self.video_label.image = img
        self.master.after(10, self.update)

root = tk.Tk()
app = CameraApp(root)
root.mainloop()