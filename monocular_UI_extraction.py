import tkinter as tk
import cv2
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

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
        self.cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
        # Set the resolution to 640x480
        color_code = 'MJPG'
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*color_code))

        self.update()

    def capture_image(self):
        ret, frame = self.cap.read()
        filename1 = self.filename_entry.get() + ".jpg"
        filename2 = self.filename_entry.get() + ".png"
        parent_folder = self.foldername_entry.get()
        distance = self.distance_entry.get()
        lighting = self.lighting_entry.get()
        pathh = os.path.join(parent_folder,distance, lighting)
        if not os.path.exists(pathh):
            os.makedirs(pathh, exist_ok=True)
        cv2.imwrite(os.path.join(pathh,filename1), frame)
        cv2.imwrite(os.path.join(pathh,filename2), frame)

        model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

        midas = torch.hub.load("intel-isl/MiDaS", model_type)

        """Move model to GPU if available"""

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()

        """Load transforms to resize and normalize the image for large or small model"""

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

        """Load image and apply transforms"""

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)

        """Predict and resize to original resolution"""

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()

        """Show result"""

        plt.imshow(output)
        plt.show()
        cv2.imwrite(os.path.join(pathh,'depth_frame.png'), output)
        np.save(os.path.join(pathh,'depth_map'), output)

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(frame, (1280, 720))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # define range of blue color in HSV

        H_range = self.slider.get() #128
        S_range = self.slider.get() #39
        V_range = self.slider.get() #255
        lower_neon = np.array([10,30,30])
        upper_neon = np.array([60,255,255])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_neon, upper_neon)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)
        plt.imshow(res)
        plt.show()

        # Convert the image to grayscale
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        plt.imshow(gray, 'gray')
        plt.show()

        # Threshold the grayscale image to extract the black object
        x = 420-100
        y = 300
        deltax = 500
        deltay = 400
        _, thresh = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY)
        thresh_cropped = thresh[y:y+deltay, x:x+deltax]
        #remember to change heatmap as well
        plt.imshow(thresh_cropped, 'gray')
        plt.show()

        contours, _ = cv2.findContours(thresh_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through the contours and find the contour with the largest area, which corresponds to the black object
        max_area = 0
        max_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        with_contours = cv2.drawContours(thresh_cropped,contours,-1,(0,255,0),3) 
        plt.imshow(with_contours)
        plt.show()

        depth_map = output[y:y+deltay, x:x+deltax]
        # print(depth_map)
        # print(type(depth_map))
        if max_contour is not None:
            x, y, w, h = cv2.boundingRect(max_contour)
            depth_cylinder = depth_map[y:y+h,x:x+w]
            df = pd.DataFrame(depth_cylinder)
            df.to_csv(os.path.join(pathh,'depth_cylinder.csv'), index=False, header=False)
            center_x = x + w//2
            center_y = y + h//2
            depth = depth_map[center_y, center_x]
            print("Depth of black object (center): ", depth)
            print('mean', np.mean(depth_cylinder))
        else:
            print("Black object not found in the image.")


#conversion depth estimate - 13cm for test 

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # img = cv2.resize(frame, (1280, 720))
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # define range of blue color in HSV

            H_range = self.slider.get() #128
            S_range = self.slider.get() #39
            V_range = self.slider.get() #255
            lower_neon = np.array([10,30,30])
            upper_neon = np.array([60,255,255])

            # Threshold the HSV image to get only blue colors
            mask = cv2.inRange(hsv, lower_neon, upper_neon)

            # Bitwise-AND mask and original image
            res = cv2.bitwise_and(frame,frame, mask= mask)
            img = tk.PhotoImage(data=cv2.imencode('.png', res)[1].tobytes())
            self.video_label.config(image=img)
            self.video_label.image = img
        self.master.after(10, self.update)

root = tk.Tk()
app = CameraApp(root)
root.mainloop()