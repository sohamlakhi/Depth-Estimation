import tkinter as tk
import cv2
import os

class CameraApp:
    def __init__(self, master):
        self.master = master
        master.title("Camera App")

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

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # img = cv2.resize(frame, (1280, 720))
            img = tk.PhotoImage(data=cv2.imencode('.png', frame)[1].tobytes())
            self.video_label.config(image=img)
            self.video_label.image = img
        self.master.after(10, self.update)

root = tk.Tk()
app = CameraApp(root)
root.mainloop()
