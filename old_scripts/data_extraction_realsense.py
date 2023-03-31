import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import pyrealsense2 as rs

img_name = "Data_Realsense/250cm_2/light_2/color_frame.jpg"
depth_name = "Data_Realsense/250cm_2/light_2/depth_frame.jpg"

img = cv2.imread(img_name)
plt.imshow(img)
plt.show()
depth = np.load('Data_Realsense/50cm_2/light_2/depth_array.npy')
plt.imshow(depth)
plt.show()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


img_bgr = cv2.imread(img_name, 1)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# Convert the image to grayscale
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, 'gray')
plt.show()
cv2.imwrite("bad_gray_scale.jpg", gray)

# Threshold the grayscale image to extract the black object
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

plt.imshow(thresh, 'gray')
plt.show()
cv2.imwrite('bad_thresh.jpg', thresh)

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

depth_map = depth
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