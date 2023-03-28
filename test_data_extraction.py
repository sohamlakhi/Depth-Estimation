import cv2
import matplotlib.pyplot as plt

img_name = "Data_Realsense/50cm_2/light_2/color_frame.jpg"
img_bgr = cv2.imread(img_name, 1)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# img_rgb = img_bgr

print("Image shape: {}".format(img_rgb.shape))

plt.imshow(img_rgb)
# plt.imshow(img_bgr)
plt.show()

# cv2.imwrite("rgb_test.jpg", img_rgb)

bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Convert the image to grayscale
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, 'gray')
plt.show()

# Apply background subtraction to the grayscale image
fg_mask = bg_subtractor.apply(gray)

# Threshold the grayscale image to extract the black object
_, thresh = cv2.threshold(fg_mask, 130, 255, cv2.THRESH_BINARY_INV)

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
'''
build tree pipeline to extract images from folders and 
try two different pipelines
- only thresholding
- background subtraction and then thresholding

read website
use chatgpt code
finish by 8:30

possibilities -> erode and dilate some stuff out
blur image
adaptive thresholding

blur image
subtract background (might have to interchange order or remove blur)
adaptive threshold (Otsu or other)
erode and dilate 
contour 

NOTE: if things are not working for farther away images, clip the image and depth map within region of interest by hand
NOTE: 150cm background frame is wrong
'''