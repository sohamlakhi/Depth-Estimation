import cv2
import matplotlib.pyplot as plt

img_name = "Data_Realsense/150cm_2/light_2/color_frame.jpg"
bg_name = "Data_Monocular/150cm/light_2/background_frame.jpg"
# Load the input image
input_image = cv2.imread(img_name)

# Load the background image
background_image = cv2.imread(bg_name)
background_image = cv2.GaussianBlur(background_image, (5,5), 0)
background_image = cv2.resize(background_image, (640, 480))

# Convert both images to grayscale
gray_input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
gray_background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)

# Compute the absolute difference between the two grayscale images
diff_image = cv2.absdiff(gray_input_image, gray_background_image)
plt.imshow(diff_image)
plt.show()
plt.imshow(diff_image, 'gray')
plt.show()

# Apply a threshold to convert the difference image to binary
threshold_image = cv2.threshold(diff_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
plt.imshow(threshold_image)
plt.show()
plt.imshow(threshold_image, 'gray')
plt.show()

# Perform a morphological operation to remove any small holes or gaps in the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closing = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, kernel)
plt.imshow(closing)
plt.show()
plt.imshow(closing, 'gray')
plt.show()

# Apply a mask to the input image using the thresholded image as the mask
result = cv2.bitwise_and(input_image, input_image, mask=closing)
plt.imshow(result)
plt.show()
plt.imshow(result, 'gray')
plt.show()

# Convert the image to grayscale
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, 'gray')
plt.show()


# Threshold the grayscale image to extract the black object
# threshold_image = cv2.threshold(diff_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
threshold_image = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
plt.imshow(threshold_image, 'gray')
plt.show()

contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through the contours and find the contour with the largest area, which corresponds to the black object
max_area = 0
max_contour = None

for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        max_contour = contour

with_contours = cv2.drawContours(threshold_image,contours,-1,(0,255,0),3) 
plt.imshow(with_contours)
plt.show()
'''
build tree pipeline to extract images from folders and 
try two different pipelines
- only thresholding
- background subtraction and then thresholding

possibilities -> erode and dilate some stuff out
blur image
adaptive thresholding

blur image
subtract background (might have to interchange order or remove blur)
adaptive threshold (Otsu or other)
erode and dilate 
contour 

NOTE: if things are not working for farther away images, clip the image and depth map within region of interest by hand. Or add a mask
NOTE: 150cm background frame is wrong
NOTE: if classical methods are failing, use an object detection pretrained model

STEP BACK, and use chatgpt code with adaptive thresholding, morphological and bit mask in region of interest
'''