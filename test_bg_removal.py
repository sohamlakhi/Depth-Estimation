import cv2
import matplotlib.pyplot as plt

img_name = "Data_Realsense/100cm_2/light_2/color_frame.jpg"
bg_name = "Data_Monocular/100cm/light_2/background_frame.jpg"
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


