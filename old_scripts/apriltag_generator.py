import apriltag
import cv2
import numpy as np

# Define the AprilTag ID and size
tag_id = 0
tag_size = 200

# Create an AprilTag family object
tag_family = apriltag.Tag36h11()

# Create an options object with default settings
tag_options = apriltag.DetectorOptions()

# Generate the AprilTag image
tag_image = tag_family.render(tag_id, tag_size)

# Convert the image to grayscale
tag_image_gray = cv2.cvtColor(tag_image, cv2.COLOR_BGR2GRAY)

# Save the AprilTag as a PNG image
cv2.imwrite("april_tag_0.png", tag_image_gray)
