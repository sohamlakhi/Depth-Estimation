import cv2
import matplotlib.pyplot as plt
import torch

img_name = "test/40cm/natural/test.jpg"

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

img = cv2.imread(img_name)
plt.imshow(img)
plt.show()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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


img_bgr = cv2.imread(img_name, 1)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# img_rgb = img_bgr

print("Image shape: {}".format(img_rgb.shape))

plt.imshow(img_rgb)
# plt.imshow(img_bgr)
plt.show()



# Convert the image to grayscale
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, 'gray')
plt.show()



# Threshold the grayscale image to extract the black object
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

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

depth_map = output
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