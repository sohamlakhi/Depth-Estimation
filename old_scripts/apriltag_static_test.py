import cv2
import numpy as np
import apriltag

imagepath = '~/Downloads/tag41_12_00000.png'
image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
detector = apriltag("tagStandard41h12")

detections = detector.detect(image)