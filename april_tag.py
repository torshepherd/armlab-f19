import cv2
import numpy as np
from apriltag import apriltag


imagepath = './data/apriltag_test.png'
# imagepath = '~/ArmLab_Tor_Siyuan/apriltag-imgs/tag36h11/tag36_11_00341.png'
image     = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
detector = apriltag("tag36h11")
# cv2.imshow('dick', image)
cv2.waitKey(0)
detections = detector.detect(image)
print(detections)