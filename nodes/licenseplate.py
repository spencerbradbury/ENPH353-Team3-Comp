#!/usr/bin/env python3

import numpy as np
import cv2 as cv
import time
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
from keras.models import load_model
from std_msgs.msg import String

# CHARACTER_MODEL_PATH = '/home/fizzer/ros_ws/src/controller_pkg/ENPH353-Team3-Comp/NNs/character_model.h5'

class PlateDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/plate_detection", Image, self.image_callback)
        self.license_plate_pub = rospy.Publisher("/license_plate", String, queue_size = 10)
        # self.character_model = load_model('{}'.format(CHARACTER_MODEL_PATH))
        self.uh = 125
        self.us = 107
        self.uv = 180
        self.lh = 107
        self.ls = 23
        self.lv = 89
        self.thresh = 45
        self.lower_hsv = np.array([self.lh,self.ls,self.lv])
        self.upper_hsv = np.array([self.uh,self.us,self.uv])

    def image_callback(self,msg):
        try:
            # Convert your ROS Image message to OpenCV2
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Convert BGR to HSV
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        # Threshold the HSV image to get only blue colors
        mask = cv.inRange(hsv, self.lower_hsv, self.upper_hsv)

        blurred = cv.GaussianBlur(mask, (31, 31), 0)
        thresh = cv.threshold(blurred, self.thresh, 255, cv.THRESH_BINARY)[1]

        ## get the contours of tresh sorted by contour size, largest first
        contours, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)

        if len(contours) >= 2:
            contour1 = contours[0]
            contour2 = contours[1]
            (x1, y1, w1, h1) = cv.boundingRect(contour1)
            (x2, y2, w2, h2) = cv.boundingRect(contour2)
            if (abs(y1-y2) < 15):
                conts = np.concatenate((contour1, contour2), axis=0)
                (x, y, w, h) = cv.boundingRect(conts)
            else:
                (x, y, w, h) = cv.boundingRect(contour1)
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped = image[y-50:y+int(h*1.1), x-10:x+int(w*1.1)]
            cropped = image[y:y+h, x:x+w]
            cv.imshow("cropped", cropped)
            print(f"Bounding Box Size: {w}x{h}")
            print(f"Cropped Image Size: {cropped.shape[0]}x{cropped.shape[1]}")

        elif len(contours) == 1:
            contour1 = contours[0]
            (x, y, w, h) = cv.boundingRect(contour1)
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cropped = image[y-50:y+int(h*1.1), x-10:x+int(w*1.1)]
            cropped = image[y:y+h, x:x+w]
            cv.imshow("cropped", cropped)
            print(f"Bounding Box Size: {w}x{h}")
            print(f"Cropped Image Size: {cropped.shape[0]}x{cropped.shape[1]}")

            
        cv.imshow("main", image)
        cv.waitKey(1)

if __name__ =='__main__':
    rospy.init_node('licenseplate')
    plateDetector =  PlateDetector()
    rospy.spin()