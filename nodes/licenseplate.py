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
import math

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

        self.uh2 = 60
        self.us2 = 8
        self.uv2 = 203
        self.lh2 = 0
        self.ls2 = 0
        self.lv2 = 90
        self.lower_hsv2 = np.array([self.lh2,self.ls2,self.lv2])
        self.upper_hsv2 = np.array([self.uh2,self.us2,self.uv2])

        self.line_lower_hsv = np.array([0,0,208])
        self.line_upper_hsv = np.array([255,255,255])

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
                #find center of conts
                M = cv.moments(conts)
                if M["m00"] != 0:
                    platecX = int(M["m10"] / M["m00"])
                    platecY = int(M["m01"] / M["m00"])
                else:
                    platecX, platecY = 0, 0
                (x, y, w, h) = cv.boundingRect(conts)
            else:
                (x, y, w, h) = cv.boundingRect(contour1)
                M = cv.moments(contour1)
                if M["m00"] != 0:
                    platecX = int(M["m10"] / M["m00"])
                    platecY = int(M["m01"] / M["m00"])
                else:
                    platecX, platecY = 0, 0
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print(f"Bounding Box Size: {w}x{h}")

        elif len(contours) == 1:
            contour1 = contours[0]
            (x, y, w, h) = cv.boundingRect(contour1)
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            M = cv.moments(contour1)
            if M["m00"] != 0:
                platecX = int(M["m10"] / M["m00"])
                platecY = int(M["m01"] / M["m00"])
            else:
                platecX, platecY = 0, 0
            print(f"Bounding Box Size: {w}x{h}")
        else:
            print("No contours found")
            platecX, platecY = 0, 0

            
        line_mask = cv.inRange(hsv, self.line_lower_hsv, self.line_upper_hsv)
        #make the line mask all 0s 
        cv.morphologyEx(line_mask, cv.MORPH_DILATE, (5,5), line_mask, iterations=5)
        cv.imshow("line mask", line_mask)
        line_mask = cv.bitwise_not(line_mask)
        blur = cv.GaussianBlur(hsv, (3, 3), 0)
        mask2 = cv.inRange(blur, self.lower_hsv2, self.upper_hsv2)
        cv.imshow("mask2", mask2)
        mask2 = cv.bitwise_and(mask2, line_mask)
        cv.imshow("Combined Mask", mask2)
        cv.morphologyEx(mask2, cv.MORPH_OPEN, (5,5), mask2, iterations=2)
        cv.morphologyEx(mask2, cv.MORPH_CLOSE, (5,5), mask2, iterations=2)
        cv.imshow("processed", mask2)
        contours2, _ = cv.findContours(mask2.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours2 = sorted(contours2, key=cv.contourArea, reverse=True)
        
        #Draw 2 largest contours in red and the rest in blue
        for i in range(min(len(contours2), 10)):
            #find the distance between the center of the plate and the center of the contour
            M = cv.moments(contours2[i])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = image.shape[1], image.shape[0]
            dist = math.sqrt((platecX - cX)**2 + (platecY - cY)**2)

            if dist < 120:
                cv.drawContours(image, contours2, i, (0, 0, 255), 2)
            else:
                cv.drawContours(image, contours2, i, (255, 0, 0), 2)

        cv.imshow("main", image)
        cv.waitKey(1)

if __name__ =='__main__':
    rospy.init_node('licenseplate')
    plateDetector =  PlateDetector()
    rospy.spin()