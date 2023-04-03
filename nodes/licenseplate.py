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
        window_name = "HSV Calibrator"
        cv.namedWindow(window_name)


        window_name2 = "Thresh"
        cv.namedWindow(window_name2)

        blurred = cv.GaussianBlur(mask, (31, 31), 0)
        thresh = cv.threshold(blurred, self.thresh, 255, cv.THRESH_BINARY)[1]

        def nothing(x):
            print("Trackbar value: " + str(x))
            pass

        # create trackbars for Upper HSV
        cv.createTrackbar('UpperH',window_name,0,255,nothing)
        cv.setTrackbarPos('UpperH',window_name, self.uh)

        cv.createTrackbar('UpperS',window_name,0,255,nothing)
        cv.setTrackbarPos('UpperS',window_name, self.us)

        cv.createTrackbar('UpperV',window_name,0,255,nothing)
        cv.setTrackbarPos('UpperV',window_name, self.uv)

        # create trackbars for Lower HSV
        cv.createTrackbar('LowerH',window_name,0,255,nothing)
        cv.setTrackbarPos('LowerH',window_name, self.lh)

        cv.createTrackbar('LowerS',window_name,0,255,nothing)
        cv.setTrackbarPos('LowerS',window_name, self.ls)

        cv.createTrackbar('LowerV',window_name,0,255,nothing)
        cv.setTrackbarPos('LowerV',window_name, self.lv)

        # create trackbars for Threshold
        cv.createTrackbar('Thresh',window_name2,0,255,nothing)
        cv.setTrackbarPos('Thresh',window_name2, self.thresh)

        font = cv.FONT_HERSHEY_SIMPLEX

         # Threshold the HSV image to get only blue colors
        cv.putText(mask,'Lower HSV: [' + str(self.lh) +',' + str(self.ls) + ',' + str(self.lv) + ']', (10,30), font, 0.5, (200,255,155), 1, cv.LINE_AA)
        cv.putText(mask,'Upper HSV: [' + str(self.uh) +',' + str(self.us) + ',' + str(self.uv) + ']', (10,60), font, 0.5, (200,255,155), 1, cv.LINE_AA)

        cv.imshow(window_name,mask)
        cv.imshow(window_name2, thresh)
        cv.waitKey(1)


        # get current positions of Upper HSV trackbars
        self.uh = cv.getTrackbarPos('UpperH',window_name)
        self.us = cv.getTrackbarPos('UpperS',window_name)
        self.uv = cv.getTrackbarPos('UpperV',window_name)
        upper_blue = np.array([self.uh,self.us,self.uv])
        # get current positions of Lower HSCV trackbars
        self.lh = cv.getTrackbarPos('LowerH',window_name)
        self.ls = cv.getTrackbarPos('LowerS',window_name)
        self.lv = cv.getTrackbarPos('LowerV',window_name)
        self.upper_hsv = np.array([self.uh,self.us,self.uv])
        self.lower_hsv = np.array([self.lh,self.ls,self.lv])
        # get current positions of Threshold trackbars
        self.thresh = cv.getTrackbarPos('Thresh',window_name2)

if __name__ =='__main__':
    rospy.init_node('licenseplate')
    plateDetector =  PlateDetector()
    rospy.spin()