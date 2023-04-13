#!/usr/bin/env python3

import numpy as np
import cv2 as cv
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from keras.models import load_model
from std_msgs.msg import String
import time
import os
from geometry_msgs.msg import Twist

IMAGE_PATH = '/home/fizzer/ros_ws/src/controller_pkg/ENPH353-Team3-Comp/media/Plates/'
CHARACTER_MODEL_PATH = '/home/fizzer/ros_ws/src/controller_pkg/ENPH353-Team3-Comp/NNs/Chars/Chars_model_V4.h5'
DRIVING_MODEL_PATH = '/home/fizzer/ros_ws/src/controller_pkg/ENPH353-Team3-Comp/NNs/Imitation_model_V11_2_80_01_smaller.h5'

class PlateDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.character_model = load_model('{}'.format(CHARACTER_MODEL_PATH))

        self.plates_seen = np.zeros(8)

        self.thresh = 45
        self.lower_hsv = np.array([107,23,89])
        self.upper_hsv = np.array([125,107,180])

        self.backofcar_lower_hsv = np.array([0,0,90])
        self.backofcar_upper_hsv = np.array([60,8,203])

        self.line_lower_hsv = np.array([0,0,208])
        self.line_upper_hsv = np.array([255,255,255])

        self.char_upper_hsv = np.array([121,255,255])
        self.char_lower_hsv = np.array([118,38,88])

        #for recording
        self.park_spot = 1
        
        self.license_plate_pub = rospy.Publisher("/license_plate", String, queue_size = 10)
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.image_callback)

    def image_callback(self,msg):
        if (self.plates_seen[-1]):
            self.license_plate_pub.publish(str('Team3,SS,-1,DONE'))
        try:
            # Convert your ROS Image message to OpenCV2
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        #for testing with a 2nd driving model
        raw_image = image

        # Convert BGR to HSV
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        # Threshold the HSV image to get only blue colors
        mask = cv.inRange(hsv, self.lower_hsv, self.upper_hsv)
        blurred = cv.GaussianBlur(mask, (31, 31), 0)
        thresh = cv.threshold(blurred, self.thresh, 255, cv.THRESH_BINARY)[1]
        # cv.imshow("thresh", thresh)

        ## get the contours of tresh sorted by contour size, largest first
        contours, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)

        w_thresh = 30

        if len(contours) >= 2:
            contour1 = contours[0]
            contour2 = contours[1]
            (x1, y1, w1, h1) = cv.boundingRect(contour1)
            (x2, y2, w2, h2) = cv.boundingRect(contour2)
            if (abs(y1-y2) < 15):
                plate = np.concatenate((contour1, contour2), axis=0)
                (x, y, w, h) = cv.boundingRect(plate)
                M = cv.moments(plate)
                if M["m00"] != 0 and w > w_thresh:
                    platecX = int(M["m10"] / M["m00"])
                    platecY = int(M["m01"] / M["m00"])
                else:
                    platecX, platecY = 0, 0
            else:
                plate = contour1
                (x, y, w, h) = cv.boundingRect(contour1)
                M = cv.moments(contour1)
                if M["m00"] != 0 and w > w_thresh:
                    platecX = int(M["m10"] / M["m00"])
                    platecY = int(M["m01"] / M["m00"])
                else:
                    platecX, platecY = 0, 0

        elif len(contours) == 1:
            contour1 = contours[0]
            plate = contour1
            (x, y, w, h) = cv.boundingRect(contour1)
            M = cv.moments(contour1)
            if M["m00"] != 0 and w > w_thresh:
                platecX = int(M["m10"] / M["m00"])
                platecY = int(M["m01"] / M["m00"])
            else:
                platecX, platecY = 0, 0
        else:
            platecX, platecY = 0, 0

            
        line_mask = cv.inRange(hsv, self.line_lower_hsv, self.line_upper_hsv)
        #make the line mask all 0s in the upper half of the image
        line_mask[0:10*720//17,:] = 0
        blurred_line = cv.GaussianBlur(line_mask, (41, 41), 0)
        thresh_line = cv.threshold(blurred_line, 1, 255, cv.THRESH_BINARY)[1]
        line_mask = thresh_line.copy()
        line_mask = cv.bitwise_not(line_mask)
        mask2 = cv.inRange(hsv, self.backofcar_lower_hsv, self.backofcar_upper_hsv)
        mask2 = cv.bitwise_and(mask2, line_mask)
        cv.morphologyEx(mask2, cv.MORPH_OPEN, (5,5), mask2, iterations=2)
        cv.morphologyEx(mask2, cv.MORPH_CLOSE, (5,5), mask2, iterations=2)
        contours2, _ = cv.findContours(mask2.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours2 = sorted(contours2, key=cv.contourArea, reverse=True)
        

        all_contours = []
        for i in range(min(len(contours2), 5)):
            #find the distance between the center of the plate and the center of the contour
            #pass if contour does not have a size between 100 and 1000
            if cv.contourArea(contours2[i]) < 200:
                continue

            M = cv.moments(contours2[i])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = image.shape[1], image.shape[0]

            xdist = abs(platecX - cX)
            ydist = abs(platecY - cY)

            if ydist < 150 and xdist < 25:
                all_contours.append(contours2[i])
            else:
                pass

        if len(all_contours) == 2:
            #find the corners of each contour
            all_corners = []
            for c in all_contours:
                epsilon = 0.03 * cv.arcLength(c, True)
                approx = cv.approxPolyDP(c, epsilon, True)
                hull = cv.convexHull(approx)
                corners = np.int0(hull)
                for c in corners:         
                    all_corners.append(c)
            
            #sort all corners based on y height
            all_corners.sort(key=lambda x: x[0][1])
            top = all_corners[:2]
            bot = all_corners[-2:]
            top.sort(key=lambda x: x[0][0])
            bot.sort(key=lambda x: x[0][0])

            corners = np.array([top[1][0], top[0][0], bot[1][0], bot[0][0]])
            
            # Define the new set of four points representing the desired perspective-shifted shape
            dst_points = np.array([[300, 0],[0, 0],[300, 400],[0, 400]],dtype=np.float32)
            
            # Find the perspective transformation matrix that maps the original contour to the new contour
            M = cv.getPerspectiveTransform(corners.astype(np.float32), dst_points)
            
            # # Apply the transformation to the original image
            result = cv.warpPerspective(image, M, (300, 400))

            hsv_result = cv.cvtColor(result, cv.COLOR_BGR2HSV)
            result_mask = cv.inRange(hsv_result, self.char_lower_hsv, self.char_upper_hsv)
            cv.morphologyEx(result_mask, cv.MORPH_OPEN, (5,5), result_mask, iterations=2)
            contours, _ = cv.findContours(result_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv.contourArea, reverse=True)

            total_area = 0

            for c in contours:
                if cv.contourArea(c) > 300:
                    (x, y, w, h) = cv.boundingRect(c)
                    total_area += w*h
            
            #Look for area in this range, and at least 2 contours
            #6500 and 7000 worked very well, but somtimes missed a car. this never missed, but sometimes got a false positivegti
            if total_area > 6100 and total_area < 7500 and len(contours) >= 2:
                try:
                    chars = self.get_chars_from_image(result)
                    id_im = self.get_ID_from_image(result)
                    if (len(chars) == 4 and chars[3].shape == (40, 50)):
                        char0 = np.argmax(self.character_model(chars[0].reshape((1, 40, 50, 1)))[0][:26])
                        char1 = np.argmax(self.character_model(chars[1].reshape((1, 40, 50, 1)))[0][:26])
                        char2 = np.argmax(self.character_model(chars[2].reshape((1, 40, 50, 1)))[0][26:])
                        char3 = np.argmax(self.character_model(chars[3].reshape((1, 40, 50, 1)))[0][26:])
                        id = np.argmax(self.character_model(id_im.reshape((1, 40, 50, 1)))[0][26:])
                        
                        char0 = chr(char0+65)
                        char1 = chr(char1+65)
                        char2 = chr(char2+48)
                        char3 = chr(char3+48)
                        self.plates_seen[id-1] = 1
                        id = chr(id+48)
                        print(f"P{id}: {char0}{char1}{char2}{char3}")
                        self.license_plate_pub.publish(str(f'Team3,SS,{id},{char0}{char1}{char2}{char3}')) 
                except Exception as e:
                    print(e)

    def get_chars_from_image(self, result):
        hsv_result = cv.cvtColor(result, cv.COLOR_BGR2HSV)
        result_mask = cv.inRange(hsv_result, self.char_lower_hsv, self.char_upper_hsv)
        cv.morphologyEx(result_mask, cv.MORPH_OPEN, (5,5), result_mask, iterations=2)
        contours, _ = cv.findContours(result_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)

        bounding_boxes = []
        for c in contours:
                            if cv.contourArea(c) > 300:
                                (x, y, w, h) = cv.boundingRect(c)
                                bounding_boxes.append((x, y, w, h))
        bounding_boxes.sort(key=lambda x: x[0], reverse=False)
        chars = []

        for box in bounding_boxes:
            if box[2] > 60:
                chars.append(cv.cvtColor(result[box[1]: box[1]+40, box[0]: box[0]+50], cv.COLOR_BGR2GRAY))
                chars.append(cv.cvtColor(result[box[1]: box[1]+40, box[0]+box[2]//2: box[0]+box[2]//2+50], cv.COLOR_BGR2GRAY))
            else:
                chars.append(cv.cvtColor(result[box[1]: box[1]+40, box[0]: box[0]+50], cv.COLOR_BGR2GRAY))
        return chars
    
    def get_ID_from_image(self, result):
        hsv_result = cv.cvtColor(result, cv.COLOR_BGR2HSV)
        result_mask = cv.inRange(hsv_result, np.array([0,0,0]), np.array([0,0,70]))
        cv.morphologyEx(result_mask, cv.MORPH_OPEN, (5,5), result_mask, iterations=2)
        contours, _ = cv.findContours(result_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)

        bounding_boxes = []
        for c in contours:
            if cv.contourArea(c) > 300:
                (x, y, w, h) = cv.boundingRect(c)
                bounding_boxes.append((x, y, w, h))
        bounding_boxes.sort(key=lambda x: x[0], reverse=False)
        if (len(bounding_boxes) == 2):
            return cv.resize(cv.cvtColor(result[bounding_boxes[1][1]-10: bounding_boxes[1][1]+70, bounding_boxes[1][0]-20: bounding_boxes[1][0]+130], cv.COLOR_BGR2GRAY),(50,40))
        else:
            raise Exception("No ID found")

if __name__ =='__main__':
    rospy.init_node('licenseplate')
    plateDetector =  PlateDetector()
    rospy.spin()