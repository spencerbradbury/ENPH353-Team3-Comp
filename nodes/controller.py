#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
import numpy as np
import os
import time
from keras.models import load_model
from std_msgs.msg import String
#from tensorflow.keras import optimizers
#from tensorflow.keras.optimizers.experimental import WeightDecay

IMITATION_PATH = '/home/fizzer/ros_ws/src/controller_pkg/ENPH353-Team3-Comp/media/x-walks/'
DRIVING_MODEL_PATH_1 = '/home/fizzer/ros_ws/src/controller_pkg/ENPH353-Team3-Comp/NNs/Imitation_model_V14_1_80_01_smaller.h5'
INPUT1 = [36, 64]
F1 = 0.05
MASKING_PATH = '/home/fizzer/ros_ws/src/controller_pkg/ENPH353-Team3-Comp/media/masking/'
DRIVING_MODEL_PATH_2 = '/home/fizzer/ros_ws/src/controller_pkg/ENPH353-Team3-Comp/NNs/Imitation_model_V15_2_100_01_smaller.h5'
INPUT2 = [36, 64]
F2 = 0.05
##
# Class that will contain functions to control the robot
class Controller:
    def __init__(self):

        self.bridge = CvBridge()

        #define ros nodes
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size = 10)
        self.license_plate_pub = rospy.Publisher("/license_plate", String, queue_size = 10)
        self.plate_detection_pub = rospy.Publisher("/plate_detection", Image, queue_size = 1)
        #set initial fields for robot velocity, 
        self.isrecording = False 
        self.recording_count = -1
        self.frame_count = 0
        self.xspeed = 0
        self.zang = 0
        self.record_count = 0
        self.state = -1 #for autopilot purposes
        self.robot_state = 0
        self.num_x_walks = 0
        self.innit_frames = 0
        self.x_frames = 0
        self.time_last_x_walk = 0
        self.truck_passing = 0
        self.is_inside = False
        self.last_frame = np.zeros((1280, 720)) 
        self.autopilot = False
        self.driving_model_1 = load_model('{}'.format(DRIVING_MODEL_PATH_1))
        self.driving_model_2 = load_model('{}'.format(DRIVING_MODEL_PATH_2))
        self.license_plate_pub.publish(str("Team3,SS,0,GOGO"))
    
    def state_machine(self, camera_image):

        # innitialize
        if (self.robot_state == 0):
            self.innitialize_robot()
        #autopilot 1
        if (self.robot_state == 1 or self.robot_state == 4):
            self.drive_with_autopilot(camera_image)
        #x-walk stop
        if (self.robot_state == 2):
            self.pedestrian_crossing_stop()
        #wait for ped and cross the x-walk
        if (self.robot_state == 3):
            self.wait_for_ped(camera_image)
        if (self.robot_state == 5):
            self.wait_for_truck(camera_image)

    def image_callback(self, msg):
        try:
            # Convert the image message to a cv2 object
            camera_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            
        self.plate_detection_pub.publish(msg)
        self.state_machine(camera_image)
        #cv2.imshow("Camera Feed", camera_image)
        #cv2.waitKey(1)

    def drive_with_autopilot(self, camera_image):
        if (self.is_x_walk_in_front(camera_image) and (time.time() - self.time_last_x_walk) > 2):
            self.robot_state = 2
        elif (self.robot_state == 4 and not self.is_inside and self.has_entered_inner_loop(camera_image)):
            self.robot_state = 5
        else:
            if self.robot_state == 1:
                camera_image = cv2.resize(camera_image, (0,0), fx=F1, fy=F1) 
                camera_image = np.float16(camera_image/255.)
                camera_image = camera_image.reshape((1, INPUT1[0], INPUT1[1], 3))
            else:
                camera_image = cv2.resize(camera_image, (0,0), fx=F2, fy=F2) 
                camera_image = np.float16(camera_image/255.)
                camera_image = camera_image.reshape((1, INPUT2[0], INPUT2[1], 3))   
            if self.robot_state == 1:
                predicted_actions = self.driving_model_1(camera_image)
                linear_x = 0.5 #0.3
                angular_z = 2.8
            else: 
                predicted_actions = self.driving_model_2(camera_image)
                if self.is_inside == True:
                    linear_x = 0.3
                    angular_z = 2.8
                else:
                    linear_x = 0.4 #0.3
                    angular_z = 2.5 #2.5
            action = np.argmax(predicted_actions)
            cmd_vel_msg = Twist()
            if (action == 0): #drive forward
                cmd_vel_msg.linear.x = linear_x
                cmd_vel_msg.angular.z = 0
            elif(action == 1): #turn left 
                cmd_vel_msg.linear.x = linear_x
                cmd_vel_msg.angular.z = angular_z #2.2
            else:
                cmd_vel_msg.linear.x = linear_x
                cmd_vel_msg.angular.z = -angular_z
            self.cmd_vel_pub.publish(cmd_vel_msg)

    def innitialize_robot(self):
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = .3
        cmd_vel_msg.angular.z = 1.
        self.cmd_vel_pub.publish(cmd_vel_msg)
        if(self.innit_frames > 15):
            self.robot_state = 1 #transition to autopilot 1 state
        self.innit_frames+=1
    
    def pedestrian_crossing_stop(self):
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = 0
        cmd_vel_msg.angular.z = 0
        self.cmd_vel_pub.publish(cmd_vel_msg)
        self.robot_state = 3 #change to wait for ped in the future

    def wait_for_ped(self, camera_image):
        if(self.is_ped_crossing(camera_image)):
            self.cross_x_walk()
            
            
    def cross_x_walk(self):
        if self.num_x_walks >= 2:
            self.robot_state = 4
        else: self.robot_state = 1
        self.time_last_x_walk = time.time()
        
    def is_x_walk_in_front (self, camera_image):
        hsv = cv2.cvtColor(camera_image, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0,112,114])
        upper_hsv = np.array([0,255,255])
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        threshold = 30
        max_value = 255

        _, mask = cv2.threshold(mask, threshold, max_value, cv2.THRESH_BINARY)
        mask = cv2.GaussianBlur(mask,(5,5),cv2.BORDER_DEFAULT)
        #cv2.imshow("mask", mask)
        #cv2.waitKey(1)
        # Find the contours of the white shapes in the binary image
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return False
        # Find the largest contour
        #largest_contour = max(contours, key=cv2.contourArea)
        bottom = int(camera_image.shape[0] / 9)
        for contour in contours:
            moments = cv2.moments(contour)
            centroid_y = int(moments["m01"] / moments["m00"])
            if (centroid_y > camera_image.shape[0] - bottom):
                self.num_x_walks+=1
                return True
        return False 
        
    def is_ped_crossing(self, camera_image):
        height, width = camera_image.shape[:2]
        # Set the number of pixels to cut from each side
        num_pixels_v = 500
        num_pixels_h = 300

        # Cut the image by removing the specified number of pixels from each side
        image = camera_image[0:height-num_pixels_h, num_pixels_v:width-num_pixels_v]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0,112,114])
        upper_hsv = np.array([0,255,255])
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        threshold = 30
        max_value = 255

        _, mask = cv2.threshold(mask, threshold, max_value, cv2.THRESH_BINARY)
        mask = cv2.GaussianBlur(mask,(5,5),cv2.BORDER_DEFAULT)
        # Find the contours of the white shapes in the binary image
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if (len(contours) >= 2):
            return True
        return False

    def has_entered_inner_loop(self, camera_image):
        threshold = 90
        height, width = camera_image.shape[:2]
        num_pixels_top = 450
        num_pixels_bot = 220
        num_pixels_l = 500
        num_pixels_r = 150
        gray = cv2.cvtColor(camera_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        binary = binary[num_pixels_top:height-num_pixels_bot, num_pixels_l:width-num_pixels_r]
        # cv2.imshow("Camera Feed", binary)
        # cv2.waitKey(1)
        if (np.sum(binary) == 0):
            cmd_vel_msg = Twist()
            cmd_vel_msg.linear.x = 0
            cmd_vel_msg.angular.z = 0
            self.cmd_vel_pub.publish(cmd_vel_msg)
            self.last_frame = camera_image
            self.is_inside = True
            return True
        else: return False
    
    def wait_for_truck(self, camera_image):
        last_gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)/255.
        current_gray = cv2.cvtColor(camera_image, cv2.COLOR_BGR2GRAY)/255.
        diff_img = cv2.absdiff(last_gray, current_gray)
        difference = diff_img.sum()
        if difference >= 10_000 or difference <= 7_500:
            if(self.truck_passing > 4):
                self.robot_state = 4
                if difference >= 10_000:
                    increment = 2
                    curent_time = time.time()
                    end_time = curent_time + increment
                    while(curent_time < end_time):
                        curent_time = time.time()
            self.truck_passing+=1
        self.last_frame = camera_image
    
if __name__ =='__main__':
    rospy.init_node('camera_velocity_nodes')
    controller_nodes =  Controller()
    rospy.spin()