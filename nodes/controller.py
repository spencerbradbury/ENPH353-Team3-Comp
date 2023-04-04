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
#from tensorflow.keras import optimizers
#from tensorflow.keras.optimizers.experimental import WeightDecay

IMITATION_PATH = '/home/fizzer/ros_ws/src/controller_pkg/ENPH353-Team3-Comp/media/x-walks/'
DRIVING_MODEL_PATH_1 = '/home/fizzer/ros_ws/src/controller_pkg/ENPH353-Team3-Comp/NNs/Imitation_model_color_more_grass_correction_V4.h5'
DRIVING_MODEL_PATH_2 = '/home/fizzer/ros_ws/src/controller_pkg/ENPH353-Team3-Comp/NNs/Imitation_model_color_more_grass_correction_V6_40_01.h5'
##
# Class that will contain functions to control the robot
class Controller:
    def __init__(self):

        self.bridge = CvBridge()

        #define ros nodes
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size = 10)
        self.cmd_vel_sub = rospy.Subscriber("/R1/cmd_vel", Twist, self.velocity_callback)
        #set initial fields for robot velocity, 
        self.isrecording = False 
        self.xspeed = 0
        self.zang = 0
        self.record_count = 0
        self.state = -1 #for autopilot purposes
        self.robot_state = 0
        self.num_x_walks = 0
        self.innit_frames = 0
        self.x_frames = 0
        self.time_last_x_walk = 0
        self.autopilot = False
        self.driving_model_1 = load_model('{}'.format(DRIVING_MODEL_PATH_1))
        self.driving_model_2 = load_model('{}'.format(DRIVING_MODEL_PATH_2))
    
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

    def image_callback(self, msg):
        try:
            # Convert the image message to a cv2 object
            camera_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.state_machine(camera_image)
        cv2.imshow("Camera Feed", camera_image)
        cv2.waitKey(1)

    def velocity_callback(self, msg):
        #press t to start/stop recording 
        if (msg.linear.z > 0):
            self.isrecording = not self.isrecording
            print("Recording {}".format(self.isrecording))
        
        #press b to start/stop autopilot
        if (msg.linear.z < 0):
            self.autopilot = not self.autopilot
            print("Autopilot {}".format(self.autopilot))

        self.xspeed = msg.linear.x
        self.zang = msg.angular.z

    def record_frames_states(self, camera_image):
        if (self.record_count < 2000):
                if (self.xspeed != 0 or self.zang != 0):
                    if (self.xspeed > 0): #forward
                        self.state = 1
                    elif (self.zang > 0): #turn left 
                        self.state = 2
                    else: #turn right
                        self.state = 3
                    image_name = f"{self.state}_{time.time()}.jpg"
                    cv2.imwrite(os.path.join(IMITATION_PATH, image_name), camera_image)
                    self.record_count += 1
                    if (self.record_count % 100 == 0):
                        print(f"Recorded {self.record_count} frames")

    def drive_with_autopilot(self, camera_image):
        if (self.is_x_walk_in_front(camera_image) and (time.time() - self.time_last_x_walk) > 5):
            self.robot_state = 2
        else:
            camera_image = cv2.resize(camera_image, (0,0), fx=0.2, fy=0.2) #if model uses grayscale
            #camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2GRAY)
            camera_image = np.float16(camera_image/255.)
            camera_image = camera_image.reshape((1, 144, 256, 3)) # 1 for gay, 3 for bgr
            
            if self.robot_state == 1:
                predicted_actions = self.driving_model_1.predict(camera_image)
            else: predicted_actions = self.driving_model_2.predict(camera_image)
            #print(predicted_actions)
            action = np.argmax(predicted_actions)
            #comparator = np.random.randint(10, )/10.
            cmd_vel_msg = Twist()
            if (action == 0): #drive forwardcomparator < predicted_actions[0][0]
                cmd_vel_msg.linear.x = 0.3
                cmd_vel_msg.angular.z = 0
            elif(action == 1): #turn left comparator > predicted_actions[0][0] and comparator < predicted_actions[0][0]+predicted_actions[0][1]
                cmd_vel_msg.linear.x = 0.02
                cmd_vel_msg.angular.z = 1.
            else:
                cmd_vel_msg.linear.x = 0.02
                cmd_vel_msg.angular.z = -1.
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
        print("stop for x-walk")
        self.time_last_x_walk = time.time()
        self.robot_state = 3 #change to wait for ped in the future

    def wait_for_ped(self, camera_image):
        if(self.is_ped_crossing(camera_image)):
            self.cross_x_walk()
            
            
    def cross_x_walk(self):
        increment = 0.2
        #if (self.num_x_walks > 0):
        #    increment = 0.7
        curent_time = time.time()
        end_time = curent_time + increment
        #while(curent_time < end_time):
            # cmd_vel_msg = Twist()
            # cmd_vel_msg.linear.x = 0.5
            # cmd_vel_msg.angular.z = 0
            # self.cmd_vel_pub.publish(cmd_vel_msg)
            # curent_time = time.time()
        if self.num_x_walks >= 2:
            self.robot_state = 4
        else: self.robot_state = 1
        
    def is_x_walk_in_front (self, camera_image):
        hsv = cv2.cvtColor(camera_image, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0,112,114])
        upper_hsv = np.array([0,255,255])
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        threshold = 30
        max_value = 255

        _, mask = cv2.threshold(mask, threshold, max_value, cv2.THRESH_BINARY)
        mask = cv2.GaussianBlur(mask,(3,3),cv2.BORDER_DEFAULT)

        # Find the contours of the white shapes in the binary image
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return False
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest_contour)
        centroid_y = int(moments["m01"] / moments["m00"])
        bottom = int(camera_image.shape[0] / 9)
        if (centroid_y > camera_image.shape[0] - bottom and len(contours) > 1):
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


        
                

if __name__ =='__main__':
    rospy.init_node('camera_velocity_nodes')
    controller_nodes =  Controller()
    rospy.spin()