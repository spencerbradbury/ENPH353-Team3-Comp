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

IMITATION_PATH = '/home/fizzer/ros_ws/src/controller_pkg/ENPH353-Team3-Comp/media/Ped/'
DRIVING_MODEL_PATH = '/home/fizzer/ros_ws/src/controller_pkg/ENPH353-Team3-Comp/NNs/Imitation_model_color_more_grass_correction_V4.h5'
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
        self.state = -1
        self.autopilot = False
        self.driving_model = load_model('{}'.format(DRIVING_MODEL_PATH))
    
    def image_callback(self, msg):
        try:
            # Convert the image message to a cv2 object
            camera_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        if (self.isrecording == True):
            self.record_frames_states(camera_image)
        
        if (self.autopilot == True):
            self.drive_with_autopilot(camera_image)

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
            image_name = f"{time.time()}.jpg"
            cv2.imwrite(os.path.join(IMITATION_PATH, image_name), camera_image)
            self.record_count += 1
            if (self.record_count % 100 == 0):
                print(f"Recorded {self.record_count} frames")

    def drive_with_autopilot(self, camera_image):
        camera_image = cv2.resize(camera_image, (0,0), fx=0.2, fy=0.2) #if model uses grayscale
        #camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2GRAY)
        camera_image = np.float16(camera_image/255.)
        camera_image = camera_image.reshape((1, 144, 256, 3)) # 1 for gay, 3 for bgr
        
        predicted_actions = self.driving_model.predict(camera_image)
        print(predicted_actions)
        action = np.argmax(predicted_actions)
        comparator = np.random.randint(10, )/10.
        cmd_vel_msg = Twist()
        if (action == 0): #drive forwardcomparator < predicted_actions[0][0]
            cmd_vel_msg.linear.x = 0.3
            cmd_vel_msg.angular.z = 0
        elif(action == 1): #turn left comparator > predicted_actions[0][0] and comparator < predicted_actions[0][0]+predicted_actions[0][1]
            cmd_vel_msg.linear.x = 0.01
            cmd_vel_msg.angular.z = 1.
        else:
            cmd_vel_msg.linear.x = 0.01
            cmd_vel_msg.angular.z = -1.
        self.cmd_vel_pub.publish(cmd_vel_msg)

    def initialize_robot(self):
        pass
    
    def pedestrian_crossing(self, camera_image):
        pass
    
    def get_to_inner_loop(self, camera_image):
        pass


        
                

if __name__ =='__main__':
    rospy.init_node('camera_velocity_nodes')
    controller_nodes =  Controller()
    rospy.spin()