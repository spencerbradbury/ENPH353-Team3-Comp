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

IMITATION_PATH = '/home/fizzer/ros_ws/src/controller_pkg/ENPH353-Team3-Comp/media/Imitation Learning Feed/'
MASKING_PATH = '/home/fizzer/ros_ws/src/controller_pkg/ENPH353-Team3-Comp/media/masking/'
DRIVING_MODEL_PATH = '/home/fizzer/ros_ws/src/controller_pkg/ENPH353-Team3-Comp/NNs/Imitation_model.h5'
##
# Class that will contain functions to control the robot
class Controller:
    def __init__(self):

        self.bridge = CvBridge()


        #Initialize time, used to stop clock in time trial
        self.start_time = time.time()

        #define ros nodes
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size = 10)
        self.cmd_vel_sub = rospy.Subscriber("/R1/cmd_vel", Twist, self.velocity_callback)
        self.license_plate_pub = rospy.Publisher("/license_plate", String, queue_size = 10)
        self.plate_detection_pub = rospy.Publisher("/plate_detection", Image, queue_size = 1)
        #set initial fields for robot velocity, 
        self.isrecording = False 
        self.recording_count = -1
        self.frame_count = 0
        self.xspeed = 0
        self.zang = 0
        self.state = -1
        self.autopilot = False
        self.driving_model = load_model('{}'.format(DRIVING_MODEL_PATH))

        #start timer
        time.sleep(1)
        self.license_plate_pub.publish(str('Team3,multi21,0,XR58'))
    
    def image_callback(self, msg):
        try:
            # Convert the image message to a cv2 object
            camera_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.plate_detection_pub.publish(msg)

        if (self.isrecording == True and self.frame_count % 20 == 0):
            self.record_frames_states(camera_image)
        
        if (self.autopilot == True):
            self.drive_with_autopilot(camera_image)

        self.frame_count += 1
        # cv2.imshow("Camera Feed", camera_image)
        # cv2.waitKey(1)

        #Check to stop timer after 30 seconds
        if (time.time() - self.start_time > 30 and time.time() - self.start_time < 31):
            self.license_plate_pub.publish(str('Team3,multi21,-1,XR58'))

    def velocity_callback(self, msg):
        #press t to start/stop recording 
        if (msg.linear.z > 0):
            self.recording_count += 1
            if (self.recording_count % 2 == 0):
                self.isrecording = True
            else:
                self.isrecording = False
            print("Recording {}".format(self.isrecording))
        
        #press b to start/stop autopilot
        if (msg.linear.z < 0):
            self.autopilot = not self.autopilot
            print("Autopilot {}".format(self.autopilot))

        self.xspeed = msg.linear.x
        self.zang = msg.angular.z

    def record_frames_states(self, camera_image):
        image_name = f"Plate_{(self.recording_count/2)+1}_{time.time()}.jpg"
        cv2.imwrite(os.path.join(MASKING_PATH, image_name), camera_image)
        print("Recording frame {}".format(image_name))

    def drive_with_autopilot(self, camera_image):
        camera_image = cv2.resize(camera_image, (0,0), fx=0.2, fy=0.2)
        camera_image = camera_image/255.
        camera_image = camera_image.reshape((1, 144, 256, 3))
        
        predicted_actions = self.driving_model.predict(camera_image)
        action = np.argmax(predicted_actions)
        cmd_vel_msg = Twist()
        if (action == 0): #drive forward
            cmd_vel_msg.linear.x = 0.3
            cmd_vel_msg.angular.z = 0
        elif(action == 1): #turn left
            cmd_vel_msg.linear.x = 0
            cmd_vel_msg.angular.z = 1.
        else:
            cmd_vel_msg.linear.x = 0
            cmd_vel_msg.angular.z = -1.
        self.cmd_vel_pub.publish(cmd_vel_msg)


        
                

if __name__ =='__main__':
    rospy.init_node('camera_velocity_nodes')
    controller_nodes =  Controller()
    rospy.spin()