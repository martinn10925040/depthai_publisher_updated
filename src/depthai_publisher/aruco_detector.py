#!/usr/bin/env python3

import cv2
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import os  # library for executing shell commands
from std_msgs.msg import Float32MultiArray
from collections import defaultdict, deque


class ArucoDetector():
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    aruco_params = cv2.aruco.DetectorParameters_create()

    #frame_sub_topic = '/processor_node/image/compressed'

    def __init__(self):
        rospy.loginfo("Initialising Aruco Detector class...")

        # Marker detection publisher
        self.aruco_pub = rospy.Publisher(
            '/processed_aruco/image/compressed', CompressedImage, queue_size=10)
        rospy.loginfo("Publisher '/processed_aruco/image/compressed' initialised")

        # ArUco POSE estimator publisher
        self.aruco_pub_pose = rospy.Publisher(
            '/aruco_pose', Float32MultiArray, queue_size=10)
        rospy.loginfo("Publisher '/aruco_pose' initialised for Autopilot Integration")
        
        # Initialize CvBridge for converting ROS images to OpenCV format
        self.br = CvBridge()
        rospy.loginfo("CvBridge initialised")        
        self.image_sub = rospy.Subscriber('/camera/image/compressed', CompressedImage, self.img_callback)
        rospy.loginfo(f"Subscriber to topic '/camera/image/compressed' initialised")

        if not rospy.is_shutdown():  # To check if ROS is shutting down
            self.frame_sub = rospy.Subscriber(
                '/processor_node/image/compressed', CompressedImage, self.img_callback) 
        rospy.loginfo(f"Subscriber to topic '/processor_node/image/compressed' initialised")

        # Keep Unique IDs
        self.published_aruco_ids = set()
        # Store last 10 sets of coordinates for each marker
        self.coordinate_buffers = defaultdict(lambda: deque(maxlen=10)) 

    def img_callback(self, msg_in):
        try:
            frame = self.br.compressed_imgmsg_to_cv2(msg_in)  # Convert ROS (compressed) image to OpenCV format
        except CvBridgeError as e:
            rospy.logerr(f"Error converting image: {e}")  # Log error if conversion fails
            return

        aruco = self.find_aruco(frame)  # Search for marker in the frame
        self.publish_to_ros(aruco)  # Publish frame to ROS

    def find_aruco(self, frame):
        (corners, ids, _) = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.aruco_params)  # Search for markers in the frame

        if len(corners) > 0:  # If marker detected, flatten array of marker ID
            ids = ids.flatten()

            for (marker_corner, marker_ID) in zip(corners, ids):
                # ArUco Bounding Box 
                corners = marker_corner.reshape((4, 2))
                (top_left, top_right, bottom_right, bottom_left) = corners  # Unpack the four pairs of corner points
                
                # Draw lines around the marker
                cv2.line(frame, tuple(map(int, top_left)), tuple(map(int, top_right)), (0, 255, 0), 3)
                cv2.line(frame, tuple(map(int, top_right)), tuple(map(int, bottom_right)), (0, 255, 0), 3)
                cv2.line(frame, tuple(map(int, bottom_right)), tuple(map(int, bottom_left)), (0, 255, 0), 3)
                cv2.line(frame, tuple(map(int, bottom_left)), tuple(map(int, top_left)), (0, 255, 0), 3)
                
                # Display Marker ID and add text on frame
                cv2.putText(frame, str(marker_ID), (int(top_left[0]), int(top_right[1]) - 15),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
                
                self.coordinate_buffers[marker_ID].append(corners.flatten())

                #  and marker_ID not in self.published_aruco_ids

                if len(self.coordinate_buffers[marker_ID]) == 10 and marker_ID not in self.published_aruco_ids:
                    # Compute average coordinates
                    avg_corners = np.mean(self.coordinate_buffers[marker_ID], axis=0).reshape((4,2))

                    #Publish average coordinates
                    aruco_detection_msg = Float32MultiArray()
                    aruco_detection_msg.data = [float(marker_ID)] + [coord for point in avg_corners for coord in point]

                    self.aruco_pub_pose.publish(aruco_detection_msg)
                    rospy.loginfo("Published ArUco Identification and BBox corners: {}".format(aruco_detection_msg.data))

                    # Speak the marker Id using espeak
                    os.system(f"espeak 'Detected ArUco Marker {marker_ID}'")  # Use espeak to speak the ID
                    rospy.loginfo("Aruco detected, ID: {}".format(marker_ID))

                    self.published_aruco_ids.add(marker_ID)
                    self.coordinate_buffers[marker_ID].clear()

        return frame
    
    def publish_to_ros(self, frame):
        msg_out = CompressedImage()  # Create a ROS message 
        msg_out.header.stamp = rospy.Time.now()  # Sets timestamp of message header to current time
        msg_out.format = "jpeg"  # Makes sure format is set to jpeg format
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tobytes()  # Compresses frame to jpeg form
        self.aruco_pub.publish(msg_out)  # Publishes msg_out to ROS topic - our topic name

# Entry point of the program
def main():
    rospy.init_node('EGH450_vision', anonymous=True)  # Initializes a new ROS node
    rospy.loginfo("Node 'ArUco Detected' started")
    rospy.loginfo("Processing images...")

    aruco_detect = ArucoDetector()  # Run specified class
    rospy.spin()  # Keeps the node active and running

if __name__ == "__main__":
    main()