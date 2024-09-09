#!/usr/bin/env python3
import cv2
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import os  # library for executing shell commands


class ArucoDetector():
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    aruco_params = cv2.aruco.DetectorParameters_create()

    frame_sub_topic = '/processor_node/image/compressed'

    def __init__(self):
        rospy.loginfo("Initialising Aruco Detector class...")

        self.aruco_pub = rospy.Publisher(
            '/processed_aruco/image/compressed', CompressedImage, queue_size=20)
        rospy.loginfo("Publisher '/processed_aruco/image/compressed' initialised")
        
        # Initialize CvBridge for converting ROS images to OpenCV format
        self.br = CvBridge()
        rospy.loginfo("CvBridge initialised")

        if not rospy.is_shutdown():  # To check if ROS is shutting down
            self.frame_sub = rospy.Subscriber(
                self.frame_sub_topic, CompressedImage, self.img_callback)
            
        rospy.loginfo(f"Subscriber to topic '{self.frame_sub_topic}' initialised")

        # Keep Unique IDs
        self.detected_marker_ids = set()

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

            new_ids = []  # To store new detected ids
            for (marker_corner, marker_ID) in zip(corners, ids):
                corners = marker_corner.reshape((4, 2))
                (top_left, top_right, bottom_right, bottom_left) = corners  # Unpack the four pairs of corner points
                
                # Converts corner points to integer (x,y) coordinates 
                top_right = (int(top_right[0]), int(top_right[1])) 
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
                top_left = (int(top_left[0]), int(top_left[1]))

                # Draw lines around the marker
                cv2.line(frame, top_left, top_right, (0, 255, 0), 3)
                cv2.line(frame, top_right, bottom_right, (0, 255, 0), 3)
                cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 3)
                cv2.line(frame, bottom_left, top_left, (0, 255, 0), 3)

                # Display Marker ID and add text on frame
                cv2.putText(frame, str(marker_ID), (top_left[0], top_right[1] - 15),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

                # To add new IDs to list of existing detected IDs
                if marker_ID not in self.detected_marker_ids:
                    new_ids.append(marker_ID)  # Append new ID
                    self.detected_marker_ids.add(marker_ID)

            # Speak the marker Id using espeak
            if new_ids:
                for marker_ID in new_ids:
                    os.system(f"espeak 'Detected ArUco Marker {marker_ID}'")  # Use espeak to speak the ID
                rospy.loginfo("Aruco detected, ID: {}".format(marker_ID))

        return frame
    
    def publish_to_ros(self, frame):
        msg_out = CompressedImage()  # Create a ROS message 
        msg_out.header.stamp = rospy.Time.now()  # Sets timestamp of message header to current time
        msg_out.format = "jpeg"  # Makes sure format is set to jpeg format
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tobytes()  # Compresses frame to jpeg form
        self.aruco_pub.publish(msg_out)  # Publishes msg_out to ROS topic - our topic name

# Entry point of the program
def main():
    rospy.init_node('processor_node', anonymous=True)  # Initializes a new ROS node
    rospy.loginfo("Node 'EGH450 vision' started")
    rospy.loginfo("Processing images...")

    aruco_detect = ArucoDetector()  # Run specified class
    rospy.spin()  # Keeps the node active and running

if __name__ == "__main__":
    main()
