#!/usr/bin/env python3
import cv2
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import numpy as np 
import pyttsx3
import threading
import queue

class ArucoDetector():
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    aruco_params = cv2.aruco.DetectorParameters_create()

    frame_sub_topic = '/processor_node/image/compressed'

    def __init__(self):
        rospy.loginfo("Initialising Aruco Detector class...")

        self.aruco_pub = rospy.Publisher(
            '/processed_aruco/image/compressed', CompressedImage, queue_size=2)
        rospy.loginfo("Publisher '/processed_aruco/image/compressed' initialised")
        
        self.br = CvBridge()
        rospy.loginfo("CvBridge initialised")

        if not rospy.is_shutdown():
            self.frame_sub = rospy.Subscriber(
                self.frame_sub_topic, CompressedImage, self.img_callback)
            
        rospy.loginfo(f"Subscriber to topic '{self.frame_sub_topic}' initialised")

        # Set Up speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 165)	#speed
        self.engine.setProperty("pitch", 100)   #pitch

        # Queue and thread
        self.speech_queue = queue.Queue()
        self.speech_thread = threading.Thread(target = self.process_speech_queue)
        self.speech_thread.daemon = True
        self.speech_thread.start()

        # Keep Unique IDs
        self.detected_marker_ids = set()

    def img_callback(self, msg_in):
        #rospy.loginfo("Received image message")
        try:
            frame = self.br.compressed_imgmsg_to_cv2(msg_in)
            #rospy.loginfo("Converted compressed image to openCV format")
        except CvBridgeError as e:
            rospy.logerr(f"Error converting image: {e}")

        aruco = self.find_aruco(frame)
        self.publish_to_ros(aruco)


    def find_aruco(self, frame):
        (corners, ids, _) = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.aruco_params)

        if len(corners) > 0:
            ids = ids.flatten()

            new_ids = []
            for (marker_corner, marker_ID) in zip(corners, ids):
                corners = marker_corner.reshape((4, 2))
                (top_left, top_right, bottom_right, bottom_left) = corners

                top_right = (int(top_right[0]), int(top_right[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
                top_left = (int(top_left[0]), int(top_left[1]))

                cv2.line(frame, top_left, top_right, (0, 255, 0), 3)
                cv2.line(frame, top_right, bottom_right, (0, 255, 0), 3)
                cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 3)
                cv2.line(frame, bottom_left, top_left, (0, 255, 0), 3)
                
        # Display Marker ID
                rospy.loginfo("Aruco detected, ID: {}".format(marker_ID))
                cv2.putText(frame, str(
                    marker_ID), (top_left[0], top_right[1] - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

                # Unique IDS
                if marker_ID not in self.detected_marker_ids:
                    new_ids.append(marker_ID)
                    self.detected_marker_ids.add(marker_ID)

	      # Speak the marker Id in a separate threaad
            # Queue marker ID
            if new_ids:
                self.speech_queue.put(new_ids)
                rospy.loginfo("Queued IDs for speech")

        return frame
    
    def process_speech_queue(self):
        rospy.loginfo("Initiate speech engine")

        while not rospy.is_shutdown():
            try:
                marker_ids = self.speech_queue.get(timeout=1)

                for marker_ID in marker_ids:
                    self.engine.say(f"Detected ArUco Marker: {marker_ID}")
                self.engine.runAndWait()  # Wait for speech to finish
                self.speech_queue.task_done()
            except queue.Empty:
                continue

    def publish_to_ros(self, frame):
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
        self.aruco_pub.publish(msg_out)

def main():
    rospy.init_node('EGB349_vision', anonymous=True)
    rospy.loginfo("Node 'EGB450 vision' started")
    rospy.loginfo("Processing images...")

    aruco_detect = ArucoDetector()
    rospy.spin()

if __name__ == "__main__":
    main()
