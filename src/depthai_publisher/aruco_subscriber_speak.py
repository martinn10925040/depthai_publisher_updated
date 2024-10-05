# aruco subscriber with new (Cassie's) speak function implemented but not tested

#!/usr/bin/env python3

import cv2
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from std_msgs.msg import Float32MultiArray, String
import threading


# Declare pub_speaker as a global variable
pub_speaker = None  

class ArucoDetector():
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    aruco_params = cv2.aruco.DetectorParameters_create()

    frame_sub_topic = '/depthai_node/image/compressed'

    def __init__(self):
        self.image_pub = rospy.Publisher(
            '/processed_aruco/image/compressed', CompressedImage, queue_size=10)  # Publisher for processed images

        self.aruco_detection_pub = rospy.Publisher('/aruco_detection', Float32MultiArray, queue_size=10)  # Publisher for marker data

        self.br = CvBridge()
        self.frame = None  # Shared resource between threads
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()  # Event to signal the arrival of a new frame

        # Dictionary to store accumulated corner data per marker ID
        self.marker_corners = {}  # Key: marker_ID, value: dict with sums, count, published flag

        if not rospy.is_shutdown():
            self.frame_sub = rospy.Subscriber(
                self.frame_sub_topic, CompressedImage, self.img_callback, queue_size=1)

        # Start the processing thread
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()


    def img_callback(self, msg_in):
        try:
            frame = self.br.compressed_imgmsg_to_cv2(msg_in)
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Acquire lock and update the shared frame
        with self.lock:
            self.frame = frame
            self.new_frame_event.set()  # Signal that a new frame is available

    def process_frames(self):
        while not rospy.is_shutdown():
            # Wait until a new frame is available
            self.new_frame_event.wait()
            with self.lock:
                frame = self.frame.copy()
                self.new_frame_event.clear()  # Reset the event

            # Process the frame
            processed_frame = self.find_aruco(frame)

            # Publish the processed image
            self.publish_to_ros(processed_frame)

    def find_aruco(self, frame):
        # Convert to grayscale (optional, can speed up detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            ids = ids.flatten()

            for marker_corner, marker_ID in zip(corners, ids):
                # Draw the bounding box around the detected ArUco marker
                corners = marker_corner.reshape((4, 2))
                (top_left, top_right, bottom_right, bottom_left) = corners

                # Draw marker edges
                cv2.line(frame, tuple(map(int, top_left)), tuple(map(int, top_right)), (0, 255, 0), 2)
                cv2.line(frame, tuple(map(int, top_right)), tuple(map(int, bottom_right)), (0, 255, 0), 2)
                cv2.line(frame, tuple(map(int, bottom_right)), tuple(map(int, bottom_left)), (0, 255, 0), 2)
                cv2.line(frame, tuple(map(int, bottom_left)), tuple(map(int, top_left)), (0, 255, 0), 2)

                # Annotate the frame with the detected ID
                cv2.putText(frame, str(marker_ID), (int(top_left[0]), int(top_left[1]) - 15),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

                # Initialize accumulation if not already done
                if marker_ID not in self.marker_corners:
                    self.marker_corners[marker_ID] = {
                        'sum_tl_x': 0, 'sum_tl_y': 0,
                        'sum_tr_x': 0, 'sum_tr_y': 0,
                        'sum_br_x': 0, 'sum_br_y': 0,
                        'sum_bl_x': 0, 'sum_bl_y': 0,
                        'count': 0,
                        'published': False
                    }

                # Check if this marker has already been published
                if self.marker_corners[marker_ID]['published']:
                    rospy.loginfo(f"Marker {marker_ID} has already been published. Skipping further detections.")
                    continue

                # Accumulate corner coordinates
                self.marker_corners[marker_ID]['sum_tl_x'] += top_left[0]
                self.marker_corners[marker_ID]['sum_tl_y'] += top_left[1]
                self.marker_corners[marker_ID]['sum_tr_x'] += top_right[0]
                self.marker_corners[marker_ID]['sum_tr_y'] += top_right[1]
                self.marker_corners[marker_ID]['sum_br_x'] += bottom_right[0]
                self.marker_corners[marker_ID]['sum_br_y'] += bottom_right[1]
                self.marker_corners[marker_ID]['sum_bl_x'] += bottom_left[0]
                self.marker_corners[marker_ID]['sum_bl_y'] += bottom_left[1]
                self.marker_corners[marker_ID]['count'] += 1

                # Only publish after 10 detections
                if self.marker_corners[marker_ID]['count'] == 10:
                    # Calculate the average coordinates for each corner
                    avg_tl_x = self.marker_corners[marker_ID]['sum_tl_x'] / 10
                    avg_tl_y = self.marker_corners[marker_ID]['sum_tl_y'] / 10
                    avg_tr_x = self.marker_corners[marker_ID]['sum_tr_x'] / 10
                    avg_tr_y = self.marker_corners[marker_ID]['sum_tr_y'] / 10
                    avg_br_x = self.marker_corners[marker_ID]['sum_br_x'] / 10
                    avg_br_y = self.marker_corners[marker_ID]['sum_br_y'] / 10
                    avg_bl_x = self.marker_corners[marker_ID]['sum_bl_x'] / 10
                    avg_bl_y = self.marker_corners[marker_ID]['sum_bl_y'] / 10

                    # Log the averaged corner coordinates
                    rospy.loginfo(f"Averaged corners for Marker {marker_ID}: "
                                  f"Top-left({avg_tl_x}, {avg_tl_y}), Top-right({avg_tr_x}, {avg_tr_y}), "
                                  f"Bottom-right({avg_br_x}, {avg_br_y}), Bottom-left({avg_bl_x}, {avg_bl_y})")

                    # Create a Float32MultiArray message for the combined ID and averaged corners
                    detection_msg = Float32MultiArray()
                    # Create a list with marker ID and averaged corner coordinates
                    detection_msg.data = [float(marker_ID),
                                          avg_tl_x, avg_tl_y,
                                          avg_tr_x, avg_tr_y,
                                          avg_br_x, avg_br_y,
                                          avg_bl_x, avg_bl_y]

                    # Publish the combined message
                    self.aruco_detection_pub.publish(detection_msg)
                    rospy.loginfo(f"Published averaged Aruco ID and corners: {detection_msg.data}")

                    # New Speak Functionality: Construct and publish the message to be spoken
                    speaker_msg = String()
                    speaker_msg.data = f"Detected Marker ID: {marker_ID}, Position X: {avg_tl_x:.2f}, Y: {avg_tl_y:.2f}"
                    pub_speaker.publish(speaker_msg)

                    # Call the speak function to announce the detected marker ID
                    # self.speak(f"Detected Marker ID: {marker_ID}")

                    # Mark as published
                    self.marker_corners[marker_ID]['published'] = True

        return frame

    def publish_to_ros(self, frame):
        # Publishing the processed image
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tobytes()

        self.image_pub.publish(msg_out)  # Publish the processed image

def main():
    global pub_speaker
    # Initialize speak_pub publisher
    pub_speaker = rospy.Publisher('spoken_text', String, queue_size=10)

    rospy.init_node('EGB349_vision', anonymous=True)
    rospy.loginfo("Processing images...")
    aruco_detect = ArucoDetector()

    rospy.spin()

if __name__ == '__main__':
    main()
