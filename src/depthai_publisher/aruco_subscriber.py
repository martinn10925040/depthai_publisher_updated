#!/usr/bin/env python3

import cv2
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from std_msgs.msg import Float32MultiArray
import threading

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
                    #rospy.loginfo(f"Marker {marker_ID} has already been published. Skipping further detections.")
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
    rospy.init_node('EGB349_vision', anonymous=True)
    rospy.loginfo("Processing images...")

    aruco_detect = ArucoDetector()

    rospy.spin()

if __name__ == '__main__':
    main()

# import cv2
# import rospy
# from sensor_msgs.msg import CompressedImage
# from cv_bridge import CvBridge, CvBridgeError
# import numpy as np
# from std_msgs.msg import Float32MultiArray
# import threading
# from collections import defaultdict, deque

# class ArucoDetector():
#     aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
#     aruco_params = cv2.aruco.DetectorParameters_create()

#     frame_sub_topic = '/depthai_node/image/compressed'

#     def __init__(self):
#         self.image_pub = rospy.Publisher(
#             '/processed_aruco/image/compressed', CompressedImage, queue_size=10)  # Publisher for processed images

#         # Publisher for raw marker detections
#         self.aruco_detection_raw_pub = rospy.Publisher('/aruco_detections_raw', Float32MultiArray, queue_size=50)

#         # Publisher for averaged marker data
#         self.aruco_detection_pub = rospy.Publisher('/aruco_detection', Float32MultiArray, queue_size=50)  # Existing publisher

#         self.br = CvBridge()
#         self.frame = None  # Shared resource between threads
#         self.lock = threading.Lock()
#         self.new_frame_event = threading.Event()  # Event to signal the arrival of a new frame

#         # Initialize a dictionary to store accumulated corner data per marker ID
#         self.marker_corners = defaultdict(lambda: {'corners': [], 'count': 0})

#         if not rospy.is_shutdown():
#             self.frame_sub = rospy.Subscriber(
#                 self.frame_sub_topic, CompressedImage, self.img_callback, queue_size=1)

#         # Subscriber to receive raw marker detections
#         self.marker_detection_sub = rospy.Subscriber(
#             '/aruco_detections_raw', Float32MultiArray, self.marker_detection_callback)

#         # Start the processing thread
#         self.processing_thread = threading.Thread(target=self.process_frames)
#         self.processing_thread.daemon = True
#         self.processing_thread.start()

#     def img_callback(self, msg_in):
#         try:
#             frame = self.br.compressed_imgmsg_to_cv2(msg_in)
#         except CvBridgeError as e:
#             rospy.logerr(e)
#             return

#         # Acquire lock and update the shared frame
#         with self.lock:
#             self.frame = frame
#             self.new_frame_event.set()  # Signal that a new frame is available

#     def process_frames(self):
#         while not rospy.is_shutdown():
#             # Wait until a new frame is available
#             self.new_frame_event.wait()
#             with self.lock:
#                 frame = self.frame.copy()
#                 self.new_frame_event.clear()  # Reset the event

#             # Process the frame
#             processed_frame = self.find_aruco(frame)

#             # Publish the processed image
#             self.publish_to_ros(processed_frame)

#     def find_aruco(self, frame):
#         # Convert to grayscale (optional, can speed up detection)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Detect markers
#         corners, ids, _ = cv2.aruco.detectMarkers(
#             gray, self.aruco_dict, parameters=self.aruco_params)

#         if ids is not None:
#             ids = ids.flatten()

#             for marker_corner, marker_ID in zip(corners, ids):
#                 # Draw the bounding box around the detected ArUco marker
#                 corners = marker_corner.reshape((4, 2))
#                 (top_left, top_right, bottom_right, bottom_left) = corners

#                 # Draw marker edges
#                 cv2.line(frame, tuple(map(int, top_left)), tuple(map(int, top_right)), (0, 255, 0), 2)
#                 cv2.line(frame, tuple(map(int, top_right)), tuple(map(int, bottom_right)), (0, 255, 0), 2)
#                 cv2.line(frame, tuple(map(int, bottom_right)), tuple(map(int, bottom_left)), (0, 255, 0), 2)
#                 cv2.line(frame, tuple(map(int, bottom_left)), tuple(map(int, top_left)), (0, 255, 0), 2)

#                 # Annotate the frame with the detected ID
#                 cv2.putText(frame, str(marker_ID), (int(top_left[0]), int(top_left[1]) - 15),
#                             cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

#                 # Publish the raw detection data
#                 detection_msg = Float32MultiArray()
#                 detection_msg.data = [float(marker_ID)] + corners.flatten().tolist()
#                 self.aruco_detection_raw_pub.publish(detection_msg)

#         return frame

#     def marker_detection_callback(self, msg):
#         data = msg.data
#         marker_ID = int(data[0])
#         corners = np.array(data[1:]).reshape((4, 2))

#         # Accumulate corner coordinates
#         self.marker_corners[marker_ID]['corners'].append(corners)
#         self.marker_corners[marker_ID]['count'] += 1

#         if self.marker_corners[marker_ID]['count'] == 10:
#             # Compute average corners
#             all_corners = np.array(self.marker_corners[marker_ID]['corners'])
#             avg_corners = np.mean(all_corners, axis=0)

#             # Log the averaged corner coordinates
#             rospy.loginfo(f"Averaged corners for Marker {marker_ID}: {avg_corners}")

#             # Create a Float32MultiArray message for the averaged corners
#             averaged_msg = Float32MultiArray()
#             averaged_msg.data = [float(marker_ID)] + avg_corners.flatten().tolist()

#             # Publish the averaged data
#             self.aruco_detection_pub.publish(averaged_msg)
#             rospy.loginfo(f"Published averaged Aruco ID and corners: {averaged_msg.data}")

#             # Reset the accumulation for this marker
#             self.marker_corners[marker_ID]['corners'].clear()
#             self.marker_corners[marker_ID]['count'] = 0

#     def publish_to_ros(self, frame):
#         # Publishing the processed image
#         msg_out = CompressedImage()
#         msg_out.header.stamp = rospy.Time.now()
#         msg_out.format = "jpeg"
#         msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tobytes()

#         self.image_pub.publish(msg_out)  # Publish the processed image

# def main():
#     rospy.init_node('EGB349_vision', anonymous=True)
#     rospy.loginfo("Processing images...")

#     aruco_detect = ArucoDetector()

#     rospy.spin()

# if __name__ == '__main__':
#     main()



# import cv2
# import rospy
# from sensor_msgs.msg import CompressedImage
# from cv_bridge import CvBridge, CvBridgeError
# from std_msgs.msg import String
# import subprocess
# import numpy as np
# import threading
# import time

# class ArucoDetector():
#     aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
#     aruco_params = cv2.aruco.DetectorParameters_create()

#     frame_sub_topic = '/depthai_node/image/compressed'

#     def __init__(self):
#         self.aruco_pub = rospy.Publisher(
#             '/processed_aruco/image/compressed', CompressedImage, queue_size=10)
        
#         self.br = CvBridge()

#         # A set to store detected marker IDs
#         self.detected_ids = set()

#         # A dictionary to keep track of last announcement time for each marker
#         self.marker_announce_times = {}

#         # Time delay (in seconds) to wait before announcing the same marker again
#         self.announce_delay = 30  # Adjust this value if you want a shorter or longer wait

#         if not rospy.is_shutdown():
#             self.frame_sub = rospy.Subscriber(
#                 self.frame_sub_topic, CompressedImage, self.img_callback)

#     def speak(self, text):
#         global speak_pub
#         # Run the espeak subprocess asynchronously to avoid blocking
#         threading.Thread(target=subprocess.run, args=(['espeak', text],)).start()
#         if speak_pub:
#             speak_pub.publish(text)  # Publish the text to the ROS topic

#     def img_callback(self, msg_in):
#         try:
#             frame = self.br.compressed_imgmsg_to_cv2(msg_in)
#         except CvBridgeError as e:
#             rospy.logerr(e)

#         aruco = self.find_aruco(frame)
#         self.publish_to_ros(aruco)

#     def find_aruco(self, frame):
#         (corners, ids, _) = cv2.aruco.detectMarkers(
#             frame, self.aruco_dict, parameters=self.aruco_params)

#         if len(corners) > 0:
#             ids = ids.flatten()

#             for (marker_corner, marker_ID) in zip(corners, ids):
#                 current_time = time.time()

#                 # Convert the corners to integer pixel coordinates
#                 corners = marker_corner.reshape((4, 2))
#                 top_left, top_right, bottom_right, bottom_left = corners.astype(int)

#                 # Draw lines between the corners to form a frame
#                 cv2.line(frame, tuple(top_left), tuple(top_right), (0, 255, 0), 2)  # Top-left to top-right
#                 cv2.line(frame, tuple(top_right), tuple(bottom_right), (0, 255, 0), 2)  # Top-right to bottom-right
#                 cv2.line(frame, tuple(bottom_right), tuple(bottom_left), (0, 255, 0), 2)  # Bottom-right to bottom-left
#                 cv2.line(frame, tuple(bottom_left), tuple(top_left), (0, 255, 0), 2)  # Bottom-left to top-left

#                 # Draw the marker ID on the frame
#                 cv2.putText(frame, str(marker_ID), (top_left[0], top_left[1] - 10),
#                             cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

#                 # Check if this marker has already been detected and throttled for re-announcement
#                 if marker_ID not in self.marker_announce_times or \
#                    (current_time - self.marker_announce_times[marker_ID]) > self.announce_delay:
                    
#                     # Log and speak the detection
#                     rospy.loginfo(f"Aruco detected, ID: {marker_ID}")
#                     message =  "Aruco detected, ID: " + str(marker_ID)
#                     #self.speak(f"Aruco detected, ID: {marker_ID}")
#                     self.speak(message)
#                     # Update the last announce time for this marker
#                     self.marker_announce_times[marker_ID] = current_time

#         return frame

#     def publish_to_ros(self, frame):
#         msg_out = CompressedImage()
#         msg_out.header.stamp = rospy.Time.now()
#         msg_out.format = "jpeg"
#         msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()

#         self.aruco_pub.publish(msg_out)


# def main():
#     global speak_pub
#     # Set up the publisher for spoken text on the 'spoken_text' topic
#     speak_pub = rospy.Publisher('spoken_text', String, queue_size=10)

#     rospy.init_node('EGB349_vision', anonymous=True)
#     rospy.loginfo("Processing images...")

#     aruco_detect = ArucoDetector()

#     rospy.spin()


# if __name__ == "__main__":
# =======
# # 
# >>>>>>> 0c6d320e02ba0f42adec8d15bcbafdd02eb96e99

# import cv2
# import rospy
# from sensor_msgs.msg import CompressedImage
# from cv_bridge import CvBridge, CvBridgeError
# import numpy as np
# from std_msgs.msg import Float32MultiArray
# import threading
# <<<<<<< HEAD
# =======
# from collections import defaultdict, deque
# >>>>>>> 0c6d320e02ba0f42adec8d15bcbafdd02eb96e99

# class ArucoDetector():
#     aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
#     aruco_params = cv2.aruco.DetectorParameters_create()

#     frame_sub_topic = '/depthai_node/image/compressed'

#     def __init__(self):
#         self.image_pub = rospy.Publisher(
#             '/processed_aruco/image/compressed', CompressedImage, queue_size=10)  # Publisher for processed images

# <<<<<<< HEAD
#         self.aruco_detection_pub = rospy.Publisher('/aruco_detection', Float32MultiArray, queue_size=10)  # Publisher for marker data
# =======
#         # Publisher for raw marker detections
#         self.aruco_detection_raw_pub = rospy.Publisher('/aruco_detections_raw', Float32MultiArray, queue_size=10)

#         # Publisher for averaged marker data
#         self.aruco_detection_pub = rospy.Publisher('/aruco_detection', Float32MultiArray, queue_size=10)  # Existing publisher
# >>>>>>> 0c6d320e02ba0f42adec8d15bcbafdd02eb96e99

#         self.br = CvBridge()
#         self.frame = None  # Shared resource between threads
#         self.lock = threading.Lock()
#         self.new_frame_event = threading.Event()  # Event to signal the arrival of a new frame

# <<<<<<< HEAD
#         # Dictionary to store accumulated corner data per marker ID
#         self.marker_corners = {}  # Key: marker_ID, value: dict with sums, count, published flag
# =======
#         # Initialize a dictionary to store accumulated corner data per marker ID
#         self.marker_corners = defaultdict(lambda: {'corners': [], 'count': 0})
# >>>>>>> 0c6d320e02ba0f42adec8d15bcbafdd02eb96e99

#         if not rospy.is_shutdown():
#             self.frame_sub = rospy.Subscriber(
#                 self.frame_sub_topic, CompressedImage, self.img_callback, queue_size=1)

# <<<<<<< HEAD
# =======
#         # Subscriber to receive raw marker detections
#         self.marker_detection_sub = rospy.Subscriber(
#             '/aruco_detections_raw', Float32MultiArray, self.marker_detection_callback)

# >>>>>>> 0c6d320e02ba0f42adec8d15bcbafdd02eb96e99
#         # Start the processing thread
#         self.processing_thread = threading.Thread(target=self.process_frames)
#         self.processing_thread.daemon = True
#         self.processing_thread.start()

#     def img_callback(self, msg_in):
#         try:
#             frame = self.br.compressed_imgmsg_to_cv2(msg_in)
#         except CvBridgeError as e:
#             rospy.logerr(e)
#             return

#         # Acquire lock and update the shared frame
#         with self.lock:
#             self.frame = frame
#             self.new_frame_event.set()  # Signal that a new frame is available

#     def process_frames(self):
#         while not rospy.is_shutdown():
#             # Wait until a new frame is available
#             self.new_frame_event.wait()
#             with self.lock:
#                 frame = self.frame.copy()
#                 self.new_frame_event.clear()  # Reset the event

#             # Process the frame
#             processed_frame = self.find_aruco(frame)

#             # Publish the processed image
#             self.publish_to_ros(processed_frame)

#     def find_aruco(self, frame):
#         # Convert to grayscale (optional, can speed up detection)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Detect markers
#         corners, ids, _ = cv2.aruco.detectMarkers(
#             gray, self.aruco_dict, parameters=self.aruco_params)

#         if ids is not None:
#             ids = ids.flatten()

#             for marker_corner, marker_ID in zip(corners, ids):
#                 # Draw the bounding box around the detected ArUco marker
#                 corners = marker_corner.reshape((4, 2))
#                 (top_left, top_right, bottom_right, bottom_left) = corners

#                 # Draw marker edges
#                 cv2.line(frame, tuple(map(int, top_left)), tuple(map(int, top_right)), (0, 255, 0), 2)
#                 cv2.line(frame, tuple(map(int, top_right)), tuple(map(int, bottom_right)), (0, 255, 0), 2)
#                 cv2.line(frame, tuple(map(int, bottom_right)), tuple(map(int, bottom_left)), (0, 255, 0), 2)
#                 cv2.line(frame, tuple(map(int, bottom_left)), tuple(map(int, top_left)), (0, 255, 0), 2)

#                 # Annotate the frame with the detected ID
#                 cv2.putText(frame, str(marker_ID), (int(top_left[0]), int(top_left[1]) - 15),
#                             cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

# <<<<<<< HEAD
#                 # Initialize accumulation if not already done
#                 if marker_ID not in self.marker_corners:
#                     self.marker_corners[marker_ID] = {
#                         'sum_tl_x': 0, 'sum_tl_y': 0,
#                         'sum_tr_x': 0, 'sum_tr_y': 0,
#                         'sum_br_x': 0, 'sum_br_y': 0,
#                         'sum_bl_x': 0, 'sum_bl_y': 0,
#                         'count': 0,
#                         'published': False
#                     }

#                 # Check if this marker has already been published
#                 if self.marker_corners[marker_ID]['published']:
#                     rospy.loginfo(f"Marker {marker_ID} has already been published. Skipping further detections.")
#                     continue

#                 # Accumulate corner coordinates
#                 self.marker_corners[marker_ID]['sum_tl_x'] += top_left[0]
#                 self.marker_corners[marker_ID]['sum_tl_y'] += top_left[1]
#                 self.marker_corners[marker_ID]['sum_tr_x'] += top_right[0]
#                 self.marker_corners[marker_ID]['sum_tr_y'] += top_right[1]
#                 self.marker_corners[marker_ID]['sum_br_x'] += bottom_right[0]
#                 self.marker_corners[marker_ID]['sum_br_y'] += bottom_right[1]
#                 self.marker_corners[marker_ID]['sum_bl_x'] += bottom_left[0]
#                 self.marker_corners[marker_ID]['sum_bl_y'] += bottom_left[1]
#                 self.marker_corners[marker_ID]['count'] += 1

#                 # Only publish after 10 detections
#                 if self.marker_corners[marker_ID]['count'] == 10:
#                     # Calculate the average coordinates for each corner
#                     avg_tl_x = self.marker_corners[marker_ID]['sum_tl_x'] / 10
#                     avg_tl_y = self.marker_corners[marker_ID]['sum_tl_y'] / 10
#                     avg_tr_x = self.marker_corners[marker_ID]['sum_tr_x'] / 10
#                     avg_tr_y = self.marker_corners[marker_ID]['sum_tr_y'] / 10
#                     avg_br_x = self.marker_corners[marker_ID]['sum_br_x'] / 10
#                     avg_br_y = self.marker_corners[marker_ID]['sum_br_y'] / 10
#                     avg_bl_x = self.marker_corners[marker_ID]['sum_bl_x'] / 10
#                     avg_bl_y = self.marker_corners[marker_ID]['sum_bl_y'] / 10

#                     # Log the averaged corner coordinates
#                     rospy.loginfo(f"Averaged corners for Marker {marker_ID}: "
#                                   f"Top-left({avg_tl_x}, {avg_tl_y}), Top-right({avg_tr_x}, {avg_tr_y}), "
#                                   f"Bottom-right({avg_br_x}, {avg_br_y}), Bottom-left({avg_bl_x}, {avg_bl_y})")

#                     # Create a Float32MultiArray message for the combined ID and averaged corners
#                     detection_msg = Float32MultiArray()
#                     # Create a list with marker ID and averaged corner coordinates
#                     detection_msg.data = [float(marker_ID),
#                                           avg_tl_x, avg_tl_y,
#                                           avg_tr_x, avg_tr_y,
#                                           avg_br_x, avg_br_y,
#                                           avg_bl_x, avg_bl_y]

#                     # Publish the combined message
#                     self.aruco_detection_pub.publish(detection_msg)
#                     rospy.loginfo(f"Published averaged Aruco ID and corners: {detection_msg.data}")

#                     # Mark as published
#                     self.marker_corners[marker_ID]['published'] = True

#         return frame

# =======
#                 # Publish the raw detection data
#                 detection_msg = Float32MultiArray()
#                 detection_msg.data = [float(marker_ID)] + corners.flatten().tolist()
#                 self.aruco_detection_raw_pub.publish(detection_msg)

#         return frame

#     def marker_detection_callback(self, msg):
#         data = msg.data
#         marker_ID = int(data[0])
#         corners = np.array(data[1:]).reshape((4, 2))

#         # Accumulate corner coordinates
#         self.marker_corners[marker_ID]['corners'].append(corners)
#         self.marker_corners[marker_ID]['count'] += 1

#         if self.marker_corners[marker_ID]['count'] == 10:
#             # Compute average corners
#             all_corners = np.array(self.marker_corners[marker_ID]['corners'])
#             avg_corners = np.mean(all_corners, axis=0)

#             # Log the averaged corner coordinates
#             rospy.loginfo(f"Averaged corners for Marker {marker_ID}: {avg_corners}")

#             # Create a Float32MultiArray message for the averaged corners
#             averaged_msg = Float32MultiArray()
#             averaged_msg.data = [float(marker_ID)] + avg_corners.flatten().tolist()

#             # Publish the averaged data
#             self.aruco_detection_pub.publish(averaged_msg)
#             rospy.loginfo(f"Published averaged Aruco ID and corners: {averaged_msg.data}")

#             # Reset the accumulation for this marker
#             self.marker_corners[marker_ID]['corners'].clear()
#             self.marker_corners[marker_ID]['count'] = 0

# >>>>>>> 0c6d320e02ba0f42adec8d15bcbafdd02eb96e99
#     def publish_to_ros(self, frame):
#         # Publishing the processed image
#         msg_out = CompressedImage()
#         msg_out.header.stamp = rospy.Time.now()
#         msg_out.format = "jpeg"
#         msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tobytes()

#         self.image_pub.publish(msg_out)  # Publish the processed image

# def main():
#     rospy.init_node('EGB349_vision', anonymous=True)
#     rospy.loginfo("Processing images...")

#     aruco_detect = ArucoDetector()

#     rospy.spin()

# if __name__ == '__main__':

#     main()




# import cv2
# import rospy
# from sensor_msgs.msg import CompressedImage
# from cv_bridge import CvBridge, CvBridgeError
# from std_msgs.msg import String
# import subprocess
# import numpy as np
# import threading
# import time

# class ArucoDetector():
#     aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
#     aruco_params = cv2.aruco.DetectorParameters_create()

#     frame_sub_topic = '/depthai_node/image/compressed'

#     def __init__(self):
#         self.aruco_pub = rospy.Publisher(
#             '/processed_aruco/image/compressed', CompressedImage, queue_size=10)
        
#         self.br = CvBridge()

#         # A set to store detected marker IDs
#         self.detected_ids = set()

#         # A dictionary to keep track of last announcement time for each marker
#         self.marker_announce_times = {}

#         # Time delay (in seconds) to wait before announcing the same marker again
#         self.announce_delay = 30  # Adjust this value if you want a shorter or longer wait

#         # Use a threading lock to ensure thread safety
#         self.lock = threading.Lock()

#         # Image storage for processing in another thread
#         self.frame = None

#         # Thread to process ArUco markers asynchronously
#         self.detection_thread = threading.Thread(target=self.process_frame)
#         self.detection_thread.daemon = True  # Daemon thread will close when the main program ends
#         self.detection_thread.start()

#         if not rospy.is_shutdown():
#             self.frame_sub = rospy.Subscriber(
#                 self.frame_sub_topic, CompressedImage, self.img_callback)

#     def speak(self, text):
#         global speak_pub
#         # Run the espeak subprocess asynchronously to avoid blocking
#         threading.Thread(target=subprocess.run, args=(['espeak', text],)).start()
#         if speak_pub:
#             speak_pub.publish(text)  # Publish the text to the ROS topic

#     def img_callback(self, msg_in):
#         try:
#             # Convert the ROS message to a cv2 image
#             frame = self.br.compressed_imgmsg_to_cv2(msg_in)
#         except CvBridgeError as e:
#             rospy.logerr(e)
#             return

#         # Acquire the lock and store the frame for processing
#         with self.lock:
#             self.frame = frame

#     def process_frame(self):
#         while not rospy.is_shutdown():
#             # Continuously check for new frames to process
#             with self.lock:
#                 if self.frame is None:
#                     continue  # Skip if there's no new frame

#                 # Clone the frame to avoid processing the same data multiple times
#                 frame = self.frame.copy()

#             # Process the frame (detect ArUco markers)
#             aruco_frame = self.find_aruco(frame)

#             # Publish the processed frame
#             self.publish_to_ros(aruco_frame)

#             # Sleep briefly to prevent CPU overload
#             time.sleep(0.01)

#     def find_aruco(self, frame):
#         (corners, ids, _) = cv2.aruco.detectMarkers(
#             frame, self.aruco_dict, parameters=self.aruco_params)

#         if len(corners) > 0:
#             ids = ids.flatten()

#             for (marker_corner, marker_ID) in zip(corners, ids):
#                 current_time = time.time()

#                 # Convert the corners to integer pixel coordinates
#                 corners = marker_corner.reshape((4, 2))
#                 top_left, top_right, bottom_right, bottom_left = corners.astype(int)

#                 # Draw lines between the corners to form a frame
#                 cv2.line(frame, tuple(top_left), tuple(top_right), (0, 255, 0), 2)  # Top-left to top-right
#                 cv2.line(frame, tuple(top_right), tuple(bottom_right), (0, 255, 0), 2)  # Top-right to bottom-right
#                 cv2.line(frame, tuple(bottom_right), tuple(bottom_left), (0, 255, 0), 2)  # Bottom-right to bottom-left
#                 cv2.line(frame, tuple(bottom_left), tuple(top_left), (0, 255, 0), 2)  # Bottom-left to top-left

#                 # Draw the marker ID on the frame
#                 cv2.putText(frame, str(marker_ID), (top_left[0], top_left[1] - 10),
#                             cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

#                 # Check if this marker has already been detected and throttled for re-announcement
#                 if marker_ID not in self.marker_announce_times or \
#                    (current_time - self.marker_announce_times[marker_ID]) > self.announce_delay:
                    
#                     # Log the detection
#                     rospy.loginfo(f"Aruco detected, ID: {marker_ID}")
#                     message =  "Aruco detected, ID: " + str(marker_ID)

#                     # Update the last announce time for this marker
#                     self.marker_announce_times[marker_ID] = current_time

#         return frame

#     def publish_to_ros(self, frame):
#         msg_out = CompressedImage()
#         msg_out.header.stamp = rospy.Time.now()
#         msg_out.format = "jpeg"
#         msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()

#         self.aruco_pub.publish(msg_out)


# def main():
#     global speak_pub
#     # Set up the publisher for spoken text on the 'spoken_text' topic
#     speak_pub = rospy.Publisher('spoken_text', String, queue_size=10)

#     rospy.init_node('EGB349_vision', anonymous=True)
#     rospy.loginfo("Processing images...")

#     aruco_detect = ArucoDetector()

#     rospy.spin()


# if __name__ == "__main__":
#     main()

# <<<<<<< HEAD
# =======
# #!/usr/bin/env python3

# import cv2
# import rospy
# from sensor_msgs.msg import CompressedImage
# from cv_bridge import CvBridge, CvBridgeError
# import numpy as np
# from std_msgs.msg import Float32MultiArray, Int32
# import threading
# from collections import defaultdict, deque

# class ArucoDetector():
#     aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
#     aruco_params = cv2.aruco.DetectorParameters_create()

#     frame_sub_topic = '/depthai_node/image/compressed'

#     def __init__(self):
#         # Publisher for processed images
#         self.image_pub = rospy.Publisher(
#             '/processed_aruco/image/compressed', CompressedImage, queue_size=10)

#         # Publisher for initial and averaged marker data
#         self.aruco_detection_pub = rospy.Publisher(
#             '/aruco_detection', Float32MultiArray, queue_size=10)

#         self.br = CvBridge()
#         self.frame = None  # Shared resource between threads
#         self.lock = threading.Lock()
#         self.new_frame_event = threading.Event()  # Event to signal the arrival of a new frame

#         # Dictionary to store accumulated corner data per marker ID
#         self.marker_corners = defaultdict(lambda: deque(maxlen=10))

#         # Set to keep track of markers published upon first detection
#         self.published_markers = set()

#         if not rospy.is_shutdown():
#             self.frame_sub = rospy.Subscriber(
#                 self.frame_sub_topic, CompressedImage, self.img_callback, queue_size=1)

#         # Subscriber to receive averaging trigger
#         self.average_trigger_sub = rospy.Subscriber(
#             '/aruco/average_trigger', Int32, self.average_trigger_callback)

#         # Start the processing thread
#         self.processing_thread = threading.Thread(target=self.process_frames)
#         self.processing_thread.daemon = True
#         self.processing_thread.start()

#     def img_callback(self, msg_in):
#         try:
#             frame = self.br.compressed_imgmsg_to_cv2(msg_in)
#         except CvBridgeError as e:
#             rospy.logerr(e)
#             return

#         # Acquire lock and update the shared frame
#         with self.lock:
#             self.frame = frame
#             self.new_frame_event.set()  # Signal that a new frame is available

#     def process_frames(self):
#         while not rospy.is_shutdown():
#             # Wait until a new frame is available
#             self.new_frame_event.wait()
#             with self.lock:
#                 frame = self.frame.copy()
#                 self.new_frame_event.clear()  # Reset the event

#             # Process the frame
#             processed_frame = self.find_aruco(frame)

#             # Publish the processed image
#             self.publish_to_ros(processed_frame)

#     def find_aruco(self, frame):
#         # Convert to grayscale (optional, can speed up detection)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Detect markers
#         corners_list, ids, _ = cv2.aruco.detectMarkers(
#             gray, self.aruco_dict, parameters=self.aruco_params)

#         if ids is not None:
#             ids = ids.flatten()

#             for marker_corners, marker_ID in zip(corners_list, ids):
#                 # Draw the bounding box around the detected ArUco marker
#                 corners = marker_corners.reshape((4, 2))
#                 (top_left, top_right, bottom_right, bottom_left) = corners

#                 # Draw marker edges
#                 cv2.line(frame, tuple(map(int, top_left)), tuple(map(int, top_right)), (0, 255, 0), 2)
#                 cv2.line(frame, tuple(map(int, top_right)), tuple(map(int, bottom_right)), (0, 255, 0), 2)
#                 cv2.line(frame, tuple(map(int, bottom_right)), tuple(map(int, bottom_left)), (0, 255, 0), 2)
#                 cv2.line(frame, tuple(map(int, bottom_left)), tuple(map(int, top_left)), (0, 255, 0), 2)

#                 # Annotate the frame with the detected ID
#                 cv2.putText(frame, str(marker_ID), (int(top_left[0]), int(top_left[1]) - 15),
#                             cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

#                 # If marker hasn't been published yet upon first detection, publish its pose
#                 if marker_ID not in self.published_markers:
#                     detection_msg = Float32MultiArray()
#                     detection_msg.data = [float(marker_ID)] + corners.flatten().tolist()
#                     self.aruco_detection_pub.publish(detection_msg)
#                     rospy.loginfo(f"Published initial detection for Marker {marker_ID}")
#                     self.published_markers.add(marker_ID)

#                 # Accumulate corner coordinates
#                 self.marker_corners[marker_ID].append(corners)
#                 rospy.loginfo(f"Accumulated detection for Marker {marker_ID}, count: {len(self.marker_corners[marker_ID])}")

#         return frame

#     def average_trigger_callback(self, msg):
#         rospy.loginfo(f"Received trigger message: {msg.data}")
#         if msg.data == 1:
#             rospy.loginfo("Received averaging trigger")

#             # For each marker, compute average if there are detections
#             for marker_ID, corners_list in self.marker_corners.items():
#                 count = len(corners_list)
#                 if count > 0:
#                     all_corners = np.array(corners_list)
#                     avg_corners = np.mean(all_corners, axis=0)

#                     # Log the averaged corner coordinates
#                     rospy.loginfo(f"Averaged corners for Marker {marker_ID}: {avg_corners}")

#                     # Create a Float32MultiArray message for the averaged corners
#                     averaged_msg = Float32MultiArray()
#                     averaged_msg.data = [float(marker_ID)] + avg_corners.flatten().tolist()

#                     # Publish the averaged data
#                     self.aruco_detection_pub.publish(averaged_msg)
#                     rospy.loginfo(f"Published averaged Aruco ID and corners: {averaged_msg.data}")
#                 else:
#                     rospy.loginfo(f"No detections for Marker {marker_ID}, skipping averaging.")

#             # Reset the accumulation data
#             self.marker_corners.clear()
#             rospy.loginfo("Accumulation data cleared after averaging")
#         else:
#             rospy.loginfo("Received trigger with value other than 1, ignoring.")

#     def publish_to_ros(self, frame):
#         # Publishing the processed image
#         msg_out = CompressedImage()
#         msg_out.header.stamp = rospy.Time.now()
#         msg_out.format = "jpeg"
#         msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tobytes()

#         self.image_pub.publish(msg_out)  # Publish the processed image

# def main():
#     rospy.init_node('EGB349_vision', anonymous=True)
#     rospy.loginfo("Processing images...")

#     aruco_detect = ArucoDetector()

#     rospy.spin()

# if __name__ == '__main__':
#     main()

