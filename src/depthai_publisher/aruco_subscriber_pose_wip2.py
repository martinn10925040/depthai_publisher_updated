#!/usr/bin/env python3

import sys
import rospy
import cv2
import numpy as np
import tf2_ros
import tf_conversions
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from collections import defaultdict, deque
import os  # Library for executing shell commands

class ArucoPoseEstimator:
    def __init__(self):
        rospy.loginfo("Initializing Aruco Pose Estimator...")

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Initialize ROS parameters
        self.param_use_compressed = rospy.get_param("~use_compressed", False)

        # Subscribers for receiving images and camera info
        if self.param_use_compressed:
            self.image_sub = rospy.Subscriber("~image_raw/compressed", CompressedImage, self.img_callback)
            self.pub_overlay = rospy.Publisher("~overlay/image_raw/compressed", CompressedImage, queue_size=1)
        else:
            self.image_sub = rospy.Subscriber("~image_raw", Image, self.img_callback)
            self.pub_overlay = rospy.Publisher("~overlay/image_raw", Image, queue_size=1)

        self.sub_info = rospy.Subscriber("~camera_info", CameraInfo, self.callback_info)

        # TF Broadcaster
        self.tfbr = tf2_ros.TransformBroadcaster()

        # Initialize ArUco dictionary and parameters
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # Marker detection buffer for averaging coordinates over 10 iterations
        self.coordinate_buffers = defaultdict(lambda: deque(maxlen=10))  # Store last 10 sets of coordinates
        self.marker_data = {}  # Store averaged corner data keyed by marker ID

        # Camera calibration parameters
        self.got_camera_info = False
        self.camera_matrix = None
        self.dist_coeffs = None

        # Model object for pose estimation (based on 200mm marker size)
        marker_side_length = 0.2
        self.model_object = np.array([
            (0.0, 0.0, 0.0),  # Center point
            (-marker_side_length / 2, marker_side_length / 2, 0.0),  # Top-left corner
            (marker_side_length / 2, marker_side_length / 2, 0.0),  # Top-right corner
            (marker_side_length / 2, -marker_side_length / 2, 0.0),  # Bottom-right corner
            (-marker_side_length / 2, -marker_side_length / 2, 0.0)  # Bottom-left corner
        ])

    def callback_info(self, msg_in):
        self.dist_coeffs = np.array([[msg_in.D[0], msg_in.D[1], msg_in.D[2], msg_in.D[3], msg_in.D[4]]], dtype="double")
        self.camera_matrix = np.array([
            (msg_in.P[0], msg_in.P[1], msg_in.P[2]),
            (msg_in.P[4], msg_in.P[5], msg_in.P[6]),
            (msg_in.P[8], msg_in.P[9], msg_in.P[10])],
            dtype="double")

        if not self.got_camera_info:
            rospy.loginfo("Camera info received")
            self.got_camera_info = True

    def img_callback(self, msg_in):
        # Convert ROS image to OpenCV format
        try:
            if self.param_use_compressed:
                frame = self.bridge.compressed_imgmsg_to_cv2(msg_in, "bgr8")
            else:
                frame = self.bridge.imgmsg_to_cv2(msg_in, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"Error converting image: {e}")
            return

        # Detect and process ArUco markers in the frame
        self.find_and_process_aruco(frame)

    def find_and_process_aruco(self, frame):
        # Detect ArUco markers in the frame
        (corners, ids, _) = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)

        if len(corners) > 0:
            ids = ids.flatten()

            for (marker_corner, marker_ID) in zip(corners, ids):
                corners = marker_corner.reshape((4, 2))
                # Draw lines around the marker
                self.draw_aruco_marker(frame, corners, marker_ID)
                
                # Store coordinates for averaging
                self.coordinate_buffers[marker_ID].append(corners.flatten())

                # Check if we have enough data (10 sets of coordinates)
                if len(self.coordinate_buffers[marker_ID]) == 10:
                    # Compute the average coordinates
                    avg_corners = np.mean(self.coordinate_buffers[marker_ID], axis=0).reshape((4, 2))
                    
                    # Store the averaged coordinates
                    self.marker_data[marker_ID] = avg_corners

                    # Estimate pose using averaged corners
                    self.estimate_pose(marker_ID, avg_corners)

                    # Speak the marker ID using espeak (only once per averaged detection)
                    os.system(f"espeak 'Detected ArUco Marker {marker_ID}'")
                    rospy.loginfo(f"Averaged ArUco Marker {marker_ID} coordinates published")

                    # Clear buffer after processing
                    self.coordinate_buffers[marker_ID].clear()

        # Publish the processed frame with detected markers
        self.publish_to_ros(frame)

    def draw_aruco_marker(self, frame, corners, marker_ID):
        # Draw lines around the marker
        (top_left, top_right, bottom_right, bottom_left) = corners
        cv2.line(frame, tuple(map(int, top_left)), tuple(map(int, top_right)), (0, 255, 0), 3)
        cv2.line(frame, tuple(map(int, top_right)), tuple(map(int, bottom_right)), (0, 255, 0), 3)
        cv2.line(frame, tuple(map(int, bottom_right)), tuple(map(int, bottom_left)), (0, 255, 0), 3)
        cv2.line(frame, tuple(map(int, bottom_left)), tuple(map(int, top_left)), (0, 255, 0), 3)

        # Display Marker ID
        cv2.putText(frame, str(marker_ID), (int(top_left[0]), int(top_left[1]) - 15),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    def estimate_pose(self, marker_ID, corners):
        if corners is not None and len(corners) == 4:
            # Compute model_image points from corners
            model_image = np.array([
                ((corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2),  # Center point
                (corners[1][0], corners[1][1]),  # Top-left
                (corners[2][0], corners[2][1]),  # Top-right
                (corners[3][0], corners[3][1]),  # Bottom-right
                (corners[0][0], corners[0][1])   # Bottom-left
            ])

            # SolvePnP method for pose estimation
            (success, rvec, tvec) = cv2.solvePnP(self.model_object, model_image, self.camera_matrix, self.dist_coeffs)

            # If a result was found, send to TF2
            if success:
                msg_out = TransformStamped()
                msg_out.header.stamp = rospy.Time.now()
                msg_out.child_frame_id = f"Aruco_Marker_{marker_ID}"
                msg_out.transform.translation.x = tvec[0] * 10e-2
                msg_out.transform.translation.y = tvec[1] * 10e-2
                msg_out.transform.translation.z = tvec[2] * 10e-2
                q = tf_conversions.transformations.quaternion_from_euler(rvec[0], rvec[1], rvec[2])
                msg_out.transform.rotation.w = q[3]
                msg_out.transform.rotation.x = q[0]
                msg_out.transform.rotation.y = q[1]
                msg_out.transform.rotation.z = q[2]

                self.tfbr.sendTransform(msg_out)
                rospy.loginfo(f"Translation for Marker {marker_ID}: [x: {msg_out.transform.translation.x:.2f}, y: {msg_out.transform.translation.y:.2f}, z: {msg_out.transform.translation.z:.2f}]")
                rospy.loginfo(f"Rotation for Marker {marker_ID}: [x: {msg_out.transform.rotation.x:.2f}, y: {msg_out.transform.rotation.y:.2f}, z: {msg_out.transform.rotation.z:.2f}, w: {msg_out.transform.rotation.w:.2f}]")

    def publish_to_ros(self, frame):
        msg_out = CompressedImage() if self.param_use_compressed else Image()
        msg_out.header.stamp = rospy.Time.now()
        if self.param_use_compressed:
            msg_out.format = "jpeg"
            msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tobytes()
            self.pub_overlay.publish(msg_out)
        else:
            try:
                self.pub_overlay.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
            except CvBridgeError as e:
                rospy.logerr(f"Error converting CV image to ROS image: {e}")

# Entry point of the program
def main():
    rospy.init_node('aruco_pose_estimator', anonymous=True)
    rospy.loginfo("Node 'aruco_pose_estimator' started")
    aruco_pose_estimator = ArucoPoseEstimator()  # Instantiate the class
    rospy.spin()

if __name__ == "__main__":
    main()
