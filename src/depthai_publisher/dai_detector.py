#!/usr/bin/env python3
############################### Libraries ###############################
from pathlib import Path
import os
import threading
import csv
import argparse
import time
import sys
import json     # Yolo conf use json files
import cv2
import numpy as np
import depthai as dai
import rospy
import tf2_ros
import tf_conversions
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from collections import defaultdict, deque

############################### Parameters ###############################
# Global variables to deal with pipeline creation
pipeline = None
cam_source = 'rgb' #'rgb', 'left', 'right'
cam=None
# sync outputs
syncNN = True
# model path
modelsPath = "/home/uavteam2/models"
# modelName = 'exp31Yolov5_ov21.4_6sh'
modelName = 'best_mission1'
# confJson = 'exp31Yolov5.json'
confJson = 'best_mission1.json'
# set class colours
class_colors = {
    0: (127, 0, 255),   # pink
    1: (128, 255, 0),   # green
    2: (252, 202, 66),  # blue
}

################################ Yolo Config File
# Parse Config
configPath = Path(f'{modelsPath}/{modelName}/{confJson}')
if not configPath.exists():
    raise ValueError("Path {} does not exist!".format(configPath))

with configPath.open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})

# Extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})
# Parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})
rospy.loginfo("Parse Configured")

class DepthaiCamera():
    res = [416, 416]
    fps = 20.0

    pub_topic = '/depthai_node/image/compressed'
    pub_topic_raw = '/depthai_node/image/raw'
    pub_topic_detect = '/depthai_node/detection/compressed'
    pub_topic_cam_inf = '/depthai_node/camera/camera_info'
    pub_topic_objects = '/object_pose'

    def __init__(self):
        self.pipeline = dai.Pipeline()

         # Input image size
        if "input_size" in nnConfig:
            self.nn_shape_w,self.nn_shape_h=tuple(map(int, nnConfig.get("input_size").split('x')))

        # Pulbish ROS image data
        self.pub_image = rospy.Publisher(self.pub_topic, CompressedImage, queue_size=10)
        self.pub_image_raw = rospy.Publisher(self.pub_topic_raw, Image, queue_size=10)
        self.pub_image_detect = rospy.Publisher(self.pub_topic_detect, CompressedImage, queue_size=10)
        # Create a publisher for the object pose data
        self.pub_object_detect = rospy.Publisher(self.pub_topic_objects, Float32MultiArray, queue_size=10)
        # Create a publisher for the CameraInfo topic
        self.pub_cam_inf = rospy.Publisher(self.pub_topic_cam_inf, CameraInfo, queue_size=10)
        # Create a timer for the callback
        self.timer = rospy.Timer(rospy.Duration(1.0 / 10), self.publish_camera_info, oneshot=False)

        self.br = CvBridge()
        self.published_objects = set()

        # Keep Unique IDs after 10 counts
        self.coordinate_buffers = defaultdict(lambda: deque(maxlen=10)) 

        rospy.loginfo("Publishing to all topics initialised")
        #rospy.on_shutdown(lambda: self.shutdown())

    def publish_object_data(self, frame, detection):
        # Structure IDs based on labels to avoid class confusion, Nav uses 101 and 102 for objects, and 0 - 100 inc for arucos
        if labels[detection.label] == "backpack":
            object_id = 101
        elif labels[detection.label] == 'person':
            object_id = 102
        else:
            rospy.logwarn("This Object Identifier is unknown")
            return

        obj_frame = frame.copy()
        # Convert to pixel coordinates for absolute values and calculate dynamic marker box size
        bbox = self.frameNorm(obj_frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        corners = np.array([
            (bbox[0], bbox[1]),
            (bbox[2], bbox[1]),
            (bbox[2], bbox[3]),
            (bbox[0], bbox[3])
        ])
        height = bbox[3] - bbox[1]
        width = bbox[2] - bbox[0]

        marker_length_x = width / frame.shape[1]
        marker_length_y = height / frame.shape[0]

        # Append the detection to the buffer
        self.coordinate_buffers[object_id].append(corners.flatten())

        if len(self.coordinate_buffers[object_id]) == 10 and object_id not in self.published_objects:
            # Compute average coordinates
            avg_corners = np.mean(self.coordinate_buffers[object_id], axis=0).reshape((4, 2))
            
            # Publish Object box data and coordinates 
            object_detection_msg = Float32MultiArray()
            object_detection_msg.data = [float(object_id)] + [coord for point in avg_corners for coord in point] + [marker_length_x, marker_length_y]
 
            self.pub_object_detect.publish(object_detection_msg)
            rospy.loginfo("Found Target: {}".format(labels[detection.label]))
            rospy.loginfo("Published Object Identity and Coordinates for: {}".format(object_detection_msg.data[0]))

            #os.system(f"espeak 'Detected Target: {labels[detection.label]}'")  # Use espeak to speak the ID
            #rospy.loginfo("Target detected: {}".format(labels[detection.label]))
            
            # Mark objects as published
            self.published_objects.add(object_id)
            self.coordinate_buffers[object_id].clear()

    def publish_camera_info(self, timer=None):
        # Create a CameraInfo message
        camera_info_msg = CameraInfo()
        camera_info_msg.header.frame_id = "camera"
        camera_info_msg.height = self.nn_shape_h # Set the height of the camera image
        camera_info_msg.width = self.nn_shape_w  # Set the width of the camera image

        # Set the camera intrinsic matrix (fx, fy, cx, cy)
        camera_info_msg.K = [615.381, 0.0, 320.0, 0.0, 615.381, 240.0, 0.0, 0.0, 1.0]
        # Set the distortion parameters (k1, k2, p1, p2, k3)
        camera_info_msg.D = [-0.10818, 0.12793, 0.00000, 0.00000, -0.04204]
        # Set the rectification matrix (identity matrix)
        camera_info_msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        # Set the projection matrix (P)
        camera_info_msg.P = [615.381, 0.0, 320.0, 0.0, 0.0, 615.381, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        # Set the distortion model
        camera_info_msg.distortion_model = "plumb_bob"
        # Set the timestamp
        camera_info_msg.header.stamp = rospy.Time.now()

        self.pub_cam_inf.publish(camera_info_msg)  # Publish the camera info message

    def rgb_camera(self):
        cam_rgb = self.pipeline.createColorCamera()
        cam_rgb.setPreviewSize(self.res[0], self.res[1])
        cam_rgb.setInterleaved(False)
        cam_rgb.setFps(self.fps)

        # Def xout / xin
        ctrl_in = self.pipeline.createXLinkIn()
        ctrl_in.setStreamName("cam_ctrl")
        ctrl_in.out.link(cam_rgb.inputControl)

        xout_rgb = self.pipeline.createXLinkOut()
        xout_rgb.setStreamName("video")

        cam_rgb.preview.link(xout_rgb.input)

    def run(self):
        ###############################RunModel###############################
        # Pipeline defined, now the device is assigned and pipeline is started
        pipeline = None
        # Model parameters
        modelPathName = f'{modelsPath}/{modelName}/{modelName}.blob'
        print(metadata)
        nnPath = str((Path(__file__).parent / Path(modelPathName)).resolve().absolute())
        print(nnPath)

        pipeline = self.createPipeline(nnPath)

        with dai.Device() as device:
            cams = device.getConnectedCameras()
            depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
            if cam_source != "rgb" and not depth_enabled:
                raise RuntimeError("Unable to run the experiment on {} camera! Available cameras: {}".format(cam_source, cams))
            device.startPipeline(pipeline)

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            q_nn_input = device.getOutputQueue(name="nn_input", maxSize=4, blocking=False)
            q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

            frame = None
            detections = []
            start_time = time.time()
            counter = 0
            fps = 0

            while True:
                found_classes = []
                inRgb = q_nn_input.get()
                inDet = q_nn.get()

                if inRgb is not None:
                    frame = inRgb.getCvFrame()
                else:
                    print("Cam Image empty, trying again...")
                    continue
                
                if inDet is not None:
                    detections = inDet.detections
                    for detection in detections:
                        # Publish the object Id
                        self.publish_object_data(frame, detection)
                    found_classes = np.unique(found_classes)
                    overlay = self.show_yolo(frame, detections)
                else:
                    print("Detection empty, trying again...")
                    continue

                if frame is not None:
                    cv2.putText(overlay, "NN fps: {:.2f}".format(fps), (2, overlay.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 255, 0))
                    cv2.putText(overlay, "Found classes {}".format(found_classes), (2, 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 255, 0))
                    # cv2.imshow("nn_output_yolo", overlay)
                    self.publish_to_ros(frame)
                    self.publish_detect_to_ros(overlay)
                    self.publish_camera_info()

                ## Function to compute FPS
                counter+=1
                if (time.time() - start_time) > 1 :
                    fps = counter / (time.time() - start_time)
                    counter = 0
                    start_time = time.time()

    # 
    def publish_to_ros(self, frame):
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.header.frame_id = "home"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
        self.pub_image.publish(msg_out)
        # Publish image raw
        msg_img_raw = self.br.cv2_to_imgmsg(frame, encoding="bgr8")
        self.pub_image_raw.publish(msg_img_raw)

    def publish_detect_to_ros(self, frame):
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.header.frame_id = "home"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
        self.pub_image_detect.publish(msg_out)
        
    ############################### Functions ###############################
    ######### Functions for Yolo Decoding
    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def show_yolo(self, frame, detections):
        # Both YoloDetectionNetwork and MobileNetDetectionNetwork output this message. This message contains a list of detections, which contains label, confidence, and the bounding box information (xmin, ymin, xmax, ymax).
        overlay =  frame.copy()
        for detection in detections:
            bbox = self.frameNorm(overlay, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            class_id = detection.label
            color = class_colors.get(class_id, (0, 0, 255)) # default colour red if class not found
            cv2.putText(overlay, labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(overlay, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        return overlay

    # Start defining a pipeline
    def createPipeline(self, nnPath):

        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2022_1)

        # Define a neural network that will make predictions based on the source frames
        detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
        # Network specific settings
        detection_nn.setConfidenceThreshold(confidenceThreshold)
        detection_nn.setNumClasses(classes)
        detection_nn.setCoordinateSize(coordinates)
        detection_nn.setAnchors(anchors)
        detection_nn.setAnchorMasks(anchorMasks)
        detection_nn.setIouThreshold(iouThreshold)
        # generic nn configs
        detection_nn.setBlobPath(nnPath)
        detection_nn.setNumPoolFrames(4)
        detection_nn.input.setBlocking(False)
        detection_nn.setNumInferenceThreads(2)

        # Define a source - color camera
        if cam_source == 'rgb':
            cam = pipeline.create(dai.node.ColorCamera)
            cam.setPreviewSize(self.nn_shape_w,self.nn_shape_h)
            cam.setInterleaved(False)
            cam.preview.link(detection_nn.input)
            cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            cam.setFps(10)
            print("Using RGB camera...")
        elif cam_source == 'left':
            cam = pipeline.create(dai.node.MonoCamera)
            cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
            print("Using BW Left cam")
        elif cam_source == 'right':
            cam = pipeline.create(dai.node.MonoCamera)
            cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            print("Using BW Rigth cam")

        if cam_source != 'rgb':
            manip = pipeline.create(dai.node.ImageManip)
            manip.setResize(self.nn_shape_w,self.nn_shape_h)
            manip.setKeepAspectRatio(True)
            # manip.setFrameType(dai.RawImgFrame.Type.BGR888p)
            manip.setFrameType(dai.RawImgFrame.Type.RGB888p)
            cam.out.link(manip.inputImage)
            manip.out.link(detection_nn.input)

        # Create outputs
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("nn_input")
        xout_rgb.input.setBlocking(False)

        detection_nn.passthrough.link(xout_rgb.input)

        xinDet = pipeline.create(dai.node.XLinkOut)
        xinDet.setStreamName("nn")
        xinDet.input.setBlocking(False)

        detection_nn.out.link(xinDet.input)

        return pipeline

    def shutdown(self):
        cv2.destroyAllWindows()

# Main Code
def main():
    global speak_pub 
# Set up the publisher for spoken text on the 'spoken_text' topic
    #speak_pub = rospy.Publisher('spoken_text', String, queue_size=10)
    rospy.init_node('depthai_node')
    dai_cam = DepthaiCamera()
    while not rospy.is_shutdown():
        dai_cam.run()
    dai_cam.shutdown()

if __name__ == "__main__":
    main()


# # Libraries and Parameters Setup
# from pathlib import Path
# import os  #espeak library
# import time
# import json     # Configuration uses JSON format
# import cv2
# import numpy as np
# import depthai as dai
# import rospy
# from sensor_msgs.msg import CompressedImage, Image, CameraInfo
# from std_msgs.msg import Float32MultiArray
# from cv_bridge import CvBridge
# from std_msgs.msg import String
# import subprocess
# import threading

# # YOLO Config File and Model Parameters
# pipeline = None
# cam_source = 'rgb' #'rgb', 'left', 'right'
# cam=None
# # sync outputs
# syncNN = True

# modelsPath = "/home/uavteam2/models"
# modelName = 'best_mission1'
# confJson = 'best_mission1.json'
# class_colors = {
#     0: (127, 0, 255),   # pink
#     1: (128, 255, 0),   # green
#     2: (252, 202, 66),  # blue
# }

# configPath = Path(f'{modelsPath}/{modelName}/{confJson}')
# if not configPath.exists():
#     raise ValueError(f"Path {configPath} does not exist!")

# with configPath.open() as f:
#     config = json.load(f)

# nnConfig = config.get("nn_config", {})
# metadata = nnConfig.get("NN_specific_metadata", {})
# classes = metadata.get("classes", {})
# coordinates = metadata.get("coordinates", {})
# anchors = metadata.get("anchors", {})
# anchorMasks = metadata.get("anchor_masks", {})
# iouThreshold = metadata.get("iou_threshold", {})
# confidenceThreshold = metadata.get("confidence_threshold", {})
# nnMappings = config.get("mappings", {})
# object_labels = nnMappings.get("labels", {})

# class DepthaiCamera:
#     fps = 20.0
#     pub_topic = '/depthai_node/image/compressed'
#     pub_topic_raw = '/depthai_node/image/raw'
#     pub_topic_detect = '/depthai_node/detection/compressed'
#     pub_topic_cam_inf = '/depthai_node/camera/camera_info'
#     pub_object_pose = '/object_pose'  # Publisher topic to match the first script

#     def __init__(self):
#         self.pipeline = dai.Pipeline()

#         if "input_size" in nnConfig:
#             self.nn_shape_w, self.nn_shape_h = tuple(map(int, nnConfig.get("input_size").split('x')))

#         # ROS Publishers
#         self.pub_image = rospy.Publisher(self.pub_topic, CompressedImage, queue_size=10)
#         self.pub_image_raw = rospy.Publisher(self.pub_topic_raw, Image, queue_size=10)
#         self.pub_image_detect = rospy.Publisher(self.pub_topic_detect, CompressedImage, queue_size=10)
#         self.pub_cam_inf = rospy.Publisher(self.pub_topic_cam_inf, CameraInfo, queue_size=50)
#         self.pub_object_detect = rospy.Publisher(self.pub_object_pose, Float32MultiArray, queue_size=10)  # Publisher for object pose

#         self.timer = rospy.Timer(rospy.Duration(1.0 / 10), self.publish_camera_info, oneshot=False)
#         self.br = CvBridge()

#         # Dictionary to store the accumulated coordinates for each corner and detection count
#         self.object_corners = {
#             101: {  # Drone
#                 'sum_tl_x': 0, 'sum_tl_y': 0,  # Top-left
#                 'sum_tr_x': 0, 'sum_tr_y': 0,  # Top-right
#                 'sum_br_x': 0, 'sum_br_y': 0,  # Bottom-right
#                 'sum_bl_x': 0, 'sum_bl_y': 0,  # Bottom-left
#                 'count': 0,  # Detection count
#                 'published': False  # Flag to indicate if data has been published for this object
#             },
#             102: {  # Phone
#                 'sum_tl_x': 0, 'sum_tl_y': 0,  # Top-left
#                 'sum_tr_x': 0, 'sum_tr_y': 0,  # Top-right
#                 'sum_br_x': 0, 'sum_br_y': 0,  # Bottom-right
#                 'sum_bl_x': 0, 'sum_bl_y': 0,  # Bottom-left
#                 'count': 0,  # Detection count
#                 'published': False  # Flag to indicate if data has been published for this object
#             }
#         }

#         rospy.loginfo("Publishing images to rostopic: {}".format(self.pub_topic))

#     def speak(self, text):
#         global speak_pub
#         # Run the espeak subprocess asynchronously to avoid blocking
#         threading.Thread(target=subprocess.run, args=(['espeak', text],)).start()
#         if speak_pub:
#             speak_pub.publish(text)  # Publish the text to the ROS topic


#     def publish_object_pose(self, detection, frame):
#         """
#         Publishes the object pose after accumulating 10 detections.
#         """
#         # Assign object ID based on detection label
#         if object_labels[detection.label] == 'backpack':
#             obj_id = 101
#             obj_name = "backpack"
#         elif object_labels[detection.label] == 'person':
#             obj_id = 102
#             obj_name = "person"
#         else:
#             rospy.logwarn("Unknown object label detected")
#             return

#         # Stop detecting if the object has already been published
#         if self.object_corners[obj_id]['published']:
#             rospy.loginfo(f"Object {obj_name} (ID {obj_id}) has already been published. Skipping further detections.")
#             return

#         # Convert bounding box from normalized values to pixel coordinates
#         bounding_box = self.frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

#         # Calculate bounding box corners in pixel coordinates
#         x1, y1 = bounding_box[0], bounding_box[1]  # Top-left
#         x2, y2 = bounding_box[2], bounding_box[1]  # Top-right
#         x3, y3 = bounding_box[2], bounding_box[3]  # Bottom-right
#         x4, y4 = bounding_box[0], bounding_box[3]  # Bottom-left

#         # Update the coordinates and count for this object
#         self.object_corners[obj_id]['sum_tl_x'] += x1
#         self.object_corners[obj_id]['sum_tl_y'] += y1
#         self.object_corners[obj_id]['sum_tr_x'] += x2
#         self.object_corners[obj_id]['sum_tr_y'] += y2
#         self.object_corners[obj_id]['sum_br_x'] += x3
#         self.object_corners[obj_id]['sum_br_y'] += y3
#         self.object_corners[obj_id]['sum_bl_x'] += x4
#         self.object_corners[obj_id]['sum_bl_y'] += y4
#         self.object_corners[obj_id]['count'] += 1

#         # Only publish after 10 detections
#         if self.object_corners[obj_id]['count'] == 10:
#             # Calculate the average coordinates for each corner
#             avg_tl_x = self.object_corners[obj_id]['sum_tl_x'] / 10
#             avg_tl_y = self.object_corners[obj_id]['sum_tl_y'] / 10
#             avg_tr_x = self.object_corners[obj_id]['sum_tr_x'] / 10
#             avg_tr_y = self.object_corners[obj_id]['sum_tr_y'] / 10
#             avg_br_x = self.object_corners[obj_id]['sum_br_x'] / 10
#             avg_br_y = self.object_corners[obj_id]['sum_br_y'] / 10
#             avg_bl_x = self.object_corners[obj_id]['sum_bl_x'] / 10
#             avg_bl_y = self.object_corners[obj_id]['sum_bl_y'] / 10

#             # Log the averaged corner coordinates
#             rospy.loginfo(f"Averaged corners for {obj_name}: "
#                           f"Top-left({avg_tl_x}, {avg_tl_y}), Top-right({avg_tr_x}, {avg_tr_y}), "
#                           f"Bottom-right({avg_br_x}, {avg_br_y}), Bottom-left({avg_bl_x}, {avg_bl_y})")

#             # Create Float32MultiArray for object pose in the required format
#             msg = Float32MultiArray()
#             msg.data = [
#                 obj_id,              # Class ID
#                 avg_tl_x, avg_tl_y,  # Average top-left corner
#                 avg_tr_x, avg_tr_y,  # Average top-right corner
#                 avg_br_x, avg_br_y,  # Average bottom-right corner
#                 avg_bl_x, avg_bl_y,  # Average bottom-left corner
#                 (avg_br_x - avg_tl_x) / frame.shape[1],  # Width (normalized)
#                 (avg_br_y - avg_tl_y) / frame.shape[0]   # Height (normalized)
#             ]

#             # Publish the message
#             self.pub_object_detect.publish(msg)
#             rospy.loginfo(f"Published object pose for {obj_name} (ID {obj_id})")
            
#             self.speak(f"Detected Target: {object_labels[detection.label]}")
#             # os.system(f"espeak 'Detected Target: {object_labels[detection.label]}'") #use espeak

#             # Mark as published to stop further messages
#             self.object_corners[obj_id]['published'] = True

#     def publish_camera_info(self, timer=None):
#         camera_info_msg = CameraInfo()
#         camera_info_msg.header.frame_id = "camera_frame"
#         camera_info_msg.height = self.nn_shape_h
#         camera_info_msg.width = self.nn_shape_w
#         camera_info_msg.distortion_model = "plumb_bob"
#         camera_info_msg.header.stamp = rospy.Time.now()

#         self.pub_cam_inf.publish(camera_info_msg)

#     def run(self):
#         modelPathName = f'{modelsPath}/{modelName}/{modelName}.blob'
#         nnPath = str((Path(__file__).parent / Path(modelPathName)).resolve().absolute())
#         pipeline = self.createPipeline(nnPath)

#         with dai.Device() as device:
#             cams = device.getConnectedCameras()
#             depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
#             if cam_source != "rgb" and not depth_enabled:
#                 raise RuntimeError("Unable to run the experiment on {} camera! Available cameras: {}".format(cam_source, cams))
#             device.startPipeline(pipeline)

#             q_nn_input = device.getOutputQueue(name="nn_input", maxSize=5, blocking=False)
#             q_nn = device.getOutputQueue(name="nn", maxSize=5, blocking=False)

#             frame = None
#             detections = []
#             start_time = time.time()
#             counter = 0
#             fps = 0

#             while True:
#                 found_classes = []
#                 inRgb = q_nn_input.get()
#                 inDet = q_nn.get()

#                 if inRgb is not None:
#                     frame = inRgb.getCvFrame()
#                 else:
#                     print("Image empty, trying again...")
#                     continue

#                 if inDet is not None:
#                     detections = inDet.detections
#                     for detection in detections:
#                         self.publish_object_pose(detection, frame)  # Publish object pose
#                     found_classes = np.unique([d.label for d in detections])
#                     overlay = self.show_yolo(frame, detections)
#                 else:
#                     print("Detection empty, trying again...")
#                     continue

#                 if frame is not None:
#                     cv2.putText(overlay, "NN fps: {:.2f}".format(fps), (2, overlay.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 255, 0))
#                     cv2.putText(overlay, "Found classes {}".format(found_classes), (2, 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 255, 0))
#                     self.publish_to_ros(frame)
#                     self.publish_detect_to_ros(overlay)
#                     self.publish_camera_info()

#                 counter += 1
#                 if (time.time() - start_time) > 1:
#                     fps = counter / (time.time() - start_time)
#                     counter = 0
#                     start_time = time.time()

#     def publish_to_ros(self, frame):
#         msg_out = CompressedImage()
#         msg_out.header.stamp = rospy.Time.now()
#         msg_out.format = "jpeg"
#         msg_out.header.frame_id = "home"
#         msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
#         self.pub_image.publish(msg_out)
#         msg_img_raw = self.br.cv2_to_imgmsg(frame, encoding="bgr8")
#         self.pub_image_raw.publish(msg_img_raw)

#     def publish_detect_to_ros(self, frame):
#         msg_out = CompressedImage()
#         msg_out.header.stamp = rospy.Time.now()
#         msg_out.format = "jpeg"
#         msg_out.header.frame_id = "home"
#         msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
#         self.pub_image_detect.publish(msg_out)

#     def frameNorm(self, frame, bbox):
#         normVals = np.full(len(bbox), frame.shape[0])
#         normVals[::2] = frame.shape[1]
#         return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

#     def show_yolo(self, frame, detections):
#         overlay = frame.copy()
#         for detection in detections:
#             bbox = self.frameNorm(overlay, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
#             class_id = detection.label
#             color = class_colors.get(class_id, (0, 0, 255)) # default colour red if class not found
#             cv2.putText(overlay, object_labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
#             cv2.putText(overlay, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
#             cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
#         return overlay

#     def createPipeline(self, nnPath):
#         pipeline = dai.Pipeline()
#         pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2022_1)
#         detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
#         detection_nn.setConfidenceThreshold(confidenceThreshold)
#         detection_nn.setNumClasses(classes)
#         detection_nn.setCoordinateSize(coordinates)
#         detection_nn.setAnchors(anchors)
#         detection_nn.setAnchorMasks(anchorMasks)
#         detection_nn.setIouThreshold(iouThreshold)
#         detection_nn.setBlobPath(nnPath)
#         detection_nn.setNumPoolFrames(4)
#         detection_nn.input.setBlocking(False)
#         detection_nn.setNumInferenceThreads(2)

#         cam = pipeline.create(dai.node.ColorCamera)
#         cam.setPreviewSize(self.nn_shape_w, self.nn_shape_h)
#         cam.setInterleaved(False)
#         cam.preview.link(detection_nn.input)
#         cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
#         cam.setFps(40)

#         xout_rgb = pipeline.create(dai.node.XLinkOut)
#         xout_rgb.setStreamName("nn_input")
#         xout_rgb.input.setBlocking(False)

#         detection_nn.passthrough.link(xout_rgb.input)

#         xinDet = pipeline.create(dai.node.XLinkOut)
#         xinDet.setStreamName("nn")
#         xinDet.input.setBlocking(False)

#         detection_nn.out.link(xinDet.input)

#         return pipeline

#     def shutdown(self):
#         cv2.destroyAllWindows()

# # Main Code
# def main():
#     global speak_pub 
# # Set up the publisher for spoken text on the 'spoken_text' topic
#     speak_pub = rospy.Publisher('spoken_text', String, queue_size=10)
#     rospy.init_node('depthai_node')
#     dai_cam = DepthaiCamera()
#     while not rospy.is_shutdown():
#         dai_cam.run()
#     dai_cam.shutdown()

# if __name__ == "__main__":
#    main()