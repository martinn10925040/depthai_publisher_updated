 
pipeline = None
cam_source = 'rgb' #'rgb', 'left', 'right'
cam=None
# sync outputs
syncNN = True

#!/usr/bin/env python3

# Libraries and Parameters Setup
from pathlib import Path
import os  #espeak library
import time # For handling timestamps
import json     # Configuration uses JSON format
import cv2 # OpenCv for handling image processing
import numpy as np # # For numerical operatons
import depthai as dai # Depthai library for working with OAK-D pro cameras

import rospy # ROS functionality
from sensor_msgs.msg import CompressedImage, Image, CameraInfo # ROS message types for imamges
from std_msgs.msg import Float32MultiArray # ROS message types for pose data and strings
from cv_bridge import CvBridge # To convert between ROS and OpenCV images
from std_msgs.msg import String # ROS message types for pose data and strings
# import subprocess
# import threading

# YOLO Config File and Model Parameters
modelsPath = "/home/uavteam2/QUT_EGH450/src/depthai_publisher/src/depthai_publisher/models" # Path to models
modelName = 'best_mission2' # Model name to use
confJson = 'best_mission2.json' # Corresponding configuration file

# COnstruct Config File and Model Parameters
configPath = Path(f'{modelsPath}/{modelName}/{confJson}')
if not configPath.exists():
    raise ValueError(f"Path {configPath} does not exist!")  # Raise error if config file not found 
# Load the configuration from the JSON file
with configPath.open() as f:
    config = json.load(f)
# Extract YOLO-specific configuration details from the loaded file
nnConfig = config.get("nn_config", {})
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})
nnMappings = config.get("mappings", {})
object_labels = nnMappings.get("labels", {})

# global variable
pub_speaker = None #declare global publisher for speaking

class DepthaiCamera:
    fps = 30.0 # Camera frames per second
    pub_topic = '/depthai_node/image/compressed' # ROS topic for publishing compressed images
    pub_topic_raw = '/depthai_node/image/raw' # ROS topic for publishing raw images
    pub_topic_detect = '/depthai_node/detection/compressed' # ROS topic for publishing detection results
    pub_topic_cam_inf = '/depthai_node/camera/camera_info' # ROS topic for publishing camera information
    pub_object_pose = '/object_pose'  # Publisher topic for publishing object pose (Coordinates)

    def __init__(self):
        self.pipeline = dai.Pipeline() # Initialise pipline for DepthAI
        # Set neural network input size if available in the configuration
        if "input_size" in nnConfig:
            self.nn_shape_w, self.nn_shape_h = tuple(map(int, nnConfig.get("input_size").split('x')))

        # Initialise ROS publishers for different topics
        self.pub_image = rospy.Publisher(self.pub_topic, CompressedImage, queue_size=10)
        self.pub_image_raw = rospy.Publisher(self.pub_topic_raw, Image, queue_size=10)
        self.pub_image_detect = rospy.Publisher(self.pub_topic_detect, CompressedImage, queue_size=10)
        self.pub_cam_inf = rospy.Publisher(self.pub_topic_cam_inf, CameraInfo, queue_size=50)
        self.pub_object_detect = rospy.Publisher(self.pub_object_pose, Float32MultiArray, queue_size=10)  # Publisher for object pose
        # Timer to periodically publish camera information
        self.timer = rospy.Timer(rospy.Duration(1.0 / 10), self.publish_camera_info, oneshot=False)
        # OpenCV-ROS bridge for converting between ROS and OpenCV images
        self.br = CvBridge()

        # Dictionary to store the accumulated coordinates for each corner and detection count
        self.object_corners = {
            101: {  # Drone
                'sum_tl_x': 0, 'sum_tl_y': 0,  # Top-left
                'sum_tr_x': 0, 'sum_tr_y': 0,  # Top-right
                'sum_br_x': 0, 'sum_br_y': 0,  # Bottom-right
                'sum_bl_x': 0, 'sum_bl_y': 0,  # Bottom-left
                'count': 0,  # Detection count
                'published': False  # Flag to indicate if data has been published for this object
            },
            102: {  # Phone
                'sum_tl_x': 0, 'sum_tl_y': 0,  # Top-left
                'sum_tr_x': 0, 'sum_tr_y': 0,  # Top-right
                'sum_br_x': 0, 'sum_br_y': 0,  # Bottom-right
                'sum_bl_x': 0, 'sum_bl_y': 0,  # Bottom-left
                'count': 0,  # Detection count
                'published': False  # Flag to indicate if data has been published for this object
            }
        }

        rospy.loginfo("Publishing images to rostopic: {}".format(self.pub_topic))

    # def speak(self, text):
    #     global speak_pub
    #     # Run the espeak subprocess asynchronously to avoid blocking
    #     threading.Thread(target=subprocess.run, args=(['espeak', text],)).start()
    #     if speak_pub:
    #         speak_pub.publish(text)  # Publish the text to the ROS topic


    def publish_object_pose(self, detection, frame):
        """
        Publishes the object pose after accumulating 10 detections.
        """
        # Determine object ID and label based on the detection
        if object_labels[detection.label] == 'Drone':
            obj_id = 101
            obj_name = "Drone"
        elif object_labels[detection.label] == 'Phone':
            obj_id = 102
            obj_name = "Phone"
        else:
            rospy.logwarn("Unknown object label detected") # Log warning for unknown lables
            return

        # Stop detecting if the object has already been published
        if self.object_corners[obj_id]['published']:
            rospy.loginfo(f"Object {obj_name} (ID {obj_id}) has already been published. Skipping further detections.")
            return

        # Convert bounding box from normalized values to pixel coordinates
        bounding_box = self.frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

        # Calculate bounding box corners in pixel coordinates
        x1, y1 = bounding_box[0], bounding_box[1]  # Top-left
        x2, y2 = bounding_box[2], bounding_box[1]  # Top-right
        x3, y3 = bounding_box[2], bounding_box[3]  # Bottom-right
        x4, y4 = bounding_box[0], bounding_box[3]  # Bottom-left

        # Update the coordinates and count for this object
        self.object_corners[obj_id]['sum_tl_x'] += x1
        self.object_corners[obj_id]['sum_tl_y'] += y1
        self.object_corners[obj_id]['sum_tr_x'] += x2
        self.object_corners[obj_id]['sum_tr_y'] += y2
        self.object_corners[obj_id]['sum_br_x'] += x3
        self.object_corners[obj_id]['sum_br_y'] += y3
        self.object_corners[obj_id]['sum_bl_x'] += x4
        self.object_corners[obj_id]['sum_bl_y'] += y4
        self.object_corners[obj_id]['count'] += 1

        # Only publish after 10 detections
        if self.object_corners[obj_id]['count'] == 10:
            # Calculate the average coordinates for each corner
            avg_tl_x = self.object_corners[obj_id]['sum_tl_x'] / 10
            avg_tl_y = self.object_corners[obj_id]['sum_tl_y'] / 10
            avg_tr_x = self.object_corners[obj_id]['sum_tr_x'] / 10
            avg_tr_y = self.object_corners[obj_id]['sum_tr_y'] / 10
            avg_br_x = self.object_corners[obj_id]['sum_br_x'] / 10
            avg_br_y = self.object_corners[obj_id]['sum_br_y'] / 10
            avg_bl_x = self.object_corners[obj_id]['sum_bl_x'] / 10
            avg_bl_y = self.object_corners[obj_id]['sum_bl_y'] / 10

            # Log the averaged corner coordinates
            rospy.loginfo(f"Averaged corners for {obj_name}: "
                          f"Top-left({avg_tl_x}, {avg_tl_y}), Top-right({avg_tr_x}, {avg_tr_y}), "
                          f"Bottom-right({avg_br_x}, {avg_br_y}), Bottom-left({avg_bl_x}, {avg_bl_y})")

            # Create Float32MultiArray for object pose in the required format
            msg = Float32MultiArray()
            msg.data = [
                obj_id,              # Class ID
                avg_tl_x, avg_tl_y,  # Average top-left corner
                avg_tr_x, avg_tr_y,  # Average top-right corner
                avg_br_x, avg_br_y,  # Average bottom-right corner
                avg_bl_x, avg_bl_y,  # Average bottom-left corner
                (avg_br_x - avg_tl_x) / frame.shape[1],  # Width (normalized)
                (avg_br_y - avg_tl_y) / frame.shape[0]   # Height (normalized)
            ]

            # Publish the message
            self.pub_object_detect.publish(msg)
            rospy.loginfo(f"Published object pose for {obj_name} (ID {obj_id})")
            
            # New Speak Functionality: Construct and publish the message to be spoken
            speaker_msg = String()
            speaker_msg.data = f"Detected Target ID: {obj_id}, Position X: {avg_tl_x:.2f}, Y: {avg_tl_y:.2f}"
            pub_speaker.publish(speaker_msg)

            # self.speak(f"Detected Target: {object_labels[detection.label]}")
            # os.system(f"espeak 'Detected Target: {object_labels[detection.label]}'") #use espeak

            # Mark as published to stop further messages
            self.object_corners[obj_id]['published'] = True

    def publish_camera_info(self, timer=None):
        camera_info_msg = CameraInfo()
        camera_info_msg.header.frame_id = "camera_frame"
        camera_info_msg.height = self.nn_shape_h
        camera_info_msg.width = self.nn_shape_w
        # Set intrinsic camera matrix K
        camera_info_msg.K = [615.381, 0.0, 320.0, 0.0, 615.381, 240.0, 0.0, 0.0, 1.0]
        # Set distortion coefficients
        camera_info_msg.D = [-0.10818, 0.12793, 0.00000, 0.00000, -0.04204]
         # Set intrinsic camera matrix R
        camera_info_msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        # Camera projection matrix P
        camera_info_msg.P = [615.381, 0.0, 320.0, 0.0, 0.0, 615.381, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        camera_info_msg.distortion_model = "plumb_bob" # Set distortion model
        camera_info_msg.header.stamp = rospy.Time.now() # Set timestamp
        # Publish camera info message
        self.pub_cam_inf.publish(camera_info_msg)
    # Main loop to process the camera input and detections
    def run(self):
        modelPathName = f'{modelsPath}/{modelName}/{modelName}.blob' # Path to the bolb model life
        nnPath = str((Path(__file__).parent / Path(modelPathName)).resolve().absolute()) # Full path to model
        pipeline = self.createPipeline(nnPath) # Create DepthAI pipeline

        with dai.Device() as device:
            cams = device.getConnectedCameras()
            depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams # Check if depth cameras are enabled
            if cam_source != "rgb" and not depth_enabled:
                raise RuntimeError("Unable to run the experiment on {} camera! Available cameras: {}".format(cam_source, cams))
            device.startPipeline(pipeline) # Start the pipline

            q_nn_input = device.getOutputQueue(name="nn_input", maxSize=4, blocking=False) # Neural network input queue
            q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False) # Neural network output queue

            frame = None # Initialise frame
            detections = [] # Initialise detections list
            start_time = time.time() # Start time for FPS calculation
            counter = 0 # Initialise frame counter
            fps = 0 # Initialise FPS

            while True:
                found_classes = [] # Initialise list of found classes
                inRgb = q_nn_input.get() # Get Input Frame
                inDet = q_nn.get() # Get detection result

                if inRgb is not None:
                    frame = inRgb.getCvFrame() # Get OpenCV frame from input
                else:
                    print("Image empty, trying again...")
                    continue

                if inDet is not None:
                    detections = inDet.detections # Get detections from result
                    for detection in detections:
                        self.publish_object_pose(detection, frame)  # Publish object pose if detected
                    found_classes = np.unique([d.label for d in detections]) # Get unique detected classess
                    overlay = self.show_yolo(frame, detections) # Draw  bounding boxes on frame
                else:
                    print("Detection empty, trying again...") # Hanlde empty detection
                    continue

                if frame is not None:
                    # Display FPS and detected classes on the frame
                    cv2.putText(overlay, "NN fps: {:.2f}".format(fps), (2, overlay.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
                    cv2.putText(overlay, "Found classes {}".format(found_classes), (2, 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
                    self.publish_to_ros(frame) # Publish frame to ROS 
                    self.publish_detect_to_ros(overlay) # Publish detection overlay
                    self.publish_camera_info() # Publish camera info

                counter += 1
                if (time.time() - start_time) > 1:
                    fps = counter / (time.time() - start_time) # Update FPS calculation
                    counter = 0
                    start_time = time.time()
    # Function to publish frame to ROS
    def publish_to_ros(self, frame):
        msg_out = CompressedImage() # Create ROS compressed image message
        msg_out.header.stamp = rospy.Time.now() # Set timestamp
        msg_out.format = "jpeg" # Set format
        msg_out.header.frame_id = "home" # Set frame ID
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring() # Encode image as JPEG
        self.pub_image.publish(msg_out) # Publish compressed image
        msg_img_raw = self.br.cv2_to_imgmsg(frame, encoding="bgr8") # Convert frame to ROS Image
        self.pub_image_raw.publish(msg_img_raw) # Publish raw image
        # Function to publish frame to ROS
    def publish_detect_to_ros(self, frame):
        msg_out = CompressedImage() # Create ROS compressed image message
        msg_out.header.stamp = rospy.Time.now() # Set timestamp
        msg_out.format = "jpeg" # Set format
        msg_out.header.frame_id = "home" # Set frame ID
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring() # Encode image as JPEG
        self.pub_image_detect.publish(msg_out) # Publish detection overlay
    # Normalize bounding box coordinates to pixel values
    def frameNorm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0]) # Set normalization values
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int) # Clip values to [0,1] and convert to int
    
     # Function to draw YOLO detection results on the frame
    def show_yolo(self, frame, detections):
        color = (255, 0, 0) # Set bounding box color
        overlay = frame.copy() # Create a copy of the frame
        for detection in detections:
            bbox = self.frameNorm(overlay, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))  # Get bounding box
            cv2.putText(overlay, object_labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)  # Label class
            cv2.putText(overlay, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)  # Label confidence
            cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)  # Draw bounding box
        return overlay
    
     # Create DepthAI pipeline for the YOLO model
    def createPipeline(self, nnPath):
        pipeline = dai.Pipeline()  # Create pipeline
        pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2022_1)   # Set OpenVINO version
        # Create YOLO detection node
        detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
        detection_nn.setConfidenceThreshold(confidenceThreshold) # Set confidence threshold
        detection_nn.setNumClasses(classes)  # Set number of classes
        detection_nn.setCoordinateSize(coordinates)  # Set bounding box coordinate size
        detection_nn.setAnchors(anchors)  # Set anchors
        detection_nn.setAnchorMasks(anchorMasks)  # Set anchor masks
        detection_nn.setIouThreshold(iouThreshold) # Set IoU threshold
        detection_nn.setBlobPath(nnPath)   # Load neural network model
        detection_nn.setNumPoolFrames(4) # Set number of pool frames
        detection_nn.input.setBlocking(False)  # Non-blocking input
        detection_nn.setNumInferenceThreads(2) # Set number of inference threads

        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(self.nn_shape_w, self.nn_shape_h) # Set camera preview size
        cam.setInterleaved(False)  # Set interleaved mode to false
        cam.preview.link(detection_nn.input)  # Link camera preview to YOLO input
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR) # Set color order to BGR
        cam.setFps(40)  # Set camera FPS

        # Set up XLinkOut for the RGB input and detection output
        xout_rgb = pipeline.create(dai.node.XLinkOut) 
        xout_rgb.setStreamName("nn_input")   # Set stream name for RGB input
        xout_rgb.input.setBlocking(False)  # Non-blocking input

        detection_nn.passthrough.link(xout_rgb.input)  # Link YOLO output to XLinkOut

        xinDet = pipeline.create(dai.node.XLinkOut)
        xinDet.setStreamName("nn")   # Set stream name for YOLO detection output
        xinDet.input.setBlocking(False) # Non-blocking input

        detection_nn.out.link(xinDet.input) # Link YOLO output to XLinkOut

        return pipeline  # Return the pipeline

    # Cleanup function to close OpenCV windows
    def shutdown(self):
        cv2.destroyAllWindows()

# Main Code
def main():
    global pub_speaker
    # Set up the publisher for spoken text on the 'spoken_text' topic
    pub_speaker = rospy.Publisher('spoken_text', String, queue_size=10)

        # Initialize the ROS node
    rospy.init_node('depthai_node')
    dai_cam = DepthaiCamera()  # Initialize the DepthaiCamera class
    while not rospy.is_shutdown(): # Keep running the node until shutdown
        dai_cam.run()  # Run the camera processing
    dai_cam.shutdown()  # Cleanup on shutdown
# If this script is run as the main module, execute the main function
if __name__ == "__main__":
    main()

