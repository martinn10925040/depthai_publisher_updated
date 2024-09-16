 #!/usr/bin/env python3

# '''
# Run as:
# # check model path line ~30is
# rosrun depthai_publisher dai_publisher_yolov5_runner_2
# '''
# ############################### ############################### Libraries ###############################
# from pathlib import Path
# import threading
# import time
# import sys
# import json     # Configuration uses JSON format
# import cv2
# import numpy as np
# import depthai as dai
# import rospy
# from sensor_msgs.msg import CompressedImage, Image, CameraInfo
# from std_msgs.msg import Float32MultiArray
# from cv_bridge import CvBridge

# ############################### ############################### Parameters ###############################
 # Global variables to deal with pipeline creation
pipeline = None
cam_source = 'rgb' #'rgb', 'left', 'right'
cam=None
# sync outputs
syncNN = True
# model path
#modelsPath = "/home/uavteam2/QUT_EGH450/src/depthai_publisher/src/depthai_publisher/models"
# modelName = 'exp31Yolov5_ov21.4_6sh'
#modelName = 'best_mission2'
# confJson = 'exp31Yolov5.json'
#confJson = 'best_mission2.json'

# ################################  Yolo Config File
# # parse config
# configPath = Path(f'{modelsPath}/{modelName}/{confJson}')
# if not configPath.exists():
#     raise ValueError("Path {} does not exist!".format(configPath))

# with configPath.open() as f:
#     config = json.load(f)
# nnConfig = config.get("nn_config", {})

# # Extract metadata
# metadata = nnConfig.get("NN_specific_metadata", {})
# classes = metadata.get("classes", {})
# coordinates = metadata.get("coordinates", {})
# anchors = metadata.get("anchors", {})
# anchorMasks = metadata.get("anchor_masks", {})
# iouThreshold = metadata.get("iou_threshold", {})
# confidenceThreshold = metadata.get("confidence_threshold", {})
# # Parse labels
# nnMappings = config.get("mappings", {})
# object_labels = nnMappings.get("labels", {})

# class DepthaiCamera():
#     # res = [416, 416]
#     fps = 20.0

#     pub_topic = '/depthai_node/image/compressed'
#     pub_topic_raw = '/depthai_node/image/raw'
#     pub_topic_detect = '/depthai_node/detection/compressed'
#     pub_topic_cam_inf = '/depthai_node/camera/camera_info'
#     pub_roi_data = '/roi_data_publisher' # New topic for ROI publishing

#     def __init__(self):
#         self.pipeline = dai.Pipeline()

#          # Input image size
#         if "input_size" in nnConfig:
#             self.nn_shape_w, self.nn_shape_h = tuple(map(int, nnConfig.get("input_size").split('x')))

#         # Pulbish ros image data and ROI
#         self.pub_image = rospy.Publisher(self.pub_topic, CompressedImage, queue_size=10)
#         self.pub_image_raw = rospy.Publisher(self.pub_topic_raw, Image, queue_size=10)
#         self.pub_image_detect = rospy.Publisher(self.pub_topic_detect, CompressedImage, queue_size=10)
#         # Create a publisher for the CameraInfo topic
#         self.pub_cam_inf = rospy.Publisher(self.pub_topic_cam_inf, CameraInfo, queue_size=50)
#         self.pub_roi_info = rospy.Publisher(self.pub_roi_data, Float32MultiArray, queue_size=50)
#         # Create a timer for the callback

#         self.timer = rospy.Timer(rospy.Duration(1.0 / 10), self.publish_camera_info, oneshot=False)
#         # Set to track already published object IDs
#         self.published_obj_ids = set() 

#         rospy.loginfo("Publishing images to rostopic: {}".format(self.pub_topic))

#         self.br = CvBridge()
#         def send_roi_info(self, detection, frame):
#             # Assign an ID to deteced objects
#             if object_labels[detection.label] == 'Drone':
#                 obj_id = 103
#             elif object_labels[detection.label] == 'Phone':
#                 obj_id = 104
#             else:
#                 return
#             # Avoid re-publishing already sent IDs:
#             if obj_id in self.published_obj_ids:
#                 return
#             self.published_obj_ids.add(obj_id)
#             # Clone the frame
#             uframe = frame.copy()
#             # Convert bbox from normalised values to pixel coordinates
#             bounding_box =  self.normalize_bounding_box(uframe, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
#             # Calculate bbox size relative to the frame dimensions
#             width = bounding_box[2] - bounding_box[0]
#             height = bounding_box[3] - bounding_box[1]
#             normalized_width = width / uframe.shape[1]
#             normalized_height = height / uframe.shape[0]

#             # Create Float32MultiArray for bounding box info
#             msg = Float32MultiArray()
#             msg.data = [obj_id, bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3], normalized_width, normalized_height]
#             self.pub_roi_info.publish(msg)
#             rospy.loginfo("ROI Data published")
        


#             # rospy.on_shutdown(lambda: self.shutdown())

#     def publish_camera_info(self, timer=None):
#         # Create a publisher for the CameraInfo topic

#         # Create a CameraInfo message
#         camera_info_msg = CameraInfo()
#         camera_info_msg.header.frame_id = "camera_frame"
#         camera_info_msg.height = self.nn_shape_h # Set the height of the camera image
#         camera_info_msg.width = self.nn_shape_w  # Set the width of the camera image

#         # Set the camera intrinsic matrix (fx, fy, cx, cy)
#         camera_info_msg.K = [615.381, 0.0, 320.0, 0.0, 615.381, 240.0, 0.0, 0.0, 1.0]
#         # Set the distortion parameters (k1, k2, p1, p2, k3)
#         camera_info_msg.D = [-0.10818, 0.12793, 0.00000, 0.00000, -0.04204]
#         # Set the rectification matrix (identity matrix)
#         camera_info_msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
#         # Set the projection matrix (P)
#         camera_info_msg.P = [615.381, 0.0, 320.0, 0.0, 0.0, 615.381, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
#         # Set the distortion model
#         camera_info_msg.distortion_model = "plumb_bob"
#         # Set the timestamp
#         camera_info_msg.header.stamp = rospy.Time.now()

#         self.pub_cam_inf.publish(camera_info_msg)  # Publish the camera info message

#     def rgb_camera(self):
#         cam_rgb = self.pipeline.createColorCamera()
#         cam_rgb.setPreviewSize(self.res[0], self.res[1])
#         cam_rgb.setInterleaved(False)
#         cam_rgb.setFps(self.fps)

#         # Def xout / xin
#         ctrl_in = self.pipeline.createXLinkIn()
#         ctrl_in.setStreamName("cam_ctrl")
#         ctrl_in.out.link(cam_rgb.inputControl)

#         xout_rgb = self.pipeline.createXLinkOut()
#         xout_rgb.setStreamName("video")

#         cam_rgb.preview.link(xout_rgb.input)

#     def run(self):
#         #self.rgb_camera()
#         ############################### Run Model ###############################
#         # Pipeline defined, now the device is assigned and pipeline is started
#         pipeline = None
#         # Get argument first
#         # Model parameters
#         modelPathName = f'{modelsPath}/{modelName}/{modelName}.blob'
#         print(metadata)
#         nnPath = str((Path(__file__).parent / Path(modelPathName)).resolve().absolute())
#         print(nnPath)

#         pipeline = self.createPipeline(nnPath)

#         with dai.Device() as device:
#             cams = device.getConnectedCameras()
#             depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
#             if cam_source != "rgb" and not depth_enabled:
#                 raise RuntimeError("Unable to run the experiment on {} camera! Available cameras: {}".format(cam_source, cams))
#             device.startPipeline(pipeline)

#             # Output queues will be used to get the rgb frames and nn data from the outputs defined above
#             q_nn_input = device.getOutputQueue(name="nn_input", maxSize=4, blocking=False)
#             q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

#             frame = None
#             detections = []
#             start_time = time.time()
#             counter = 0
#             fps = 0
            
#             olor2 = (255, 255, 255)
#             layer_info_printed = False
#             dims = None

#             while True:
#                 found_classes = []
#                 # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
#                 inRgb = q_nn_input.get()
#                 inDet = q_nn.get()

#                 if inRgb is not None:
#                     frame = inRgb.getCvFrame()


#                 else:
#                     print("Detection empty, trying again...")
#                     continue
                
#                 if inDet is not None:
#                     detections = inDet.detections
#                     # print(detections)
#                     for detection in detections:
#                         # print(detection)
#                         # print("{},{},{},{},{},{},{}".format(detection.label,labels[detection.label],detection.confidence,detection.xmin, detection.ymin, detection.xmax, detection.ymax))
#                         bounding_box = self.frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
#                     # Print the bounding box coordinates
#                         print(f"Bounding box coordinates: xmin: {bounding_box[0]}, ymin: {bounding_box[1]}, xmax: {bounding_box[2]}, ymax: {bounding_box[3]}")
#                         found_classes.append(detection.label)
#                         # print(dai.ImgDetection.getData(detection))
#                     found_classes = np.unique(found_classes)
                    
#                     # print(found_classes)
#                     overlay = self.show_yolo(frame, detections)
#                 else:
#                     print("Detection empty, trying again...")
#                     continue

#                 if frame is not None:
#                     cv2.putText(overlay, "NN fps: {:.2f}".format(fps), (2, overlay.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
#                     cv2.putText(overlay, "Found classes {}".format(found_classes), (2, 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
#                     # cv2.imshow("nn_output_yolo", overlay)
#                     self.publish_to_ros(frame)
#                     self.publish_detect_to_ros(overlay)
#                     self.publish_camera_info()

#                 ## Function to compute FPS
#                 counter+=1
#                 if (time.time() - start_time) > 1 :
#                     fps = counter / (time.time() - start_time)

#                     counter = 0
#                     start_time = time.time()


#             # with dai.Device(self.pipeline) as device:
#             #     video = device.getOutputQueue(name="video", maxSize=1, blocking=False)

#             #     while True:
#             #         frame = video.get().getCvFrame()

#             #         self.publish_to_ros(frame)
#             #         self.publish_camera_info()

#     def publish_to_ros(self, frame):
#         msg_out = CompressedImage()
#         msg_out.header.stamp = rospy.Time.now()
#         msg_out.format = "jpeg"
#         msg_out.header.frame_id = "home"
#         msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
#         self.pub_image.publish(msg_out)
#         # Publish image raw
#         msg_img_raw = self.br.cv2_to_imgmsg(frame, encoding="bgr8")
#         self.pub_image_raw.publish(msg_img_raw)

#     def publish_detect_to_ros(self, frame):
#         msg_out = CompressedImage()
#         msg_out.header.stamp = rospy.Time.now()
#         msg_out.format = "jpeg"
#         msg_out.header.frame_id = "home"
#         msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
#         self.pub_image_detect.publish(msg_out)
        
#     ############################### ############################### Functions ###############################
#     ######### Functions for Yolo Decoding
#     # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
#     def frameNorm(self, frame, bbox):
#         normVals = np.full(len(bbox), frame.shape[0])
#         normVals[::2] = frame.shape[1]
#         return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

#     def show_yolo(self, frame, detections):
#         color = (255, 0, 0)
#         # Both YoloDetectionNetwork and MobileNetDetectionNetwork output this message. This message contains a list of detections, which contains label, confidence, and the bounding box information (xmin, ymin, xmax, ymax).
#         overlay =  frame.copy()
#         for detection in detections:
#             bbox = self.frameNorm(overlay, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
#             cv2.putText(overlay, object_labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
#             cv2.putText(overlay, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
#             cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
#         return overlay

#     # Start defining a pipeline
#     def createPipeline(self, nnPath):

#         pipeline = dai.Pipeline()

#         # pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)
#         # pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)
#         pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2022_1)

#         # Define a neural network that will make predictions based on the source frames
#         detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
#         # Network specific settings
#         detection_nn.setConfidenceThreshold(confidenceThreshold)
#         detection_nn.setNumClasses(classes)
#         detection_nn.setCoordinateSize(coordinates)
#         detection_nn.setAnchors(anchors)
#         detection_nn.setAnchorMasks(anchorMasks)
#         detection_nn.setIouThreshold(iouThreshold)
#         # generic nn configs
#         detection_nn.setBlobPath(nnPath)
#         detection_nn.setNumPoolFrames(4)
#         detection_nn.input.setBlocking(False)
#         detection_nn.setNumInferenceThreads(2)

#         # Define a source - color camera
#         if cam_source == 'rgb':
#             cam = pipeline.create(dai.node.ColorCamera)
#             cam.setPreviewSize(self.nn_shape_w,self.nn_shape_h)
#             cam.setInterleaved(False)
#             cam.preview.link(detection_nn.input)
#             cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
#             cam.setFps(40)
#             print("Using RGB camera...")
#         elif cam_source == 'left':
#             cam = pipeline.create(dai.node.MonoCamera)
#             cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
#             print("Using BW Left cam")
#         elif cam_source == 'right':
#             cam = pipeline.create(dai.node.MonoCamera)
#             cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)
#             print("Using BW Rigth cam")

#         if cam_source != 'rgb':
#             manip = pipeline.create(dai.node.ImageManip)
#             manip.setResize(self.nn_shape_w,self.nn_shape_h)
#             manip.setKeepAspectRatio(True)
#             # manip.setFrameType(dai.RawImgFrame.Type.BGR888p)
#             manip.setFrameType(dai.RawImgFrame.Type.RGB888p)
#             cam.out.link(manip.inputImage)
#             manip.out.link(detection_nn.input)

#         # Create outputs
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


# #### Main code that creates a depthaiCamera class and run it.
# def main():
#     rospy.init_node('depthai_node')
#     dai_cam = DepthaiCamera()

#     while not rospy.is_shutdown():
#         dai_cam.run()

#     dai_cam.shutdown()

#!/usr/bin/env python3

# Libraries and Parameters Setup
from pathlib import Path
import threading
import time
import sys
import json     # Configuration uses JSON format
import cv2
import numpy as np
import depthai as dai
import rospy
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

# YOLO Config File and Model Parameters
modelsPath = "/home/uavteam2/QUT_EGH450/src/depthai_publisher/src/depthai_publisher/models"
modelName = 'best_mission2'
confJson = 'best_mission2.json'

configPath = Path(f'{modelsPath}/{modelName}/{confJson}')
if not configPath.exists():
    raise ValueError("Path {} does not exist!".format(configPath))

with configPath.open() as f:
    config = json.load(f)

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

# DepthaiCamera Class Definition
class DepthaiCamera():
    fps = 20.0

    pub_topic = '/depthai_node/image/compressed'
    pub_topic_raw = '/depthai_node/image/raw'
    pub_topic_detect = '/depthai_node/detection/compressed'
    pub_topic_cam_inf = '/depthai_node/camera/camera_info'
    pub_object_pose = '/object_pose'  # Publisher topic to match the first script

    def __init__(self):
        self.pipeline = dai.Pipeline()

        if "input_size" in nnConfig:
            self.nn_shape_w, self.nn_shape_h = tuple(map(int, nnConfig.get("input_size").split('x')))

        # ROS Publishers
        self.pub_image = rospy.Publisher(self.pub_topic, CompressedImage, queue_size=10)
        self.pub_image_raw = rospy.Publisher(self.pub_topic_raw, Image, queue_size=10)
        self.pub_image_detect = rospy.Publisher(self.pub_topic_detect, CompressedImage, queue_size=10)
        self.pub_cam_inf = rospy.Publisher(self.pub_topic_cam_inf, CameraInfo, queue_size=50)
        self.pub_object_detect = rospy.Publisher(self.pub_object_pose, Float32MultiArray, queue_size=10)  # Publisher for object pose

        self.timer = rospy.Timer(rospy.Duration(1.0 / 10), self.publish_camera_info, oneshot=False)

        rospy.loginfo("Publishing images to rostopic: {}".format(self.pub_topic))

        self.br = CvBridge()

    def publish_object_pose(self, detection, frame):
        """
        Publishes the object pose to match the expected format in the first script.
        """
        # Assign object ID based on detection label
        if object_labels[detection.label] == 'Drone':
            obj_id = 101
            obj_name = "Drone"
        elif object_labels[detection.label] == 'Phone':
            obj_id = 102
            obj_name = "Phone"
        else:
            rospy.logwarn("Unknown object label detected")
            return

         # Convert bounding box from normalized values to pixel coordinates
        bounding_box = self.frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

    # Calculate bounding box corners in pixel coordinates
        x1, y1 = bounding_box[0], bounding_box[1]  # Top-left
        x2, y2 = bounding_box[2], bounding_box[1]  # Top-right
        x3, y3 = bounding_box[2], bounding_box[3]  # Bottom-right
        x4, y4 = bounding_box[0], bounding_box[3]  # Bottom-left

    # Calculate bounding box size relative to the frame dimensions
        width = bounding_box[2] - bounding_box[0]
        height = bounding_box[3] - bounding_box[1]

    # For physical dimensions, use normalized size or calibrate if possible
        normalized_width = width / frame.shape[1]
        normalized_height = height / frame.shape[0]

    # Create Float32MultiArray for object pose in the required format
        msg = Float32MultiArray()
        msg.data = [
            obj_id,  # Class ID
            x1, y1,  # Top-left corner
            x2, y2,  # Top-right corner
            x3, y3,  # Bottom-right corner
            x4, y4,  # Bottom-left corner
            normalized_width,  # Length in x (normalized or calibrated)
            normalized_height  # Length in y (normalized or calibrated)
            ]
    
    # Ensure the message has exactly 10 elements as expected
        if len(msg.data) != 11:
            rospy.logwarn("Data length is not 10; malformed target data.")
            return

    # Publish the message
        # Log the object ID and publish the message
        rospy.loginfo(f"Detected object: {obj_name} (ID: {obj_id})")
        print(f"Detected object: {obj_name} (ID: {obj_id})")  # Print to console for easier debugging
        self.pub_object_detect.publish(msg)
        # rospy.loginfo("Object pose data published for ID: {}".format(obj_id))

    def publish_camera_info(self, timer=None):
        camera_info_msg = CameraInfo()
        camera_info_msg.header.frame_id = "camera_frame"
        camera_info_msg.height = self.nn_shape_h
        camera_info_msg.width = self.nn_shape_w
        camera_info_msg.K = [615.381, 0.0, 320.0, 0.0, 615.381, 240.0, 0.0, 0.0, 1.0]
        camera_info_msg.D = [-0.10818, 0.12793, 0.00000, 0.00000, -0.04204]
        camera_info_msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        camera_info_msg.P = [615.381, 0.0, 320.0, 0.0, 0.0, 615.381, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        camera_info_msg.distortion_model = "plumb_bob"
        camera_info_msg.header.stamp = rospy.Time.now()

        self.pub_cam_inf.publish(camera_info_msg)

    def run(self):
        # Setup and start the pipeline
        modelPathName = f'{modelsPath}/{modelName}/{modelName}.blob'
        nnPath = str((Path(__file__).parent / Path(modelPathName)).resolve().absolute())
        pipeline = self.createPipeline(nnPath)

        with dai.Device() as device:
            cams = device.getConnectedCameras()
            depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
            if cam_source != "rgb" and not depth_enabled:
                raise RuntimeError("Unable to run the experiment on {} camera! Available cameras: {}".format(cam_source, cams))
            device.startPipeline(pipeline)

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
                    print("Image empty, trying again...")
                    continue
                
                if inDet is not None:
                    detections = inDet.detections
                    for detection in detections:
                        self.publish_object_pose(detection, frame)
                    found_classes = np.unique([d.label for d in detections])
                    overlay = self.show_yolo(frame, detections)
                else:
                    print("Detection empty, trying again...")
                    continue

                if frame is not None:
                    cv2.putText(overlay, "NN fps: {:.2f}".format(fps), (2, overlay.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
                    cv2.putText(overlay, "Found classes {}".format(found_classes), (2, 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
                    self.publish_to_ros(frame)
                    self.publish_detect_to_ros(overlay)
                    self.publish_camera_info()

                counter += 1
                if (time.time() - start_time) > 1:
                    fps = counter / (time.time() - start_time)
                    counter = 0
                    start_time = time.time()

    def publish_to_ros(self, frame):
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.header.frame_id = "home"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
        self.pub_image.publish(msg_out)
        msg_img_raw = self.br.cv2_to_imgmsg(frame, encoding="bgr8")
        self.pub_image_raw.publish(msg_img_raw)

    def publish_detect_to_ros(self, frame):
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.header.frame_id = "home"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
        self.pub_image_detect.publish(msg_out)

    def frameNorm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def show_yolo(self, frame, detections):
        color = (255, 0, 0)
        overlay = frame.copy()
        for detection in detections:
            bbox = self.frameNorm(overlay, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(overlay, object_labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(overlay, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        return overlay

    def createPipeline(self, nnPath):
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2022_1)
        detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
        detection_nn.setConfidenceThreshold(confidenceThreshold)
        detection_nn.setNumClasses(classes)
        detection_nn.setCoordinateSize(coordinates)
        detection_nn.setAnchors(anchors)
        detection_nn.setAnchorMasks(anchorMasks)
        detection_nn.setIouThreshold(iouThreshold)
        detection_nn.setBlobPath(nnPath)
        detection_nn.setNumPoolFrames(4)
        detection_nn.input.setBlocking(False)
        detection_nn.setNumInferenceThreads(2)

        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(self.nn_shape_w, self.nn_shape_h)
        cam.setInterleaved(False)
        cam.preview.link(detection_nn.input)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setFps(40)

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
    rospy.init_node('depthai_node')
    dai_cam = DepthaiCamera()
    while not rospy.is_shutdown():
        dai_cam.run()
    dai_cam.shutdown()
