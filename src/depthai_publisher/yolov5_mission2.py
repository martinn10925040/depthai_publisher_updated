 
pipeline = None
cam_source = 'rgb' #'rgb', 'left', 'right'
cam=None
# sync outputs
syncNN = True

#!/usr/bin/env python3

# Libraries and Parameters Setup
from pathlib import Path
import os  #espeak library
import time
import json     # Configuration uses JSON format
import cv2
import numpy as np
import depthai as dai
import rospy
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from std_msgs.msg import String
import subprocess
import threading

# YOLO Config File and Model Parameters
modelsPath = "/home/uavteam2/QUT_EGH450/src/depthai_publisher/src/depthai_publisher/models"
modelName = 'best_mission2'
confJson = 'best_mission2.json'

configPath = Path(f'{modelsPath}/{modelName}/{confJson}')
if not configPath.exists():
    raise ValueError(f"Path {configPath} does not exist!")

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

class DepthaiCamera:
    fps = 30.0
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

    def speak(self, text):
        global speak_pub
        # Run the espeak subprocess asynchronously to avoid blocking
        threading.Thread(target=subprocess.run, args=(['espeak', text],)).start()
        if speak_pub:
            speak_pub.publish(text)  # Publish the text to the ROS topic


    def publish_object_pose(self, detection, frame):
        """
        Publishes the object pose after accumulating 10 detections.
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
            
            self.speak(f"Detected Target: {object_labels[detection.label]}")
            # os.system(f"espeak 'Detected Target: {object_labels[detection.label]}'") #use espeak

            # Mark as published to stop further messages
            self.object_corners[obj_id]['published'] = True

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
                        self.publish_object_pose(detection, frame)  # Publish object pose
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
    global speak_pub 
# Set up the publisher for spoken text on the 'spoken_text' topic
    speak_pub = rospy.Publisher('spoken_text', String, queue_size=10)
    rospy.init_node('depthai_node')
    dai_cam = DepthaiCamera()
    while not rospy.is_shutdown():
        dai_cam.run()
    dai_cam.shutdown()

if __name__ == "__main__":
    main()

