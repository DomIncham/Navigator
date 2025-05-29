import os
import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TransformStamped
from cv_bridge import CvBridge

from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
import tf_transformations

from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import CompressedImage



class ArucoNode(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        self.bridge = CvBridge()

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco_marker_pose', 10)
        self.image_pub = self.create_publisher(Image, '/aruco_image', 10)

        # TF Broadcasters
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # Broadcast base_link → camera_link repeatedly to keep TF alive
        self.tf_static_timer = self.create_timer(1.0, self.broadcast_static_camera_tf)

        # === Load camera calibration ===
        package_share = get_package_share_directory('aruco_detector')
        calib_path = os.path.abspath(os.path.join(package_share, '..', '..', 'lib', 'aruco_detector'))
        self.camera_matrix = np.load(os.path.join(calib_path, 'camera_matrix.npy'))
        self.dist_coeffs = np.load(os.path.join(calib_path, 'dist_coeffs.npy'))

        # === Setup ArUco detector ===
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, cv2.aruco.DetectorParameters())

        # === Open USB camera ===
        self.cap = cv2.VideoCapture(0)

        # Timer loop ~20Hz
        self.timer = self.create_timer(0.05, self.process_frame)

    def broadcast_static_camera_tf(self):
        static_tf = TransformStamped()
        static_tf.header.stamp = self.get_clock().now().to_msg()
        static_tf.header.frame_id = 'odom'
        static_tf.child_frame_id = 'camera_link'

        # 🔧 Set the physical position of the camera on TurtleBot3 (adjust as needed)
        static_tf.transform.translation.x = 0.05
        static_tf.transform.translation.y = 0.0
        static_tf.transform.translation.z = 0.12

        quat = tf_transformations.quaternion_from_euler(0.0, -math.pi / 2, 0.0)
        static_tf.transform.rotation.x = quat[0]
        static_tf.transform.rotation.y = quat[1]
        static_tf.transform.rotation.z = quat[2]
        static_tf.transform.rotation.w = quat[3]

        self.static_tf_broadcaster.sendTransform(static_tf)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Camera frame not received.")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, 0.15, self.camera_matrix, self.dist_coeffs)

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i, marker_id in enumerate(ids):
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]

                # Publish pose
                pose = PoseStamped()
                pose.header.frame_id = "camera_link"
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = tvec

                rot, _ = cv2.Rodrigues(rvec)
                quat = tf_transformations.quaternion_from_matrix(
                    np.vstack((np.hstack((rot, [[0], [0], [0]])), [0, 0, 0, 1]))
                )

                pose.pose.orientation.x = quat[0]
                pose.pose.orientation.y = quat[1]
                pose.pose.orientation.z = quat[2]
                pose.pose.orientation.w = quat[3]

                self.pose_pub.publish(pose)

                # Publish TF for marker
                t = TransformStamped()
                t.header.stamp = pose.header.stamp
                t.header.frame_id = "camera_link"
                t.child_frame_id = f"aruco_marker_{marker_id[0]}"
                t.transform.translation.x = tvec[0]
                t.transform.translation.y = tvec[1]
                t.transform.translation.z = tvec[2]
                t.transform.rotation.x = quat[0]
                t.transform.rotation.y = quat[1]
                t.transform.rotation.z = quat[2]
                t.transform.rotation.w = quat[3]

                self.tf_broadcaster.sendTransform(t)

                # Draw Axis
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)

        # Publish camera image
        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = "camera_link"
        self.image_pub.publish(image_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoNode()
    rclpy.spin(node)
    node.cap.release()
    rclpy.shutdown()
