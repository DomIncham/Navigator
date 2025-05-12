import os
import math
import cv2
import numpy as np
import csv
from datetime import datetime, timezone


import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist,PoseStamped, TransformStamped, Point, PoseWithCovarianceStamped
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point

from visualization_msgs.msg import Marker
from tf2_ros import StaticTransformBroadcaster,TransformListener, Buffer
import tf_transformations

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from math import hypot



class ArucoNode(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        self.bridge = CvBridge()

        #self.set_parameters([
        #    Parameter('use_sim_time', Parameter.Type.BOOL, True)
        #])


        self.pose_pub = self.create_publisher(PoseStamped, '/aruco_marker_pose', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/swerve_drive/cmd_vel', 10)
        self.marker_pub = self.create_publisher(Marker, '/aruco_path_marker', 10)
        self.marker_path_pub = self.create_publisher(Marker, '/aruco_marker_path_marker', 10)
        self.expected_path_pub = self.create_publisher(Marker, '/expected_path_marker', 10)
        self.tb3_marker_pub = self.create_publisher(Marker, '/turtlebot3_path_marker', 10)

        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.turtlebot3_pose_callback, 10)

        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.broadcast_static_camera_tf()

        self.latest_transformed_aruco_point = None
        self.path_timestamps = []  # สำหรับ timestamp sync ทั้ง aruco และ turtlebot3


        self.x_center = 0.0
        self.x_tolerance = 0.10
        self.z_center = 0.40
        self.z_tolerance = 0.06
        self.x_min = self.x_center - self.x_tolerance
        self.x_max = self.x_center + self.x_tolerance
        self.z_min = self.z_center - self.z_tolerance
        self.z_max = self.z_center + self.z_tolerance

        self.last_cmd = ("INIT", 0.0, 0.0, 0.0, 0.0)
        self.last_marker_time = self.get_clock().now()
        self.marker_visible = False
        self.marker_lost = False
        self.latest_pose = None
        self.marker_paths = {}
        self.current_marker_id = None
        self.path_points = []
        self.tb3_path_points = []
        self.expected_path_points = [
            Point(x=-0.6, y=0.0, z=0.0),
            Point(x=1.0, y=0.0, z=0.0),
            Point(x=1.0, y=-3.0, z=0.0),
            Point(x=4.5, y=-3.0, z=0.0),  
        ]

        self.timer_detect = self.create_timer(0.07, self.aruco_detection_loop)
        self.timer_control = self.create_timer(0.01, self.control_loop)
        self.safety_timer = self.create_timer(0.1, self.check_marker_timeout)
        self.expected_path_timer = self.create_timer(1.0, self.publish_expected_path_marker)

        package_share = get_package_share_directory('aruco_detector')
        calib_path = os.path.abspath(os.path.join(package_share, '..', '..', 'lib', 'aruco_detector'))
        self.camera_matrix = np.load(os.path.join(calib_path, 'camera_matrix.npy'))
        self.dist_coeffs = np.load(os.path.join(calib_path, 'dist_coeffs.npy'))

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, cv2.aruco.DetectorParameters())

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 20)

        self.alignment_in_progress = False
        self.seeking_id42 = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.marker_path_in_map = []  # สำหรับเก็บเส้นทางใน map frame

    def broadcast_static_camera_tf(self):
        static_tf = TransformStamped()
        static_tf.header.stamp = self.get_clock().now().to_msg()
        #static_tf.header.stamp = rclpy.time.Time().to_msg() 
        static_tf.header.frame_id = 'base_link'
        static_tf.child_frame_id = 'camera_link'
        static_tf.transform.translation.x = -0.1
        static_tf.transform.translation.y = 0.0
        static_tf.transform.translation.z = 0.12
        quat = tf_transformations.quaternion_from_euler(0.0, -math.pi / 2, 0.0)
        static_tf.transform.rotation.x = quat[0]
        static_tf.transform.rotation.y = quat[1]
        static_tf.transform.rotation.z = quat[2]
        static_tf.transform.rotation.w = quat[3]
        self.static_tf_broadcaster.sendTransform(static_tf)

    def aruco_detection_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.marker_visible = False
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is None:
            self.marker_visible = False
            return

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, 0.15, self.camera_matrix, self.dist_coeffs
        )
        self.last_marker_time = self.get_clock().now()
        self.marker_visible = True

        for i, marker_id in enumerate(ids.flatten()):
            rvec, tvec = rvecs[i][0], tvecs[i][0]

            real_x = (tvec[0] * 100 - 0.1286) / 1.6391 / 100
            real_y = tvec[1]
            real_z = ((tvec[2] * 100 + 12.91) / 1.6267 / 100) - 6.6 / 100

            # ✅ สร้าง PoseStamped สำหรับ publish เท่านั้น
            pose = PoseStamped()
            pose.header.frame_id = "camera_link"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = real_x
            pose.pose.position.y = real_y
            pose.pose.position.z = real_z
            pose.pose.orientation.w = 1.0
            

            self.latest_raw_pose = pose 
            self.latest_pose = pose
            self.pose_pub.publish(pose)
            self.current_marker_id = marker_id


        self.publish_aruco_path_in_map()


    def control_loop(self):
        if not self.marker_visible or self.latest_pose is None:
            return

        x = self.latest_pose.pose.position.x
        z = self.latest_pose.pose.position.z
        
        if self.current_marker_id == 41 or self.current_marker_id == 43 :
            if x < self.x_min:
                self.send_swerve_command(f"Align-to-Left-Marker-{self.current_marker_id}", 1.0, 0.0, 0.0, 1.0)
                self.alignment_in_progress = True
            elif x > self.x_max:
                self.send_swerve_command(f"Align-to-Right-Marker-{self.current_marker_id}", 1.0, 0.0, 0.0, 1.0)
                self.alignment_in_progress = True
            elif self.alignment_in_progress:
                self.send_stop_command()
                self.alignment_in_progress = False
                if self.current_marker_id == 41:
                    self.send_swerve_command("Rotate-to-Find-ID42", 0.0, 0.0, -3.0, 1.0)
                    self.seeking_id42 = True
                elif self.current_marker_id == 43:
                    self.send_swerve_command("Rotate-to-Find-ID42", 0.0, 0.0, 3.0, 1.0)
                    self.seeking_id42 = True

        if self.current_marker_id == 42:
            if x < self.x_min:
                self.send_swerve_command("Forward-Left", 0.7, 0.7, 0.0, 1.0)
            elif x > self.x_max:
                self.send_swerve_command("Forward-Right", 0.7, -0.7, 0.0, 1.0)
            elif z > self.z_max:
                self.send_swerve_command("Forward", 1.0, 0.0, 0.0, 1.0)
            elif z < self.z_min:
                self.send_swerve_command("Backward", -1.0, 0.0, 0.0, 1.0)
            else:
                self.send_stop_command()

    def send_swerve_command(self, label, x, y, wz, scale_speed):
        if label == self.last_cmd[0]:
            return
        twist = Twist()
        twist.linear.x = x * scale_speed
        twist.linear.y = y * scale_speed
        twist.angular.z = wz * scale_speed
        self.cmd_vel_pub.publish(twist)
        self.last_cmd = (label, x, y, wz, scale_speed)
        self.get_logger().info(f"[{label}] x={x:.2f} y={y:.2f} wz={wz:.2f}")

    def send_stop_command(self):
        label, x, y, wz, _ = self.last_cmd
        self.send_swerve_command("STOP", x, y, wz, 0.01)

    def check_marker_timeout(self):
        now = self.get_clock().now()
        elapsed = now - self.last_marker_time
        if elapsed > Duration(seconds=0.3):
            if not self.marker_lost:
                self.send_stop_command()
                self.marker_lost = True
        else:
            self.marker_lost = False

    def publish_aruco_path_in_map(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "aruco_path_map"
        marker.id = 100
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # ✅ เปลี่ยนจากแสดง 20 จุดล่าสุด → แสดงทั้งหมด
        marker.points = self.marker_path_in_map

        marker.scale.x = 0.03
        marker.color.r = 0.96
        marker.color.g = 0.43
        marker.color.b = 0.00
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0  # optional เพื่อความสมบูรณ์
        self.marker_pub.publish(marker)

    def publish_expected_path_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "expected_path"
        marker.id = 3
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.points = self.expected_path_points
        marker.scale.x = 0.03
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.expected_path_pub.publish(marker)

    def turtlebot3_pose_callback(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        self.tb3_path_points.append(Point(x=p.x, y=p.y, z=p.z))

        if hasattr(self, "latest_raw_pose"):
            try:
                transform = self.tf_buffer.lookup_transform(
                    'map', 'camera_link', rclpy.time.Time(), timeout=Duration(seconds=1.0)
                )

                raw_point = PointStamped()
                raw_point.header = self.latest_raw_pose.header
                raw_point.point = self.latest_raw_pose.pose.position

                from tf2_geometry_msgs import do_transform_point
                transformed = do_transform_point(raw_point, transform)

                self.marker_path_in_map.append(transformed.point)

            except Exception as e:
                self.get_logger().warn(f"[aruco-sync] TF failed: {e}")

        self.publish_tb3_path_marker()
        self.publish_aruco_path_in_map()
        self.path_timestamps.append(self.get_clock().now().nanoseconds)


    def publish_tb3_path_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "turtlebot3_path"
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.points = self.tb3_path_points
        marker.scale.x = 0.03
        marker.color.r = 0.96
        marker.color.g = 0.43
        marker.color.b = 0.86
        marker.color.a = 1.0
        self.tb3_marker_pub.publish(marker)

    def export_all_paths_to_csv(self, filename: str):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        len_sync = min(len(self.marker_path_in_map), len(self.tb3_path_points), len(self.path_timestamps))

        def closest_expected(p):
            return min(self.expected_path_points, key=lambda ep: hypot(ep.x - p.x, ep.y - p.y))

        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp_ns', 'timestamp_s', 'timestamp_readable',
                'expected_x', 'expected_y', 'expected_z',
                'swerve_x', 'swerve_y', 'swerve_z',
                'turtlebot3_x', 'turtlebot3_y', 'turtlebot3_z'
            ])

            for i in range(len_sync):
                t_ns = self.path_timestamps[i]
                t_s = t_ns / 1e9
                t_dt = datetime.fromtimestamp(t_s).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                row = [t_ns, f"{t_s:.6f}", t_dt]

                # ✅ ใช้ตำแหน่งของ TurtleBot3 ในการหาจุด expected ที่ใกล้ที่สุด
                p_tb3 = self.tb3_path_points[i]
                p_expected = closest_expected(p_tb3)
                row.extend([p_expected.x, p_expected.y, p_expected.z])

                # ✅ Swerve Robot Path
                p_swerve = self.marker_path_in_map[i]
                row.extend([p_swerve.x, p_swerve.y, p_swerve.z])

                # ✅ TurtleBot3 Path
                row.extend([p_tb3.x, p_tb3.y, p_tb3.z])

                writer.writerow(row)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = '/workspaces/ros2-workspace/Path_log'
        node.export_all_paths_to_csv(f'{log_dir}/all_paths_{timestamp}.csv')
        node.cap.release()
        node.destroy_node()

        # ✅ เช็ค context ว่ายังเปิดอยู่ก่อน shutdown
        if rclpy.ok():
            rclpy.shutdown()
