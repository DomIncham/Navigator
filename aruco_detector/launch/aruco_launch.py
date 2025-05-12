from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='aruco_detector',
            executable='aruco_node',
            name='aruco_detector',
            output='screen'
        )
    ])
