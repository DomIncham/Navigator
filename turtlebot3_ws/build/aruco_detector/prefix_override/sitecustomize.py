import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/workspaces/ros2-workspace/turtlebot3_ws/install/aruco_detector'
