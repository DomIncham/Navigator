FROM ros:humble-ros-base

ARG USERNAME=rosdev
ARG UID=1000
ARG GID=$UID

# Install some dependencies packages
RUN apt update -q \
    && apt upgrade -q -y \
    && apt install -y --no-install-recommends \
    software-properties-common \
    python3-pip \
    xauth \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create and switch to user
RUN groupadd -g $GID $USERNAME \
    && useradd -lm -u $UID -g $USERNAME -s /bin/bash $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER $USERNAME

# Create workspace so that user own this directory
RUN mkdir -p /home/$USERNAME/ros2_ws/src
WORKDIR /home/$USERNAME/ros2_ws

# Copy configuration files
RUN echo 'source /opt/ros/'$ROS_DISTRO'/setup.bash' >> /home/$USERNAME/.bashrc \
    && echo 'source /home/'$USERNAME'/ros2_ws/install/setup.bash' >> /home/$USERNAME/.bashrc

# Setup entrypoint
COPY ./ros_entrypoint.sh /
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]



#insert my Dockerfile

# Upgrade all packages
RUN sudo apt update && sudo apt upgrade -y


# Install essential packages
RUN sudo apt install -y wget

# Install dependant ROS2 Package
RUN sudo apt install python3-argcomplete python3-colcon-common-extensions libboost-system-dev build-essential -y
RUN sudo apt install ros-humble-hls-lfcd-lds-driver -y
RUN sudo apt install ros-humble-turtlebot3-msgs -y
RUN sudo apt install ros-humble-dynamixel-sdk -y
RUN sudo apt install libudev-dev -y

# Add sourcing ROS setup.bash to .bashrc
RUN echo 'source /opt/ros/${ROS_DISTRO}/setup.bash' >> ~/.bashrc
RUN echo 'export ROS_DOMAIN_ID=30 #TURTLEBOT3' >> ~/.bashrc
RUN echo 'source /workspaces/ros2-workspace/turtlebot3_ws/install/setup.bash' >> ~/.bashrc
RUN echo 'export RMW_IMPLEMENTATION=rmw_fastrtps_cpp' >> ~/.bashrc
RUN echo 'export TURTLEBOT3_MODEL=burger' >> ~/.bashrc
RUN echo 'export LDS_MODEL=LDS-01' >> ~/.bashrc
RUN echo 'export OPENCR_PORT=/dev/ttyACM0' >> ~/.bashrc
RUN echo 'export OPENCR_MODEL=burger' >> ~/.bashrc
RUN echo 'export PYTHONPATH=$PYTHONPATH:/opt/ros/humble/lib/python3.10/site-packages' >> ~/.bashrc
#OPENCR
RUN sudo dpkg --add-architecture armhf 
RUN sudo apt-get update -y
RUN sudo apt-get install libc6:armhf -y
RUN sudo apt install -y ros-humble-robot-localization -y
RUN sudo sudo apt install python3-pip -y
RUN pip3 install opencv-python 
RUN sudo apt install qtwayland5 -y
RUN pip install opencv-contrib-python 
RUN sudo apt install ros-humble-cv-bridge -y
RUN sudo apt-get install ros-humble-vision-opencv -y
RUN sudo apt install ros-humble-tf-transformations -y
RUN sudo apt install ros-humble-image-transport ros-humble-image-transport-plugins -y
RUN sudo apt-get install v4l-utils -y





