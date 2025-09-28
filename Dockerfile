FROM osrf/ros:humble-desktop-full


# Create a non-root user
ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
  && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && mkdir /home/$USERNAME/.config && chown $USER_UID:$USER_GID /home/$USERNAME/.config

# Set up sudo
RUN apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# apt-get installations
RUN apt-get update \
  && apt-get install -y \
  nano \
  vim \
  build-essential \
  python3-scipy \
  libportaudio2 \
  portaudio19-dev \
  ros-humble-std-msgs \
  ros-humble-rosidl-default-generators \
  ros-humble-rosidl-default-runtime \
  cmake \
  libserial-dev \
  python3-dev \
  python3-colcon-common-extensions \
  python3-pip \
  v4l-utils \
  ros-humble-v4l2-camera \
  ros-humble-image-transport-plugins \
  ros-humble-rqt-image-view \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-controller-manager \
  ros-humble-xacro \
  ros-humble-teleop-twist-keyboard \
  ros-humble-slam-toolbox \
  ros-humble-twist-mux \
  ros-humble-navigation2 \
  ros-humble-nav2-bringup \
  ros-humble-gazebo-* \
  ros-humble-moveit-* \
  libopen3d-dev \
  ros-humble-rosbridge-server \
  dbus-x11 \
  at-spi2-core \
  && rm -rf /var/lib/apt/lists/*

RUN pip3 install \
  "numpy>=1.21.0,<2.0.0" \
  pyserial \
  inputs \
  ArducamDepthCamera \
  opencv-python \
  pyaudio \
  pysocks \
  websocket-client \
  google-genai \
  pyaudio \
  pillow \
  mss \
  python-dotenv

RUN apt-get update && apt-get install -y \
  gnome-terminal \
  tilix \
  && rm -rf /var/lib/apt/lists/*

RUN usermod -a -G dialout ${USERNAME}
RUN usermod -a -G video ${USERNAME}
RUN usermod -a -G audio ${USERNAME}

RUN echo "export ROS_DOMAIN_ID=7" >> ~/.bashrc \
 && echo "export QT_AUTO_SCREEN_SCALE_FACTOR=0" >> ~/.bashrc \
 && echo "export QT_SCALE_FACTOR=1" >> ~/.bashrc \
 && echo "export DISPLAY=:0" >> ~/.bashrc

 # Copy the entrypoint and bashrc scripts so we have
# our container's environment set up correctly
COPY entrypoint.sh /entrypoint.sh
COPY bashrc /home/${USERNAME}/.bashrc

# Set up entrypoint and default command
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
CMD ["bash"]
