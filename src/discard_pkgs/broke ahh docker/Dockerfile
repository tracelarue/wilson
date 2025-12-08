FROM ros:humble-ros-base

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

### Installations split up for debugging and layer caching

# Install basic development tools
RUN apt-get update && apt-get install -y \
  nano \
  vim \
  build-essential \
  cmake \
  python3-dev \
  python3-pip \
  python3-colcon-common-extensions \
  && rm -rf /var/lib/apt/lists/*

# Install audio and hardware libraries
RUN apt-get update && apt-get install -y \
  libportaudio2 \
  portaudio19-dev \
  libserial-dev \
  v4l-utils \
  && rm -rf /var/lib/apt/lists/*

# Install core ROS packages
RUN apt-get update && apt-get install -y \
  ros-humble-std-msgs \
  ros-humble-rosidl-default-generators \
  ros-humble-rosidl-default-runtime \
  ros-humble-rviz2 \
  ros-humble-gazebo-* \
  ros-humble-ros-gz \
  ros-humble-xacro \
  ros-humble-rosbridge-server \
  && rm -rf /var/lib/apt/lists/*

# Install ROS control and hardware interface packages
RUN apt-get update && apt-get install -y \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-controller-manager \
  ros-humble-twist-mux \
  && rm -rf /var/lib/apt/lists/*

# Install navigation and SLAM packages
RUN apt-get update && apt-get install -y \
  ros-humble-navigation2 \
  ros-humble-nav2-bringup \
  ros-humble-slam-toolbox \
  ros-humble-teleop-twist-keyboard \
  && rm -rf /var/lib/apt/lists/*

# Install camera and visualization packages
RUN apt-get update && apt-get install -y \
  ros-humble-v4l2-camera \
  ros-humble-image-transport-plugins \
  ros-humble-rqt-image-view \
  && rm -rf /var/lib/apt/lists/*

# Install MoveIt packages
RUN apt-get update && apt-get install -y \
  ros-humble-moveit-* \
  && rm -rf /var/lib/apt/lists/*

# Install Python scientific packages
RUN apt-get update && apt-get install -y \
  python3-scipy \
  && rm -rf /var/lib/apt/lists/*

# pip installations 
RUN pip3 install --upgrade pip setuptools packaging wheel && \
  pip3 install --no-cache-dir \
  pyserial \
  inputs \
  numpy \
  opencv-python \
  pyaudio \
  pysocks \
  websocket-client \
  google-genai \
  pillow \
  mss \
  "numpy<2.0.0"

RUN usermod -a -G dialout ${USERNAME}
RUN usermod -a -G video ${USERNAME}

RUN echo "export ROS_DOMAIN_ID=7" >> ~/.bashrc \
 && echo "export GOOGLE_API_KEY=AIzaSyAO4BgOBjkMofvLpboXv16V9CByIAh0ARU" >> ~/.bashrc \
 && echo "export QT_AUTO_SCREEN_SCALE_FACTOR=0" >> ~/.bashrc \
 && echo "export QT_SCALE_FACTOR=1" >> ~/.bashrc \
 && echo "export DISPLAY=:0" >> ~/.bashrc \
 && echo "export GAZEBO_MODEL_PATH=/wilson/install/wilson/share/wilson/models" >> ~/.bashrc

# Copy the entrypoint and bashrc scripts so we have
# our container's environment set up correctly
COPY entrypoint.sh /entrypoint.sh
COPY bashrc /home/${USERNAME}/.bashrc

# Set up entrypoint and default command
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
CMD ["bash"]
