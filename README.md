# wilson

![Wilson Robot](wilson.jpeg)

An AI enabled robot. Wilson can interact with the user and world through audio/visual input and audio output. Wilson's main directive is to retrieve the user's desired beverage from a mini fridge!

Wilson was custom designed in SolidWorks and 3D printed at home. 

## Try the simulation yourself! 🐳

<details>
<summary><strong>🐧 Linux Setup Instructions</strong></summary>

Follow these steps to set up Docker and run Wilson's simulation:

### Prerequisites Setup

**⚠️ Important:** You will need root or sudo access to complete these steps.

1. **Install Docker:**
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   ```

2. **Configure Docker permissions:**
   ```bash
   sudo groupadd docker
   sudo usermod -aG docker $USER
   ```

3. **Check Docker service:**
    ```bash
    systemctl is-enabled docker
    ```
    If the output is not `enabled`, start and enable Docker with:
    ```bash
    sudo systemctl start docker
    sudo systemctl enable docker
    ```

4. **Configure X11 forwarding for GUI applications:**
   ```bash
   echo "xhost +" >> ~/.bashrc
   echo "xhost +local:docker" >> ~/.bashrc
   ```

### Running Wilson Simulation
Run these commands with `sudo` privileges:

1. **Pull the ROS 2 base image:**
   ```bash
   sudo docker image pull osrf/ros:humble-desktop-full
   ```

2. **Build Wilson's Docker image** (must be run from the wilson directory):
   ```bash
   sudo docker build -t wilson_image .
   ```

3. **Run the Wilson container:** (must be run from the wilson directory):
   ```bash
   sudo docker run -it --user ros --network=host --ipc=host \
     -v $PWD:/wilson \
     -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
     --env=DISPLAY=:0 \
     --env=QT_X11_NO_MITSHM=1 \
     -v /dev:/dev \
     --privileged \
     --name wilson \
     wilson_image
   ```

### Gemini AI Setup (Optional)

**For AI voice/text commands and interactions:** Create a `.env` file in the wilson directory with your Google API key:

```bash
# In the wilson directory, create .env file
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

Replace `your_api_key_here` with your actual Google Gemini API key from [Google AI Studio](https://aistudio.google.com). Without this file, Wilson will work but won't have AI-powered voice commands and object recognition capabilities.

### Starting the Simulation

Once inside the container, start Wilson's simulation with:

```bash
colcon build --symlink-install && source install/setup.bash && ros2 launch wilson wilson_sim.launch.py
```

### Controlling Wilson 🎮

After the simulation launches, you have multiple ways to control Wilson:

- **RViz Panels**: Use the Nav2 and MoveIt panels in RViz for navigation and manipulation
- **Teleop Keyboard**: Control Wilson directly with keyboard inputs
- **AI Voice/Text Commands**: Talk or type to Gemini for natural language control

#### AI Commands Examples:
- "Go to the kitchen"
- "Go to the living room" 
- "Go to the mini fridge"
- "What do you see?"
- "Find the 3D position of [object]" - This will display a marker in RViz showing the detected object's location

Wilson combines autonomous navigation, manipulation, and AI-powered interaction to create an intelligent robotic assistant!

</details>

