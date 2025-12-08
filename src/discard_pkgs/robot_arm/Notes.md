Notes for building robot_arm from turtlebot3_manipulator

Description
- URDF
    - Base urdf (no plugin)
    - Arm urdf  (no plugin)
    - Combined urdf (with gazebo plugin)
- gazebo (these define gazebo references)
    - materials.xacro
    - arm.gazebo.xacro (includes transmissions)
    - base.gazebo.xacro
- ros2_control
    - ros2_control.xacro (for entire robot)
    
My Edits Tracking
- created the ros2_control file
- Made main urdf the same with the ros2 control system and gazebo plugin