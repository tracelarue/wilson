# Wilson Teleop GUI

Simple ROS 2 Humble teleoperation GUI for differential drive robots.

## Features

- Forward / reverse
- Rotate left / right
- Forward while rotating
- Reverse while rotating
- Stop button
- Press-and-hold controls (release to stop)
- Keyboard shortcuts: `W A S D Q E Z C`, `Space` to stop

## Launch

GUI only (for running from a separate dev computer on the same ROS network):

```bash
ros2 launch wilson_teleop_gui teleop_gui.launch.py
```

GUI + Wilson robot bringup:

```bash
ros2 launch wilson_teleop_gui with_robot.launch.py
```

## Useful Args

- `cmd_vel_topic:=/cmd_vel`
- `linear_speed:=0.25`
- `angular_speed:=0.25`
- `publish_rate_hz:=20.0` (GUI-only launch)
