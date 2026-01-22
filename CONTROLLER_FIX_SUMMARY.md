# Controller Issue Fix Summary

## Problem

The robot controllers were failing to spawn due to confusion with the `use_fake_hardware` parameter and URDF file selection in the launch system.

## Root Causes

1. **Invalid URDF Selection Logic**: The [rsp.launch.py](src/wilson/launch/base/rsp.launch.py) was using a `PythonExpression` substitution to dynamically choose between `wilson.urdf.xacro` and `wilson_real.urdf.xacro`, but this approach doesn't work correctly in ROS2 launch files because substitutions can't be used directly in file path selection.

2. **Camera Offset Confusion**: Changes were made to add conditional camera offsets in [wilson_core.xacro](src/wilson/urdf/wilson_core.xacro) based on `use_fake_hardware`, but the URDF selection logic was preventing this parameter from being passed correctly.

## Solution

### 1. Simplified URDF Selection

**Changed**: [src/wilson/launch/base/rsp.launch.py:29-36](src/wilson/launch/base/rsp.launch.py#L29-L36)

Instead of trying to dynamically select different URDF files, we now:
- Always use `wilson.urdf.xacro` as the main URDF
- Pass the `use_fake_hardware` parameter through to xacro
- Let the URDF conditionals handle hardware-specific configuration

**Before** (broken):
```python
from launch.substitutions import PythonExpression
xacro_file = PythonExpression([
    '"', os.path.join(pkg_path, 'urdf', 'wilson.urdf.xacro'), '" if "', use_fake_hardware, '" == "true" else "',
    os.path.join(pkg_path, 'urdf', 'wilson_real.urdf.xacro'), '"'
])
```

**After** (fixed):
```python
xacro_file = os.path.join(pkg_path, 'urdf', 'wilson.urdf.xacro')
robot_description_config = Command([
    'xacro ', xacro_file,
    ' use_sim_time:=', use_sim_time,
    ' use_fake_hardware:=', use_fake_hardware,
    ' use_ros2_control:=', use_ros2_control
])
```

### 2. Camera Offset Configuration

The camera offsets in [wilson_core.xacro](src/wilson/urdf/wilson_core.xacro) now correctly use conditional logic:

- **Simulation** (`use_fake_hardware=true`): Uses `-0.25 0 0` offset to work around Open3D clip distance issues
- **Real Robot** (`use_fake_hardware=false`): Uses `0 0 0` offset (no offset needed)

This applies to both:
- `camera_sensor_joint` (RGB camera)
- `depth_camera_sensor_joint` (Depth camera)

## How It Works Now

### Simulation Launch
```bash
ros2 launch wilson wilson_sim.launch.py
```
- Loads `wilson.urdf.xacro` with `use_fake_hardware=true`
- Uses Gazebo simulation hardware plugin
- Applies -0.25m camera offset for Open3D compatibility
- Controllers connect to simulated hardware

### Real Robot Launch
```bash
ros2 launch wilson wilson_real.launch.py
```
- Loads `wilson.urdf.xacro` with `use_fake_hardware=false` (set in [robot.launch.py:31](src/wilson/launch/real/robot.launch.py#L31))
- Uses real Arduino hardware interfaces:
  - `/dev/wilson/diffdrive` for differential drive
  - `/dev/wilson/arm` for arm controller
- No camera offset applied (cameras at true physical position)
- Controllers connect to real hardware via persistent device names

## Parameter Flow

1. **robot.launch.py** passes `use_fake_hardware: 'false'` to rsp.launch.py
2. **rsp.launch.py** passes it to xacro command
3. **wilson.urdf.xacro** receives the parameter and passes it to included files
4. **wilson_core.xacro** uses conditionals (`<xacro:if>` / `<xacro:unless>`) based on the parameter
5. **wilson_system.ros2_control.xacro** selects appropriate hardware plugins

## Configuration Files

### URDF Defaults
- [wilson.urdf.xacro:6](src/wilson/urdf/wilson.urdf.xacro#L6): `use_fake_hardware` default = `true` (for simulation)
- [wilson_core.xacro:8](src/wilson/urdf/wilson_core.xacro#L8): `use_fake_hardware` default = `false` (inherited from parent)

### Launch File Overrides
- [sim.launch.py](src/wilson/launch/sim/sim.launch.py): Uses sim_params.yaml which sets `use_fake_hardware: true`
- [robot.launch.py:31](src/wilson/launch/real/robot.launch.py#L31): Explicitly sets `use_fake_hardware: 'false'`

## Testing

To verify the fix is working:

### Test 1: Check URDF Processing
```bash
# From wilson directory, check that xacro processes correctly for real robot
xacro src/wilson/urdf/wilson.urdf.xacro use_fake_hardware:=false use_sim_time:=false use_ros2_control:=true > /tmp/real_robot.urdf

# Check camera offset is 0 (real robot)
grep -A 3 "camera_sensor_joint" /tmp/real_robot.urdf | grep xyz

# Check hardware interfaces point to persistent devices
grep -E "/dev/wilson/(diffdrive|arm)" /tmp/real_robot.urdf
```

### Test 2: Verify Controller Spawning
```bash
# In docker container, after launching the real robot:
ros2 control list_controllers

# Expected output:
# diff_drive_controller[diff_drive_controller/DiffDriveController] active
# arm_controller[joint_trajectory_controller/JointTrajectoryController] active
# gripper_controller[joint_trajectory_controller/JointTrajectoryController] active
# joint_state_broadcaster[joint_state_broadcaster/JointStateBroadcaster] active
```

### Test 3: Hardware Interface Check
```bash
# Check hardware interfaces loaded correctly
ros2 control list_hardware_interfaces

# Should see interfaces for:
# - left_wheel_joint and right_wheel_joint (velocity command, position/velocity state)
# - joint_1 through joint_4 (position command/state)
# - gripper joints (position command/state)
```

## Benefits

1. **Simplified Architecture**: Single URDF file with conditional logic is easier to maintain than multiple URDF files
2. **Correct Parameter Passing**: `use_fake_hardware` now flows correctly through the entire launch system
3. **Hardware Flexibility**: Same URDF works for both simulation and real robot
4. **Proper Camera Positioning**: Cameras are positioned correctly for both sim and real contexts

## Related Files

### Modified Files
- [src/wilson/launch/base/rsp.launch.py](src/wilson/launch/base/rsp.launch.py) - Simplified URDF selection
- [src/wilson/urdf/wilson_core.xacro](src/wilson/urdf/wilson_core.xacro) - Added conditional camera offsets
- [src/wilson/urdf/wilson.urdf.xacro](src/wilson/urdf/wilson.urdf.xacro) - Updated default for use_fake_hardware

### Unchanged (Working Correctly)
- [src/wilson/launch/real/robot.launch.py](src/wilson/launch/real/robot.launch.py) - Correctly passes use_fake_hardware=false
- [src/wilson/ros2_control/wilson_system.ros2_control.xacro](src/wilson/ros2_control/wilson_system.ros2_control.xacro) - Hardware plugin selection works correctly
- [src/wilson/config/robot_controller_manager.yaml](src/wilson/config/robot_controller_manager.yaml) - Controller configuration

## Troubleshooting

### Issue: Controllers fail to spawn
**Check**:
1. Verify `use_fake_hardware` is being set correctly in launch arguments
2. Check that the URDF generates without errors: `xacro <file> ... > /tmp/test.urdf`
3. Ensure hardware devices exist: `ls -la /dev/wilson/`

### Issue: "No such file /dev/ttyUSBX"
**Fix**: The persistent device names should handle this, but if you see this error it means:
1. The udev rules aren't installed or active
2. The hardware isn't plugged in
3. Run: `sudo ./install_udev_rules.sh` to reinstall

### Issue: Camera transforms are wrong
**Check**:
1. Verify which URDF is being used with `ros2 param get /robot_state_publisher robot_description`
2. Check `use_fake_hardware` parameter: Should be `false` for real robot, `true` for sim
3. View in RViz to visualize camera positions

## Next Steps

1. **Test in Docker Container**: Build and test the real robot launch to verify controllers spawn correctly
2. **Verify Hardware Communication**: Test that commands reach the Arduino controllers via persistent device names
3. **Validate Camera Frames**: Ensure camera TF frames are at correct physical positions
