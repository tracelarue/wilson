# Persistent Device Names for Wilson Robot

## Problem

Every time the Raspberry Pi restarts, USB device addresses (`/dev/ttyUSB0`, `/dev/ttyUSB1`, `/dev/video8`, etc.) can change order, causing the robot to fail to start because devices are assigned to the wrong hardware.

## Solution

We use **udev rules** to create persistent symlinks in `/dev/wilson/` that always point to the correct device regardless of connection order.

## Device Mapping

| Device | Persistent Name | Description |
|--------|----------------|-------------|
| LiDAR LD19 | `/dev/wilson/lidar` | CP2102 USB-UART Bridge (Silicon Labs) |
| Differential Drive Arduino | `/dev/wilson/diffdrive` | CH340 on USB3 Bus Port 1 |
| Arm Controller Arduino | `/dev/wilson/arm` | CH340 on USB1 Bus via hub at port 1.4 |
| RGB USB Camera | `/dev/wilson/rgb_camera` | HD USB Camera (32e4:9310) |
| CSI Camera (optional) | `/dev/wilson/csi_camera0` | Raspberry Pi CSI camera |

## Installation

### One-Time Setup

Run the installation script as root:

```bash
cd /home/pi5/wilson
sudo ./install_udev_rules.sh
```

The script will:
1. Copy the udev rules to `/etc/udev/rules.d/`
2. Reload udev rules
3. Trigger device detection
4. Verify all persistent device names are created

### Manual Installation

If you prefer to install manually:

```bash
sudo cp 99-wilson-robot.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Verification

Check that all device symlinks exist:

```bash
ls -la /dev/wilson/
```

You should see:
```
lrwxrwxrwx  1 root root  10 Oct 26 22:00 arm -> ../ttyUSB2
lrwxrwxrwx  1 root root    9 Oct 26 22:00 csi_camera0 -> ../video2
lrwxrwxrwx  1 root root   10 Oct 26 22:00 diffdrive -> ../ttyUSB1
lrwxrwxrwx  1 root root   10 Oct 26 22:00 lidar -> ../ttyUSB0
lrwxrwxrwx  1 root root    9 Oct 26 22:00 rgb_camera -> ../video0
```

The actual target devices (`ttyUSBX`, `videoX`) may vary, but the symlinks will always be correct.

## Updated Files

The following files have been updated to use persistent device names:

### Launch Files
- [src/ldlidar_stl_ros/launch/ld19.launch.py](src/ldlidar_stl_ros/launch/ld19.launch.py) - Uses `/dev/wilson/lidar`
- [src/wilson/launch/real/cameras_and_locate.launch.py](src/wilson/launch/real/cameras_and_locate.launch.py) - Uses `/dev/wilson/rgb_camera`
- [src/wilson/launch/real/robot.launch.py](src/wilson/launch/real/robot.launch.py) - Uses `/dev/wilson/rgb_camera`

### Hardware Configuration
- [src/wilson/ros2_control/wilson_system.ros2_control.xacro](src/wilson/ros2_control/wilson_system.ros2_control.xacro) - Uses `/dev/wilson/diffdrive` and `/dev/wilson/arm`

## How It Works

### Udev Rules Matching

The udev rules use the following attributes to uniquely identify each device:

1. **LiDAR**: Matches by USB Vendor/Product ID (10c4:ea60 - Silicon Labs CP2102)
2. **Differential Drive**: Matches by Vendor/Product ID (1a86:7523) AND USB bus path (usb3/3-1)
3. **Arm Controller**: Matches by Vendor/Product ID (1a86:7523) AND USB bus path (usb1/1-1/1-1.4)
4. **RGB Camera**: Matches by USB Vendor/Product ID (32e4:9310)

### Why Device Paths?

The differential drive and arm controller both use the same CH340 USB-serial chip (idVendor=1a86, idProduct=7523), so we differentiate them by their physical USB port location:
- **Diffdrive** is plugged into USB3 bus, port 1 (standalone port)
- **Arm** is plugged into USB1 bus, through a hub at port 1, subport 4

This means as long as you keep the devices plugged into the same physical USB ports, they will always get the correct persistent name.

## Troubleshooting

### Device Not Creating Symlink

If a device doesn't appear in `/dev/wilson/`:

1. **Check device is connected:**
   ```bash
   lsusb
   ls -la /dev/ttyUSB* /dev/video*
   ```

2. **Check device attributes:**
   ```bash
   udevadm info -a -n /dev/ttyUSB0  # Replace with your device
   ```

3. **Test udev rule matching:**
   ```bash
   udevadm test /sys/class/tty/ttyUSB0  # Replace with your device
   ```

4. **Check kernel messages:**
   ```bash
   dmesg | tail -20
   ```

### Device Moved to Different USB Port

If you physically move a device to a different USB port:

1. Find the new device path:
   ```bash
   udevadm info -a -n /dev/ttyUSBX | grep DEVPATH
   ```

2. Update the corresponding line in `99-wilson-robot.rules`

3. Reinstall the rules:
   ```bash
   sudo ./install_udev_rules.sh
   ```

### Verify Rule is Working

To test if a specific device matches the rule:

```bash
# For serial devices
udevadm test /sys/class/tty/ttyUSB0 2>&1 | grep wilson

# For video devices
udevadm test /sys/class/video4linux/video0 2>&1 | grep wilson
```

You should see lines indicating the symlink creation.

## Depth Camera Note

The ArducamDepthCamera is accessed via its Python library (ArducamDepthCamera) and does NOT use a `/dev/video` device, so no udev rule is needed for it.

## Benefits

1. **Reliability**: Robot will start correctly after reboot regardless of USB enumeration order
2. **Clarity**: Device names like `/dev/wilson/arm` are more readable than `/dev/ttyUSB2`
3. **Debugging**: Easy to verify which device is which with `ls -la /dev/wilson/`
4. **Robustness**: Survives USB hot-plugging and reconnection

## Testing After Reboot

To verify the persistent names survive a reboot:

```bash
# Before reboot, note which ttyUSBX devices map to which symlinks
ls -la /dev/wilson/

# Reboot
sudo reboot

# After reboot, check again
ls -la /dev/wilson/
```

The symlinks should still point to the correct hardware, even if the underlying `ttyUSBX` numbers changed.
