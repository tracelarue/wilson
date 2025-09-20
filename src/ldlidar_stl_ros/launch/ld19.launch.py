#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
import glob
import os

#def find_lidar_port():
#    """Automatically detect LIDAR USB port"""
#    import subprocess
#    
#    # Check for common USB serial ports
#    usb_ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
#    
#    if usb_ports:
#        usb_ports.sort()
#        
#        # Try to identify LIDAR by checking device info
#        for port in usb_ports:
#            try:
#                # Get USB device info using udevadm
#                result = subprocess.run(['udevadm', 'info', '-q', 'property', '-n', port], 
#                                      capture_output=True, text=True, timeout=2)
#                device_info = result.stdout.lower()
#                
#                # Look for LIDAR-related identifiers
#                lidar_indicators = ['lidar', 'ld19', 'ldrobot', 'cp210', 'ch341', 'ftdi']
#                
#                if any(indicator in device_info for indicator in lidar_indicators):
#                    print(f"Found LIDAR device at {port} based on device info")
#                    return port
#                    
#            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
#                # If udevadm fails, continue to next port
#                continue
#        
#        # If no specific LIDAR found, try to exclude known non-LIDAR devices
#        for port in usb_ports:
#            try:
#                result = subprocess.run(['udevadm', 'info', '-q', 'property', '-n', port], 
#                                      capture_output=True, text=True, timeout=2)
#                device_info = result.stdout.lower()
#                
#                # Skip known Arduino/camera devices
#                skip_indicators = ['arduino', 'camera', 'webcam', 'video']
#                
#                if not any(indicator in device_info for indicator in skip_indicators):
#                    print(f"Using {port} as LIDAR port (excluded non-LIDAR devices)")
#                    return port
#                    
#            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
#                continue
#        
#        # Last resort: return the first port
#        print(f"Using first available port {usb_ports[0]} as fallback")
#        return usb_ports[0]
#    else:
#        # Fallback to default
#        print("No USB ports found, using default /dev/ttyUSB0")
#        return '/dev/ttyUSB0'

'''
Parameter Description:
---
- Set laser scan directon: 
  1. Set counterclockwise, example: {'laser_scan_dir': True}
  2. Set clockwise,        example: {'laser_scan_dir': False}
- Angle crop setting, Mask data within the set angle range:
  1. Enable angle crop fuction:
    1.1. enable angle crop,  example: {'enable_angle_crop_func': True}
    1.2. disable angle crop, example: {'enable_angle_crop_func': False}
  2. Angle cropping interval setting:
  - The distance and intensity data within the set angle range will be set to 0.
  - angle >= 'angle_crop_min' and angle <= 'angle_crop_max' which is [angle_crop_min, angle_crop_max], unit is degress.
    example:
      {'angle_crop_min': 135.0}
      {'angle_crop_max': 225.0}
      which is [135.0, 225.0], angle unit is degress.
'''

def generate_launch_description():
  # Auto-detect LIDAR port
  #lidar_port = find_lidar_port()
  lidar_port = '/dev/ttyUSB1'
  print(f"LIDAR port: {lidar_port}")
  
  # LDROBOT LiDAR publisher node
  ldlidar_node = Node(
      package='ldlidar_stl_ros2',
      executable='ldlidar_stl_ros2_node',
      name='LD19',
      output='screen',
      parameters=[
        {'product_name': 'LDLiDAR_LD19'},
        {'topic_name': 'scan'},
        {'frame_id': 'base_laser'},
        {'port_name': lidar_port},
        {'port_baudrate': 230400},
        {'laser_scan_dir': True},
        {'enable_angle_crop_func': False},
        {'angle_crop_min': 135.0},
        {'angle_crop_max': 225.0}
      ]
  )

  # base_link to base_laser tf node
  base_link_to_laser_tf_node = Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    name='base_link_to_base_laser_ld19',
    arguments=['0','0','0.18','0','0','0','base_link','base_laser']
  )


  # Define LaunchDescription variable
  ld = LaunchDescription()

  ld.add_action(ldlidar_node)
  ld.add_action(base_link_to_laser_tf_node)

  return ld