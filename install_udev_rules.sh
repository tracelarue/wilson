#!/bin/bash
# Script to install Wilson robot udev rules

set -e

RULES_FILE="99-wilson-robot.rules"
DEST="/etc/udev/rules.d/$RULES_FILE"

echo "Wilson Robot - Persistent Device Name Setup"
echo "============================================="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Installing udev rules..."
    cp "$RULES_FILE" "$DEST"
    echo "Rules installed to $DEST"

    echo "Reloading udev rules..."
    udevadm control --reload-rules

    echo "Triggering udev to apply rules..."
    udevadm trigger

    echo ""
    echo "Waiting for devices to settle..."
    sleep 2

    echo ""
    echo "Checking persistent device names:"
    echo "---------------------------------"

    if [ -L /dev/wilson/lidar ]; then
        echo "[OK] LiDAR: /dev/wilson/lidar -> $(readlink /dev/wilson/lidar)"
    else
        echo "[MISSING] LiDAR: /dev/wilson/lidar not found"
    fi

    if [ -L /dev/wilson/diffdrive ]; then
        echo "[OK] Diff Drive: /dev/wilson/diffdrive -> $(readlink /dev/wilson/diffdrive)"
    else
        echo "[MISSING] Diff Drive: /dev/wilson/diffdrive not found"
    fi

    if [ -L /dev/wilson/arm ]; then
        echo "[OK] Arm Controller: /dev/wilson/arm -> $(readlink /dev/wilson/arm)"
    else
        echo "[MISSING] Arm Controller: /dev/wilson/arm not found"
    fi

    if [ -L /dev/wilson/rgb_camera ]; then
        echo "[OK] RGB Camera: /dev/wilson/rgb_camera -> $(readlink /dev/wilson/rgb_camera)"
    else
        echo "[MISSING] RGB Camera: /dev/wilson/rgb_camera not found"
    fi

    echo ""
    echo "Setup complete!"
    echo ""
    echo "If any devices are missing:"
    echo "1. Check that the device is plugged in"
    echo "2. Run 'lsusb' to verify USB devices are detected"
    echo "3. Check 'dmesg | tail -20' for USB connection messages"
    echo "4. Verify device attributes with 'udevadm info -a -n /dev/ttyUSBX'"

else
    echo "This script must be run as root. Usage:"
    echo "  sudo ./install_udev_rules.sh"
    exit 1
fi
