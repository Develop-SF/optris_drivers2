#!/bin/bash
# Optris Thermal Camera Supervisor Script
# This script manages the Optris driver with USB reset and process monitoring

set -e

# Parse arguments
XML_CONFIG_FILE="$1"
NAMESPACE_ARG="$2"

# Cleanup function for graceful shutdown
cleanup() {
    echo "Cleaning up Optris processes..."
    pkill -TERM -f optris_imager_node || true
    pkill -TERM -f optris_colorconvert_node || true
    sleep 2
    pkill -KILL -f optris_imager_node || true
    pkill -KILL -f optris_colorconvert_node || true
    exit 0
}

# Trap signals for cleanup
trap cleanup SIGINT SIGTERM EXIT

echo "=== Optris Supervisor Starting ==="

# Step 1: Aggressive cleanup of any existing processes
echo "Step 1: Cleaning up any existing Optris processes..."
pkill -KILL -f optris_imager_node || true
pkill -KILL -f optris_colorconvert_node || true
pkill -KILL -f ir_download_calibration || true
pkill -KILL -f ir_generate_configuration || true
pkill -KILL -f ir_find_serial || true
sleep 1

# Step 2: USB Device Reset (requires root/sudo)
echo "Step 2: Checking USB device binding..."

# Find the USB device path more reliably
USB_PATH=""
for dev in /sys/bus/usb/devices/*; do
    if [ -f "$dev/idVendor" ] && [ -f "$dev/idProduct" ]; then
        vendor=$(cat "$dev/idVendor" 2>/dev/null)
        product=$(cat "$dev/idProduct" 2>/dev/null)
        if [ "$vendor" = "0403" ] && [ "$product" = "de37" ]; then
            USB_PATH=$(basename "$dev")
            break
        fi
    fi
done

if [ -n "$USB_PATH" ]; then
    echo "Found Optris device at: $USB_PATH"
    
    # Try to unbind/bind the device using sudo
    if sudo bash -c "echo '$USB_PATH' > /sys/bus/usb/drivers/usb/unbind" 2>/dev/null; then
        echo "Unbound USB device"
        sleep 2
        if sudo bash -c "echo '$USB_PATH' > /sys/bus/usb/drivers/usb/bind" 2>/dev/null; then
            echo "Rebound USB device"
            sleep 3
            echo "USB reset complete"
        else
            echo "Failed to rebind, but continuing..."
            sleep 2
        fi
    else
        echo "No permission for USB reset (sudo failed or not available)"
        echo "Device may be in a stuck state - manual reset may be required"
    fi
else
    echo "WARNING: Optris USB device (0403:de37) not found!"
    echo "Continuing anyway, but driver may fail to initialize..."
fi

# Step 3: Generate configuration
echo "Step 3: Generating configuration file..."
rm -f "$XML_CONFIG_FILE"

if ! sudo ir_find_serial; then
    echo "ERROR: ir_find_serial failed"
    exit 1
fi

if ! sudo ir_generate_configuration > "$XML_CONFIG_FILE"; then
    echo "ERROR: ir_generate_configuration failed"
    exit 1
fi

# Fix framerate bug
sed -i 's/<framerate>inf<\/framerate>/<framerate>32.0<\/framerate>/g' "$XML_CONFIG_FILE"
echo "Configuration generated at: $XML_CONFIG_FILE"

# Step 4: Launch imager node in background
echo "Step 4: Starting optris_imager_node..."
ros2 run optris_drivers2 optris_imager_node $NAMESPACE_ARG --ros-args -p xml_config_file:="$XML_CONFIG_FILE" &
IMAGER_PID=$!
echo "Imager node started with PID: $IMAGER_PID"

# Wait a bit for imager to initialize
sleep 5

# Check if imager is still running
if ! kill -0 $IMAGER_PID 2>/dev/null; then
    echo "ERROR: Imager node died immediately"
    exit 1
fi

# Step 5: Launch colorconvert node in background
echo "Step 5: Starting optris_colorconvert_node..."
ros2 run optris_drivers2 optris_colorconvert_node $NAMESPACE_ARG &
COLORCONVERT_PID=$!
echo "Colorconvert node started with PID: $COLORCONVERT_PID"

# Monitor processes
echo "=== Optris Supervisor Running ==="
echo "Monitoring processes (Ctrl+C to stop)..."

while true; do
    # Check if imager is still running
    if ! kill -0 $IMAGER_PID 2>/dev/null; then
        echo "WARNING: Imager node died, restarting..."
        ros2 run optris_drivers2 optris_imager_node $NAMESPACE_ARG --ros-args -p xml_config_file:="$XML_CONFIG_FILE" &
        IMAGER_PID=$!
        sleep 2
    fi
    
    # Check if colorconvert is still running
    if ! kill -0 $COLORCONVERT_PID 2>/dev/null; then
        echo "WARNING: Colorconvert node died, restarting..."
        ros2 run optris_drivers2 optris_colorconvert_node $NAMESPACE_ARG &
        COLORCONVERT_PID=$!
        sleep 2
    fi
    
    sleep 5
done
