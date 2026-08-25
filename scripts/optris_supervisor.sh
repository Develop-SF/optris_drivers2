#!/bin/bash
# Optris Thermal Camera Supervisor Script
# This script manages the Optris driver with USB reset and process monitoring

set -e

# Parse arguments
XML_CONFIG_FILE="$1"
NAMESPACE_ARG="$2"
TEMPLATE_FILE="$3"
# Optional extra ros args for the imager node, e.g.
#   "-p emissivity:=0.9 -p temperature_range_min:=150 -p temperature_range_max:=900"
IMAGER_ARGS="${4:-}"

# Cleanup function for graceful shutdown.
# Preserves the exit code: a failure exit must NOT be reported as clean,
# or ros2 launch logs "finished cleanly" and hides fatal driver errors.
CLEANED=""
cleanup() {
    rc="$1"
    [ -n "$CLEANED" ] && exit "$rc"
    CLEANED=1
    echo "Cleaning up Optris processes..."
    pkill -TERM -f optris_imager_node || true
    pkill -TERM -f optris_colorconvert_node || true
    sleep 2
    pkill -KILL -f optris_imager_node || true
    pkill -KILL -f optris_colorconvert_node || true
    exit "$rc"
}

# Trap signals for cleanup (Ctrl-C / launch shutdown exit clean)
trap "cleanup 0" SIGINT SIGTERM
trap 'cleanup $?' EXIT

echo "=== Optris Supervisor Starting ==="

# Step 1: Aggressive cleanup of any existing processes
echo "Step 1: Cleaning up any existing Optris processes..."
pkill -KILL -f optris_imager_node || true
pkill -KILL -f optris_colorconvert_node || true
pkill -KILL -f ir_download_calibration || true
pkill -KILL -f ir_generate_configuration || true
pkill -KILL -f ir_find_serial || true
sleep 1

# USB reset helper: unbind/rebind the Optris device (requires root/sudo).
# Called at startup and again before every imager restart, because a wedged
# USB state makes the imager crash-loop or hang frameless until reset.
usb_reset() {
    # Find the USB device path more reliably
    local usb_path=""
    for dev in /sys/bus/usb/devices/*; do
        if [ -f "$dev/idVendor" ] && [ -f "$dev/idProduct" ]; then
            vendor=$(cat "$dev/idVendor" 2>/dev/null)
            product=$(cat "$dev/idProduct" 2>/dev/null)
            if [ "$vendor" = "0403" ] && [ "$product" = "de37" ]; then
                usb_path=$(basename "$dev")
                break
            fi
        fi
    done

    if [ -n "$usb_path" ]; then
        echo "Found Optris device at: $usb_path"

        # Try to unbind/bind the device using sudo
        if sudo bash -c "echo '$usb_path' > /sys/bus/usb/drivers/usb/unbind" 2>/dev/null; then
            echo "Unbound USB device"
            sleep 2
            if sudo bash -c "echo '$usb_path' > /sys/bus/usb/drivers/usb/bind" 2>/dev/null; then
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
}

# Step 2: USB Device Reset (requires root/sudo)
echo "Step 2: Checking USB device binding..."
usb_reset

# Step 3: Generate configuration
echo "Step 3: Generating/Updating configuration file..."
rm -f "$XML_CONFIG_FILE"

if ! sudo ir_find_serial; then
    echo "ERROR: ir_find_serial failed"
    exit 1
fi

# Need to capture the actual serial number detected by finding 1st value from ir_find_serial output?
# ir_find_serial usually just prints valid serials. We'll grab the first one.
DETECTED_SERIAL=$(sudo ir_find_serial | grep -oE '[0-9]+' | head -n1)
echo "Detected Serial: $DETECTED_SERIAL"

if [ -n "$TEMPLATE_FILE" ] && [ -f "$TEMPLATE_FILE" ]; then
    echo "Using template: $TEMPLATE_FILE"
    cp "$TEMPLATE_FILE" "$XML_CONFIG_FILE"
    
    # Update the Serial Number in the template to match the connected camera
    # This prevents using the hardcoded serial from the template
    if [ -n "$DETECTED_SERIAL" ]; then
        sed -i "s|<serial>.*</serial>|<serial>$DETECTED_SERIAL</serial>|g" "$XML_CONFIG_FILE"
    else
        echo "WARNING: Could not detect serial number to update template!"
    fi
else
    # Fallback to auto-generation
    echo "No valid template provided, auto-generating configuration..."
    if ! sudo ir_generate_configuration > "$XML_CONFIG_FILE"; then
        echo "ERROR: ir_generate_configuration failed"
        exit 1
    fi
fi

# Fix framerate bug (ensure it's not inf)
sed -i 's/<framerate>inf<\/framerate>/<framerate>32.0<\/framerate>/g' "$XML_CONFIG_FILE"
echo "Configuration ready at: $XML_CONFIG_FILE"

STALL_COUNT=0

restart_imager() {
    # Make sure no half-dead imager survives, reset USB, then start fresh.
    # A plain node restart is NOT enough: a wedged USB device makes the new
    # imager crash-loop or hang frameless (observed 2026-07-07).
    kill -KILL $IMAGER_PID 2>/dev/null || true
    pkill -KILL -f optris_imager_node || true
    sleep 1
    usb_reset
    ros2 run optris_drivers2 optris_imager_node $NAMESPACE_ARG --ros-args -p xml_config_file:="$XML_CONFIG_FILE" $IMAGER_ARGS &
    IMAGER_PID=$!
    STALL_COUNT=0
    sleep 2
}

# Step 4: Launch imager node in background
echo "Step 4: Starting optris_imager_node..."
ros2 run optris_drivers2 optris_imager_node $NAMESPACE_ARG --ros-args -p xml_config_file:="$XML_CONFIG_FILE" $IMAGER_ARGS &
IMAGER_PID=$!
echo "Imager node started with PID: $IMAGER_PID"

# Wait a bit for imager to initialize
sleep 5

# Retry with USB reset instead of giving up: right after a reset the Xi can
# take longer than our post-reset sleep to re-enumerate as UVC, so the first
# imager start sometimes races it and dies with "UVC device not found"
# (observed 2026-07-29).
STARTUP_TRIES=1
while ! kill -0 $IMAGER_PID 2>/dev/null; do
    if [ "$STARTUP_TRIES" -ge 3 ]; then
        echo "ERROR: Imager node failed to start after $STARTUP_TRIES attempts"
        exit 1
    fi
    STARTUP_TRIES=$((STARTUP_TRIES + 1))
    echo "WARNING: Imager died at startup, USB reset + retry ($STARTUP_TRIES/3)..."
    restart_imager
    sleep 5
done

# Step 5: Launch colorconvert node in background
echo "Step 5: Starting optris_colorconvert_node..."
ros2 run optris_drivers2 optris_colorconvert_node $NAMESPACE_ARG &
COLORCONVERT_PID=$!
echo "Colorconvert node started with PID: $COLORCONVERT_PID"

# Monitor processes
echo "=== Optris Supervisor Running ==="
echo "Monitoring processes (Ctrl+C to stop)..."

# Health-check topic: derive the plain namespace from NAMESPACE_ARG
# (e.g. "--ros-args -r __ns:=/emily01/head" -> "/emily01/head")
NS=$(echo "$NAMESPACE_ARG" | grep -oE '__ns:=[^ ]+' | cut -d= -f2 || true)
THERMAL_TOPIC="${NS}/thermal_image"
STALL_COUNT=0

while true; do
    # Check if imager is still running
    if ! kill -0 $IMAGER_PID 2>/dev/null; then
        echo "WARNING: Imager node died, USB reset + restart..."
        restart_imager
    else
        # Watchdog: process can be alive but frameless (stuck USB handle).
        # Two consecutive misses (>25 s without a frame) trigger recovery.
        if timeout 10 ros2 topic echo --once --no-arr "$THERMAL_TOPIC" > /dev/null 2>&1; then
            STALL_COUNT=0
        else
            STALL_COUNT=$((STALL_COUNT + 1))
            echo "WARNING: No frame on $THERMAL_TOPIC (miss $STALL_COUNT/2)"
            if [ "$STALL_COUNT" -ge 2 ]; then
                echo "WARNING: Imager stalled, USB reset + restart..."
                restart_imager
            fi
        fi
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
