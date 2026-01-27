#!/bin/bash

# ==============================================================================
# ROS 2 Tailscale Network Setup Script
# ==============================================================================
# Usage: source setup_tailscale_ros.sh
# Check if the script is being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced."
    echo "Usage: source setup_tailscale_ros.sh"
    exit 1
fi

echo ">> Configuring ROS 2 for Tailscale VPN Environment..."

# 1. Set Middleware to Cyclone DDS (Installed and recommended for interface binding)
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
echo "   [1] RMW_IMPLEMENTATION set to '$RMW_IMPLEMENTATION'"

# 2. Find Tailscale Interface (usually tailscale0)
TAILSCALE_IFACE=$(ip -o link show | awk -F': ' '{print $2}' | grep "tailscale" | head -n 1)

if [ -z "$TAILSCALE_IFACE" ]; then
    echo "   [!] WARNING: Could not auto-detect a 'tailscale' interface."
    echo "   [!] Defaulting to 'tailscale0'. If this is incorrect, edit this script."
    TAILSCALE_IFACE="tailscale0"
else
    echo "   [2] Auto-detected Tailscale interface: '$TAILSCALE_IFACE'"
fi

# 3. Configure Cyclone DDS to bind ONLY to the Tailscale interface
export CYCLONEDDS_URI="<CycloneDDS><Domain><General><NetworkInterfaceAddress>${TAILSCALE_IFACE}</NetworkInterfaceAddress></General></Domain></CycloneDDS>"
echo "   [3] CYCLONEDDS_URI configured to bind to '$TAILSCALE_IFACE'"

# 4. Disable Localhost Only (Allow external communication)
export ROS_LOCALHOST_ONLY=0
echo "   [4] ROS_LOCALHOST_ONLY set to '0' (External communication allowed)"

# 5. Set Domain ID (Must match on all machines)
export ROS_DOMAIN_ID=0
echo "   [5] ROS_DOMAIN_ID set to '0'"

echo ">> Setup Complete. You can now launch your ROS 2 nodes."
