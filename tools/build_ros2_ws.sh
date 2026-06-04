#!/usr/bin/env bash
set -euo pipefail

ROS_DISTRO_NAME="${ROS_DISTRO_NAME:-humble}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set +u
source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
set -u
cd "${PROJECT_DIR}/ros2_ws"
colcon build --symlink-install

echo "Workspace built. Run: source ${PROJECT_DIR}/ros2_ws/install/setup.bash"
