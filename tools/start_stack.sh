#!/usr/bin/env bash
set -eo pipefail

: "${PX4_AUTOPILOT_DIR:?Set PX4_AUTOPILOT_DIR to your PX4-Autopilot checkout.}"
ROS_DISTRO_NAME="${ROS_DISTRO_NAME:-humble}"
GUI="${GUI:-false}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PX4_BUILD_DIR="${PX4_AUTOPILOT_DIR}/build/px4_sitl_default"
PX4_GZ_ENV="${PX4_BUILD_DIR}/rootfs/gz_env.sh"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
GCS_HEARTBEAT="${GCS_HEARTBEAT:-true}"

source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
source "${PROJECT_DIR}/ros2_ws/install/setup.bash"

if [[ ! -f "${PX4_GZ_ENV}" ]]; then
  echo "PX4 Gazebo environment not found: ${PX4_GZ_ENV}" >&2
  echo "Build PX4 SITL first: cd ${PX4_AUTOPILOT_DIR} && make px4_sitl gz_x500" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

# PX4 does not source this file when PX4_GZ_STANDALONE=1. Gazebo still needs
# these paths so the standalone world can spawn PX4's gz_x500 model.
source "${PX4_GZ_ENV}"

if [[ ! -f "${PX4_GZ_MODELS}/x500/model.sdf" ]]; then
  echo "PX4 x500 model not found: ${PX4_GZ_MODELS}/x500/model.sdf" >&2
  exit 1
fi

export GZ_PARTITION="${GZ_PARTITION:-uav_px4_rl_$$}"
export GZ_IP="${GZ_IP:-127.0.0.1}"

pids=()
terminate_tree() {
  local pid="$1"
  local child
  for child in $(pgrep -P "${pid}" 2>/dev/null || true); do
    terminate_tree "${child}"
  done
  kill "${pid}" 2>/dev/null || true
}

cleanup() {
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do
    terminate_tree "${pid}"
  done
  if ((${#pids[@]})); then
    wait "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

MicroXRCEAgent udp4 -p 8888 &
pids+=("$!")

if [[ "${GCS_HEARTBEAT}" == "true" ]]; then
  if ! "${PYTHON_BIN}" -c "import pymavlink" >/dev/null 2>&1; then
    echo "pymavlink is required for the built-in GCS heartbeat." >&2
    echo "Install Python dependencies first: pip install -r ${PROJECT_DIR}/requirements.txt" >&2
    exit 1
  fi
  "${PYTHON_BIN}" - <<'PY' &
import time

from pymavlink import mavutil

master = mavutil.mavlink_connection(
    "udpout:127.0.0.1:18570",
    source_system=255,
    source_component=0,
)

while True:
    master.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0,
        0,
        mavutil.mavlink.MAV_STATE_ACTIVE,
    )
    time.sleep(1.0)
PY
  pids+=("$!")
fi

ros2 launch uav_px4_rl sim_bridge.launch.py gui:="${GUI}" &
pids+=("$!")
sleep 4

cd "${PX4_AUTOPILOT_DIR}"
PX4_GZ_STANDALONE=1 \
PX4_SYS_AUTOSTART=4001 \
PX4_SIM_MODEL=gz_x500 \
PX4_GZ_MODEL_POSE="0,0,0,0,0,0" \
"${PX4_BUILD_DIR}/bin/px4" &
pids+=("$!")

echo "PX4/Gazebo/ROS 2 stack running. Leave this terminal open."
echo "Gazebo partition: ${GZ_PARTITION}"
wait
