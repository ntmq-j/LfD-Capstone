#!/bin/bash
# =============================================================================
#  Setup script for the new VMware Fusion VM  (Ubuntu 22.04 / kernel 6.8)
#
#  ROS 2 Humble is ALREADY installed here - this script does not touch it.
#  It installs:
#    - librealsense SDK 2.58.4        (from packages.ros.org, already trusted)
#    - RealSense udev rules           (from the librealsense GitHub repo)
#    - realsense2_camera ROS 2 node
#    - UR robot driver + MoveIt (present on the old VM, missing here)
#    - a project venv with pyrealsense2 + OpenCV >= 4.8 (needed for ArUco)
#
#  Usage:  cd ~/lfd_project && ./scripts/setup_new_vm.sh
#          SKIP_TORCH=1 ./scripts/setup_new_vm.sh    # skip the 2.5 GB download
#          SKIP_UR=1    ./scripts/setup_new_vm.sh    # skip UR driver + MoveIt
# =============================================================================
set -euo pipefail

PYRS_VER="2.58.4.10922"      # matches librealsense 2.58.4 from the ROS repo
PROJ="$HOME/lfd_project"
VENV="$PROJ/venv"

log() { echo; echo "=== $* ==="; }

# Ask for sudo once up front so the long apt/pip steps don't stall on a prompt.
sudo -v

# -----------------------------------------------------------------------------
log "1/7  Drop the broken librealsense apt repo"
# librealsense.intel.com now signs its repo with a new key belonging to
# security@realsenseai.com (RealSense was spun out of Intel), but the old URL
# librealsense.intel.com/Debian/librealsense.pgp still serves the 2018 Intel
# key -> "NO_PUBKEY FB0B24895113F120". We take librealsense from
# packages.ros.org instead, which is already configured and trusted here.
sudo rm -f /etc/apt/sources.list.d/librealsense.list
sudo rm -f /etc/apt/keyrings/librealsense.pgp
sudo apt update

# -----------------------------------------------------------------------------
log "2/7  Install librealsense SDK + realsense2_camera node"
sudo apt install -y \
    ros-humble-librealsense2 \
    ros-humble-realsense2-camera \
    ros-humble-realsense2-camera-msgs \
    ros-humble-realsense2-description

# -----------------------------------------------------------------------------
log "3/7  Install udev rules (the ROS deb does not ship them)"
# Without these, the USB nodes stay root-only and libuvc cannot open the
# camera as a normal user.
RULES_BASE="https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config"
cd /tmp
curl -fsSLO "$RULES_BASE/99-realsense-libusb.rules"
curl -fsSLO "$RULES_BASE/99-realsense-d4xx-mipi-dfu.rules"
sudo install -m 644 99-realsense-libusb.rules        /etc/udev/rules.d/
sudo install -m 644 99-realsense-d4xx-mipi-dfu.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo usermod -aG plugdev,video "$USER"

# -----------------------------------------------------------------------------
log "4/7  Install UR robot driver + MoveIt"
# python_packages.txt from the old VM lists ur-robot-driver 2.14.0,
# ur-moveit-config, ur-msgs, ur-dashboard-msgs and moveit-msgs, but none of
# those are installed here - only the 282 base ros-humble packages came over.
if [ "${SKIP_UR:-0}" != "1" ]; then
    sudo apt install -y \
        ros-humble-ur \
        ros-humble-ur-robot-driver \
        ros-humble-ur-moveit-config \
        ros-humble-moveit
else
    echo "  skipped (SKIP_UR=1)"
fi

# -----------------------------------------------------------------------------
log "5/7  Create the project venv"
# Deliberately NOT --system-site-packages: ROS 2's cv_bridge is built against
# apt's OpenCV 4.5.4, and 4.5.4 has no cv2.aruco.ArucoDetector, which
# scripts/detect_aruco.py needs. Keeping them separate avoids an ABI clash.
sudo apt install -y python3-venv python3-dev build-essential cmake
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --quiet --upgrade pip wheel

# -----------------------------------------------------------------------------
log "6/7  Install Python packages into the venv"
# mediapipe is PINNED to 0.10.14, do not relax this to ">=0.10".
# mediapipe 1.0.1 (current latest) deleted the whole mp.solutions namespace in
# favour of the Tasks API. Four scripts in this project - test_tracking.py,
# 02_test_hand_detection.py, track_wrist_realsense.py, calibrate_camera.py -
# all use mp.solutions.hands / drawing_utils, so 1.0.1 breaks every one of them
# at import time. 0.10.14 still ships mp.solutions and is verified working here.
#
# numpy MUST stay < 2. ROS 2 Humble's cv_bridge (apt, /opt/ros/humble) is a
# compiled boost module built against numpy 1.x; on numpy 2.x it dies with
# "AttributeError: _ARRAY_API not found" the moment imgmsg_to_cv2 is called.
# track_wrist_realsense.py / calibrate_camera.py go through cv_bridge, so a
# venv on numpy 2 looks healthy in isolation and then crashes on the first
# camera frame. mediapipe 0.10.14 itself is fine on 1.26.4 (verified).
# ml-dtypes is pinned alongside because 0.6.x hard-requires numpy>=2.
pip install \
    "pyrealsense2==$PYRS_VER" \
    "opencv-contrib-python>=4.8.0" \
    "mediapipe==0.10.14" \
    "numpy>=1.26,<2" \
    "ml-dtypes<0.6" \
    "scipy>=1.9" \
    "matplotlib>=3.6" \
    "pyyaml>=6.0" \
    "pandas>=2.0" \
    "h5py>=3.8" \
    "scikit-learn>=1.3" \
    "tqdm>=4.65" \
    "colorama>=0.4.6" \
    "seaborn>=0.12"

log "6b   ur_rtde (optional, needs boost - a few minutes)"
sudo apt install -y libboost-all-dev || true
pip install "ur_rtde>=1.5.0" || echo "  ur_rtde failed - not fatal, can skip"

log "6c   PyTorch CPU (optional, ~2.5 GB)"
if [ "${SKIP_TORCH:-0}" != "1" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
    echo "  skipped (SKIP_TORCH=1)"
fi

# -----------------------------------------------------------------------------
log "7/7  Verify"
python - <<'PY'
import importlib
mods = ["pyrealsense2", "cv2", "numpy", "scipy", "mediapipe",
        "sklearn", "pandas", "h5py", "matplotlib", "yaml"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  OK    {m:14s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  FAIL  {m:14s} {e}")
import cv2
ok = hasattr(cv2.aruco, "ArucoDetector")
print(f"  {'OK   ' if ok else 'FAIL '} cv2.aruco.ArucoDetector present: {ok}")

# The project's tracking scripts need the legacy mp.solutions namespace, which
# only exists on mediapipe < 1.0. Import alone is not enough of a check.
import mediapipe as mp
has_sol = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
print(f"  {'OK   ' if has_sol else 'FAIL '} mp.solutions.hands present: {has_sol}")

# Smoke-test real inference: catches EGL / XNNPACK / model-download failures
# that a bare import would sail straight past.
if has_sol:
    import numpy as np
    try:
        h = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)
        h.process(np.zeros((480, 640, 3), np.uint8))
        h.close()
        print("  OK    mediapipe inference ran on a test frame")
    except Exception as e:
        print(f"  FAIL  mediapipe inference: {type(e).__name__}: {e}")
PY

echo
echo "=== Done ==="
cat <<'EOF'

Next steps:

  1. Log out and back in - the plugdev/video group change needs a new session.

  2. Attach the D435i in VMware Fusion:
       Virtual Machine > USB & Bluetooth > Intel RealSense D435i > Connect
     Then check it landed on the USB 3 controller, not USB 2:
       lsusb -t
       source /opt/ros/humble/setup.bash && rs-enumerate-devices | grep -i "usb type"
     You want "3.2". If it reports 2.1, set USB compatibility to USB 3.1 in
     Settings > USB & Bluetooth, then re-attach the camera.

  3. Camera / vision work runs in the venv:
       cd ~/lfd_project && source venv/bin/activate
       python3 scripts/benchmark_camera.py

  4. GUI check (binaries live in /opt/ros/humble/bin, so source ROS first):
       source /opt/ros/humble/setup.bash && realsense-viewer

EOF
