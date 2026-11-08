#!/usr/bin/env python3
"""
Track wrist position using Intel RealSense D435 RGB-D camera
Uses MediaPipe for hand detection and RealSense depth for 3D coordinates
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
import argparse
from datetime import datetime

from wrist_kalman import WristKalman

SWAP_HAND = {'Left': 'Right', 'Right': 'Left'}


class RealSenseWristTracker(Node):
    def __init__(self, output_file=None, camera_ns='/camera/camera',
                 target_hand='Right', max_speed=3.0,
                 kf_enable=True, kf_max_predict_s=0.5, kf_accel_noise=2.0,
                 kf_meas_noise=(0.005, 0.005, 0.008)):
        super().__init__('realsense_wrist_tracker')

        self.bridge = CvBridge()
        self.output_file = output_file or f"data/trajectories/realsense/trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"

        # realsense2_camera 4.58 (Humble) publishes under <camera_namespace>/<camera_name>,
        # both of which default to 'camera' -> /camera/camera/color/image_raw
        self.camera_ns = camera_ns.rstrip('/')

        # MediaPipe Hands
        # max_num_hands=2 so MediaPipe sees BOTH hands and we pick one by
        # handedness label. With max_num_hands=1 it returns whichever hand
        # scored highest that frame, which flips between hands mid-recording:
        # trial_20260929_160827 had 7 single-frame jumps of 190-480 mm from this.
        self.target_hand = target_hand
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # Safety net: a wrist cannot move faster than this. Anything above is a
        # mis-association (wrong hand, or a depth hole giving a bogus Z) and is
        # dropped instead of written into the trajectory. 0 disables.
        self.max_speed = max_speed
        self.last_point = None
        self.last_ts = None
        self.rejected = 0
        self.other_hand_detections = 0

        # Strategy 2 (proposal sec. 3.4): constant-velocity Kalman filter.
        # Smooths measured positions and bridges occlusions up to
        # kf_max_predict_s with predicted positions. Predicted samples are
        # flagged in *_measured.npy so compare_trajectories.py excludes them
        # from RMSE while the saved path stays continuous for replay.
        self.kf = (WristKalman(accel_noise=kf_accel_noise, meas_noise=kf_meas_noise)
                   if kf_enable else None)
        self.kf_max_predict_s = kf_max_predict_s
        self.last_frame_ts = None
        self.frame_times = []         # every processed frame: a gap here is a VM/camera
                                      # stall, a gap only in raw_timestamps is a lost hand
        self.raw_trajectory = []      # measurements only, no filtering
        self.raw_timestamps = []
        self.is_measured = []         # parallel to self.trajectory
        self.pending = []             # (point, ts) predictions awaiting a re-detection
        self.predicted_count = 0      # predictions committed to the trajectory
        self.dropped_predictions = 0  # predictions discarded: gap ran past the limit
        self.gap_count = 0            # occlusions too long to bridge

        # Camera intrinsics (will be updated from camera_info)
        self.fx = 615.0  # Default values for D435
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0

        # Data storage
        self.trajectory = []
        self.timestamps = []

        # Latest images
        self.color_image = None
        self.depth_image = None
        self.depth_scale = 0.001  # Default 1mm per unit

        # Subscribers
        self.color_sub = self.create_subscription(
            Image,
            f'{self.camera_ns}/color/image_raw',
            self.color_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            f'{self.camera_ns}/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            f'{self.camera_ns}/color/camera_info',
            self.camera_info_callback,
            10
        )

        # Publisher for visualization
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/realsense/wrist_pose',
            10
        )

        self.get_logger().info("RealSense wrist tracker initialized")
        self.get_logger().info(f"Subscribing under: {self.camera_ns}")
        self.get_logger().info(f"Will save trajectory to: {self.output_file}")

        # Make it obvious when the camera node is not up / topics are named
        # differently, instead of sitting silently with zero samples.
        self.frames_seen = 0
        self.create_timer(5.0, self.check_alive)

    def check_alive(self):
        if self.frames_seen == 0:
            self.get_logger().warn(
                f"No frames received yet on {self.camera_ns}/color/image_raw - "
                "is realsense2_camera running? (ros2 topic list | grep image_raw)")

    def camera_info_callback(self, msg):
        """Update camera intrinsics from CameraInfo"""
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def color_callback(self, msg):
        """Process color image for hand detection"""
        self.frames_seen += 1
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg):
        """Store depth image"""
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')

        # Process if we have both color and depth
        if self.color_image is not None:
            self.process_frame()

    def pixel_to_3d(self, u, v, depth):
        """Convert pixel coordinates and depth to 3D point"""
        if depth == 0:
            return None

        z = depth * self.depth_scale  # Convert to meters
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        return np.array([x, y, z])

    def _detect_wrist(self):
        """Run MediaPipe on the current frame and return the target hand's
        wrist as (point_3d, u, v, hand_landmarks), or None. Labels every hand
        on screen so it is obvious which physical hand carries which label."""
        rgb_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        if not results.multi_hand_landmarks:
            return None

        h, w = self.color_image.shape[:2]
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):
            # MediaPipe assumes a mirrored (selfie) image; RealSense colour is
            # not mirrored, so swap the label as the MediaPipe docs instruct.
            # 'Right' is then the operator's physical right hand.
            label = SWAP_HAND[handedness.classification[0].label]
            score = handedness.classification[0].score

            # Wrist is landmark 0; normalised coords -> clamped pixel coords
            wrist = hand_landmarks.landmark[0]
            u = max(0, min(int(wrist.x * w), w - 1))
            v = max(0, min(int(wrist.y * h), h - 1))

            is_target = (label == self.target_hand)
            colour = (0, 255, 0) if is_target else (160, 160, 160)
            cv2.putText(self.color_image, f"{label} {score:.2f}", (u + 12, v - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
            if not is_target:
                self.other_hand_detections += 1
                continue

            # Depth at the wrist: median of a 5x5 window, holes (0) ignored
            depth_region = self.depth_image[max(0, v - 2):min(h, v + 3),
                                            max(0, u - 2):min(w, u + 3)]
            depth_values = depth_region[depth_region > 0]
            if len(depth_values) == 0:
                self.get_logger().warn("No valid depth at wrist position")
                return None

            point_3d = self.pixel_to_3d(u, v, np.median(depth_values))
            if point_3d is None:
                return None
            return point_3d, u, v, hand_landmarks
        return None

    def _project(self, point_3d):
        """Camera-frame 3D point -> pixel, for drawing predicted positions."""
        x, y, z = point_3d
        if z <= 0:
            return None
        return int(self.fx * x / z + self.cx), int(self.fy * y / z + self.cy)

    def process_frame(self):
        """One tracking step: detect -> speed gate -> Kalman -> store -> draw."""
        if self.color_image is None or self.depth_image is None:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        dt = (now - self.last_frame_ts) if self.last_frame_ts is not None else 0.0
        self.last_frame_ts = now
        self.frame_times.append(now)

        det = self._detect_wrist()
        measurement = None
        if det is not None:
            point_3d = det[0]
            # Reject physically impossible jumps (see max_speed). Only applied
            # when the previous sample is recent; after the hand has been out
            # of view any position is legitimate.
            if (self.max_speed > 0 and self.last_point is not None
                    and 0 < now - self.last_ts < 0.5):
                dist = np.linalg.norm(point_3d - self.last_point)
                speed = dist / (now - self.last_ts)
                if speed > self.max_speed:
                    self.rejected += 1
                    self.get_logger().warn(
                        f"Rejected sample: {dist*1000:.0f} mm in one step = {speed:.1f} m/s")
                    det = None
            if det is not None:
                measurement = point_3d
                self.last_point = point_3d
                self.last_ts = now
                self.raw_trajectory.append(point_3d)
                self.raw_timestamps.append(now)

        # --- Kalman filter (proposal sec. 3.4, Strategy 2) -------------------
        # Predictions are BUFFERED, not stored straight away. The proposal
        # keeps predicted samples only for occlusions that turn out to be
        # short (<= kf_max_predict_s), and that is only known once the hand
        # is seen again. If it comes back in time the buffer is committed as
        # the bridge; if the gap runs past the limit the buffer is discarded,
        # so a dead-end extrapolation never ends up in the saved path followed
        # by a jump to wherever the hand reappeared. Live /realsense/wrist_pose
        # is still published every frame for real-time consumers.
        stored, is_measured = None, False
        if self.kf is None:
            if measurement is not None:
                stored, is_measured = measurement, True
        elif measurement is not None:
            if self.pending:                       # gap closed in time: commit bridge
                for p, t in self.pending:
                    self.trajectory.append(p)
                    self.timestamps.append(t)
                    self.is_measured.append(False)
                self.predicted_count += len(self.pending)
                self.pending = []
            stored, is_measured = self.kf.step(dt, measurement), True
        elif self.kf.initialised:
            if now - self.last_ts <= self.kf_max_predict_s:
                stored, is_measured = self.kf.predict(dt), False
                self.pending.append((stored, now))
            else:
                # Occlusion longer than the bridging limit: drop the dangling
                # predictions, stop, wait for a fresh detection. Once per gap.
                self.gap_count += 1
                self.dropped_predictions += len(self.pending)
                self.pending = []
                self.kf.reset()

        if stored is not None:
            if is_measured:
                self.trajectory.append(stored)
                self.timestamps.append(now)
                self.is_measured.append(True)

            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'camera_color_optical_frame'
            pose_msg.pose.position.x = float(stored[0])
            pose_msg.pose.position.y = float(stored[1])
            pose_msg.pose.position.z = float(stored[2])
            self.pose_pub.publish(pose_msg)

            if len(self.trajectory) % 30 == 0:  # roughly once a second
                tag = '' if is_measured else ' [predicted]'
                self.get_logger().info(
                    f"Wrist: ({stored[0]:.3f}, {stored[1]:.3f}, {stored[2]:.3f}) m{tag} | "
                    f"Samples: {len(self.trajectory)}")

        # --- Visualisation ---------------------------------------------------
        if det is not None:
            _, u, v, hand_landmarks = det
            self.mp_drawing.draw_landmarks(self.color_image, hand_landmarks,
                                           self.mp_hands.HAND_CONNECTIONS)
            cv2.circle(self.color_image, (u, v), 10, (0, 255, 0), -1)
            cv2.putText(self.color_image,
                        f"Wrist: {stored[0]:.2f}, {stored[1]:.2f}, {stored[2]:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif stored is not None:
            # Predicted: hollow orange circle so a bridged stretch is visible
            px = self._project(stored)
            if px is not None:
                cv2.circle(self.color_image, px, 10, (0, 165, 255), 2)
            cv2.putText(self.color_image,
                        f"PREDICTED {stored[0]:.2f}, {stored[1]:.2f}, {stored[2]:.2f} "
                        f"({now - self.last_ts:.2f}s without detection)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        cv2.putText(self.color_image,
                    f"Samples: {len(self.trajectory)}   predicted: {self.predicted_count}"
                    + (f"  (+{len(self.pending)} pending)" if self.pending else ""),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow('RealSense Wrist Tracking', self.color_image)
        cv2.waitKey(1)

    def save_trajectory(self):
        """Save trajectory to file"""
        # Predictions still waiting for a re-detection at shutdown belong to a
        # gap that never closed - drop them, same rule as a too-long occlusion.
        if self.pending:
            self.dropped_predictions += len(self.pending)
            self.pending = []
        if len(self.trajectory) == 0:
            self.get_logger().warn("No trajectory data to save!")
            return

        trajectory_array = np.array(self.trajectory)
        timestamps_array = np.array(self.timestamps)

        # Save as numpy file
        np.save(self.output_file, trajectory_array)

        # Also save timestamps
        timestamp_file = self.output_file.replace('.npy', '_timestamps.npy')
        np.save(timestamp_file, timestamps_array)

        # Sidecars for compare_trajectories.py:
        #   *_measured.npy : True = real detection, False = Kalman prediction
        #   *_raw.npy      : unfiltered measurements only (no Kalman smoothing)
        np.save(self.output_file.replace('.npy', '_measured.npy'),
                np.array(self.is_measured, dtype=bool))
        if self.raw_trajectory:
            np.save(self.output_file.replace('.npy', '_raw.npy'), np.array(self.raw_trajectory))
            np.save(self.output_file.replace('.npy', '_raw_timestamps.npy'), np.array(self.raw_timestamps))
        np.save(self.output_file.replace('.npy', '_frames.npy'), np.array(self.frame_times))

        self.get_logger().info(f"✓ Saved {len(trajectory_array)} points to {self.output_file}")
        self.get_logger().info(
            f"  tracked hand: '{self.target_hand}' | other-hand detections skipped: "
            f"{self.other_hand_detections} | samples rejected for speed: {self.rejected}")
        self.get_logger().info(
            f"  measured: {int(np.sum(self.is_measured))} | Kalman-predicted (bridged): {self.predicted_count} | "
            f"predictions dropped (gap > {self.kf_max_predict_s:.2f}s): {self.dropped_predictions} | "
            f"unbridged occlusions: {self.gap_count}")
        self.get_logger().info(f"✓ Timestamps saved to {timestamp_file}")

        # Print statistics
        self.get_logger().info("\nTrajectory Statistics:")
        self.get_logger().info(f"  Duration: {timestamps_array[-1] - timestamps_array[0]:.2f} seconds")
        self.get_logger().info(f"  Sample rate: {len(trajectory_array) / (timestamps_array[-1] - timestamps_array[0]):.1f} Hz")
        self.get_logger().info(f"  X range: {trajectory_array[:, 0].min():.3f} to {trajectory_array[:, 0].max():.3f} m")
        self.get_logger().info(f"  Y range: {trajectory_array[:, 1].min():.3f} to {trajectory_array[:, 1].max():.3f} m")
        self.get_logger().info(f"  Z range: {trajectory_array[:, 2].min():.3f} to {trajectory_array[:, 2].max():.3f} m")


def main(args=None):
    parser = argparse.ArgumentParser(description='Track wrist with RealSense')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--camera-ns', type=str, default='/camera/camera',
                        help='Topic prefix of realsense2_camera (default: /camera/camera)')
    parser.add_argument('--hand', choices=['Left', 'Right'], default='Right',
                        help="Operator's physical hand to track (default Right). "
                             "Check the green label follows that hand.")
    parser.add_argument('--max-speed', type=float, default=3.0,
                        help='Drop samples implying wrist speed above this (m/s). 0 disables.')
    parser.add_argument('--no-kalman', action='store_true',
                        help='Disable the Kalman filter: save raw MediaPipe positions, no bridging')
    parser.add_argument('--kf-max-predict', type=float, default=0.5,
                        help='Bridge occlusions up to this many seconds with predicted positions '
                             '(default 0.5 s = 15 frames at 30 fps, proposal sec. 3.4)')
    parser.add_argument('--kf-accel-noise', type=float, default=2.0,
                        help='Kalman process noise: unmodelled acceleration std, m/s^2 (default 2.0)')
    parser.add_argument('--kf-meas-noise', type=float, nargs=3, default=[0.005, 0.005, 0.008],
                        metavar=('SX', 'SY', 'SZ'),
                        help='Kalman measurement noise std per axis, m (default 5 5 8 mm)')
    cli_args = parser.parse_args()

    rclpy.init(args=args)

    tracker = RealSenseWristTracker(output_file=cli_args.output, camera_ns=cli_args.camera_ns,
                                    target_hand=cli_args.hand, max_speed=cli_args.max_speed,
                                    kf_enable=not cli_args.no_kalman,
                                    kf_max_predict_s=cli_args.kf_max_predict,
                                    kf_accel_noise=cli_args.kf_accel_noise,
                                    kf_meas_noise=tuple(cli_args.kf_meas_noise))

    try:
        print("\n" + "="*60)
        print("RealSense Wrist Tracking")
        print("="*60)
        print("\nInstructions:")
        print("1. Make sure RealSense is launched:")
        print("   ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true")
        print("2. Show your hand to the camera")
        print(f"   Tracking label '{cli_args.hand}' (green). Grey label = ignored hand.")
        print("   If the wrong hand is green, restart with --hand Left / --hand Right")
        print("3. Move your wrist through the trajectory")
        print("4. Press Ctrl+C to stop and save")
        print("\n" + "="*60 + "\n")

        rclpy.spin(tracker)

    except KeyboardInterrupt:
        print("\n\nStopping tracker...")

    finally:
        tracker.save_trajectory()
        cv2.destroyAllWindows()
        tracker.hands.close()
        tracker.destroy_node()
        rclpy.try_shutdown()  # SIGINT may already have shut the context down


if __name__ == '__main__':
    main()
