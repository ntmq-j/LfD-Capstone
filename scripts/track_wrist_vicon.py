#!/usr/bin/env python3
"""
Track wrist position using Vicon motion capture system
Subscribes to Vicon pose topic and records wrist trajectory

Driver: ros2-vicon-receiver, humble branch (~/vicon_ws). It publishes
vicon_receiver/msg/Position on /vicon/<subject>/<segment>, translation in mm.
Only Vicon *objects* are published (a rigid cluster of >= 3 markers defined in
Vicon Tracker), not single loose markers. When the object is occluded the
driver still publishes, with translation 0,0,0: those samples are dropped here.
"""

import rclpy
from rclpy.node import Node
import numpy as np
import argparse
from datetime import datetime

try:
    from vicon_receiver.msg import Position
except ImportError:  # driver workspace not sourced/built
    Position = None


class ViconWristTracker(Node):
    def __init__(self, output_file=None, marker_name='wrist'):
        super().__init__('vicon_wrist_tracker')

        self.output_file = output_file or f"data/trajectories/vicon/trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
        self.marker_name = marker_name

        # Data storage
        self.trajectory = []
        self.timestamps = []
        self.quats = []          # object orientation (x, y, z, w): lets the analysis
                                 # fit a body-fixed offset from the cluster to the wrist
        self.occluded = 0
        self.duplicates = 0
        self.last_frame = None

        # Subscriber to Vicon pose
        self.pose_sub = self.create_subscription(
            Position,
            f'/vicon/{marker_name}/{marker_name}',
            self.pose_callback,
            10
        )

        self.get_logger().info(f"Vicon wrist tracker initialized")
        self.get_logger().info(f"Listening to: /vicon/{marker_name}/{marker_name}")
        self.get_logger().info(f"Will save trajectory to: {self.output_file}")

    def pose_callback(self, msg):
        """Store wrist position from Vicon"""
        if msg.frame_number == self.last_frame:
            self.duplicates += 1
            return
        self.last_frame = msg.frame_number

        # Convert from mm to meters (Vicon outputs in mm)
        position = np.array([msg.x_trans, msg.y_trans, msg.z_trans]) / 1000.0
        if not position.any():  # occluded object -> SDK reports exactly 0,0,0
            self.occluded += 1
            return

        # Arrival time on this machine's clock, the same clock as the RealSense tracker
        timestamp = self.get_clock().now().nanoseconds / 1e9
        self.trajectory.append(position)
        self.timestamps.append(timestamp)
        self.quats.append((msg.x_rot, msg.y_rot, msg.z_rot, msg.w))

        # Log occasionally
        if len(self.trajectory) % 100 == 0:  # Every 100 frames (~1 sec at 100Hz)
            self.get_logger().info(
                f"Wrist: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) m | "
                f"Samples: {len(self.trajectory)} | occluded: {self.occluded}"
            )

    def save_trajectory(self):
        """Save trajectory to file"""
        if len(self.trajectory) == 0:
            self.get_logger().warn(f"No trajectory data to save! (occluded frames: {self.occluded})")
            return

        trajectory_array = np.array(self.trajectory)
        timestamps_array = np.array(self.timestamps)

        # Save as numpy file
        np.save(self.output_file, trajectory_array)

        # Also save timestamps
        timestamp_file = self.output_file.replace('.npy', '_timestamps.npy')
        np.save(timestamp_file, timestamps_array)
        np.save(self.output_file.replace('.npy', '_quat.npy'), np.array(self.quats))

        self.get_logger().info(f"✓ Saved {len(trajectory_array)} points to {self.output_file}")
        self.get_logger().info(f"✓ Timestamps saved to {timestamp_file}")
        self.get_logger().info(f"  occluded frames dropped: {self.occluded} | duplicate frames: {self.duplicates}")

        # Print statistics
        if len(timestamps_array) > 1:
            self.get_logger().info("\nTrajectory Statistics:")
            self.get_logger().info(f"  Duration: {timestamps_array[-1] - timestamps_array[0]:.2f} seconds")
            self.get_logger().info(f"  Sample rate: {len(trajectory_array) / (timestamps_array[-1] - timestamps_array[0]):.1f} Hz")
            self.get_logger().info(f"  X range: {trajectory_array[:, 0].min():.3f} to {trajectory_array[:, 0].max():.3f} m")
            self.get_logger().info(f"  Y range: {trajectory_array[:, 1].min():.3f} to {trajectory_array[:, 1].max():.3f} m")
            self.get_logger().info(f"  Z range: {trajectory_array[:, 2].min():.3f} to {trajectory_array[:, 2].max():.3f} m")


def main(args=None):
    parser = argparse.ArgumentParser(description='Track wrist with Vicon')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--marker', type=str, default='wrist',
                        help='Vicon Tracker object name (default: wrist)')
    cli_args = parser.parse_args()

    if Position is None:
        raise SystemExit("vicon_receiver is not available: source ~/vicon_ws/install/setup.bash first")

    rclpy.init(args=args)

    tracker = ViconWristTracker(output_file=cli_args.output, marker_name=cli_args.marker)

    try:
        print("\n" + "="*60)
        print("Vicon Wrist Tracking")
        print("="*60)
        print("\nInstructions:")
        print("1. Make sure the Vicon driver is running:")
        print("   ros2 run vicon_receiver vicon_client --ros-args -p hostname:=<VICON_PC_IP>")
        print("2. Verify the object is tracked:")
        print(f"   ros2 topic echo /vicon/{cli_args.marker}/{cli_args.marker}")
        print("3. Perform the trajectory")
        print("4. Press Ctrl+C to stop and save")
        print("\n" + "="*60 + "\n")

        rclpy.spin(tracker)

    except KeyboardInterrupt:
        print("\n\nStopping tracker...")

    finally:
        tracker.save_trajectory()
        tracker.destroy_node()
        rclpy.try_shutdown()  # SIGINT may already have shut the context down


if __name__ == '__main__':
    main()
