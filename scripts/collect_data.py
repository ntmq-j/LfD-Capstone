#!/usr/bin/env python3
"""
Automated data collection script for Capstone B experiments
Runs synchronized recording from both Vicon and RealSense
"""

import rclpy
from rclpy.node import Node
import argparse
import signal
import subprocess
import sys
import time
import os
import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))


class DataCollectionManager(Node):
    def __init__(self, tasks, conditions, trials, hand='Right', vicon=False, bag=True,
                 data_dir='/home/jay/lfd_project/data'):
        super().__init__('data_collection_manager')

        # Experiment configuration (proposal Table 2: task1 peg-in-hole 5 mm,
        # task2 bolt tightening 2-3 mm, task3 connector mating 1 mm)
        self.tasks = tasks
        self.conditions = conditions
        self.trials_per_condition = {task: trials for task in tasks}
        self.hand = hand
        self.vicon = vicon
        self.bag = bag

        self.base_dir = data_dir
        for sub in ('realsense', 'vicon'):
            os.makedirs(os.path.join(data_dir, 'trajectories', sub), exist_ok=True)
        self.current_trial = 0

    # sys.executable, not 'python3': the trackers need the venv interpreter
    # (mediapipe, numpy<2). A bare 'python3' silently picks system python
    # when the venv is not on PATH.
    def start_vicon_tracking(self, output_file):
        """Start Vicon wrist tracker"""
        cmd = [sys.executable, os.path.join(SCRIPTS, 'track_wrist_vicon.py'),
               '--output', output_file]
        return subprocess.Popen(cmd)

    def start_realsense_tracking(self, output_file):
        """Start RealSense wrist tracker"""
        cmd = [sys.executable, os.path.join(SCRIPTS, 'track_wrist_realsense.py'),
               '--output', output_file, '--hand', self.hand]
        return subprocess.Popen(cmd)

    def start_rosbag_recording(self, bag_name):
        """Start rosbag2 recording"""
        # Raw 640x480 colour + depth measured 5.1 GB per 2 min (~1.3 GB per 30 s
        # trial), so bags are opt-in (--bag) for a few evidence trials; zstd file
        # compression runs when the bag closes, not during tracking.
        cmd = [
            'ros2', 'bag', 'record',
            '-o', bag_name,
            '--compression-mode', 'file', '--compression-format', 'zstd',
            '/joint_states',
            '/scaled_joint_trajectory_controller/joint_trajectory',
            '/camera/camera/color/image_raw',
            '/camera/camera/aligned_depth_to_color/image_raw',
            '/camera/camera/color/camera_info',
            '/vicon/wrist/wrist'
        ]
        # stdin=DEVNULL: the recorder's SPACE-to-pause handler grabs the
        # terminal, and our input() then dies with EOFError.
        return subprocess.Popen(cmd, stdin=subprocess.DEVNULL)

    def check_vicon(self, marker='wrist', timeout=5.0):
        """Fail before trial 1, not after it. The Vicon tracker subprocess (and
        ros2 bag) need ~/vicon_ws sourced in this terminal, and the driver must
        be streaming; otherwise the trial silently records RealSense only."""
        try:
            from vicon_receiver.msg import Position
        except ImportError:
            raise SystemExit("--vicon: run 'source ~/vicon_ws/install/setup.bash' in this terminal first")
        got = []
        sub = self.create_subscription(Position, f'/vicon/{marker}/{marker}', got.append, 10)
        end = time.time() + timeout
        while not got and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.destroy_subscription(sub)
        if not got:
            raise SystemExit(f"--vicon: nothing on /vicon/{marker}/{marker} for {timeout:.0f} s - is "
                             "vicon_client running, with the same ROS_LOCALHOST_ONLY as this terminal?")
        p = got[-1]
        if not (p.x_trans or p.y_trans or p.z_trans):
            self.get_logger().warn("Vicon object is occluded right now (0,0,0) - check the markers in Tracker")
        self.get_logger().info("Vicon OK: /vicon/%s/%s is streaming" % (marker, marker))

    def check_rate(self, output_file):
        """Tell a stalled tracker (redo) from a lost hand (keep: Metric 2 data).
        _frames.npy holds every frame the tracker processed, so a gap there means
        no frames (VM load, frozen camera node). A gap only in the detections means
        frames kept coming but the hand was lost (Task 3 condC: 3-7 s while the
        left hand covered the right). Healthy trials ran at 16-21 Hz; 15 Hz and
        0.5 s (the Kalman bridging limit) are calibration knobs."""
        if not os.path.exists(output_file):
            return
        frames = np.load(output_file.replace('.npy', '_frames.npy'))
        det = np.load(output_file.replace('.npy', '_raw_timestamps.npy'))
        rate = (len(frames) - 1) / (frames[-1] - frames[0]) if len(frames) > 1 else 0.0
        stall = np.diff(frames).max() if len(frames) > 1 else 0.0
        lost = np.diff(det)[np.diff(det) > 0.5]
        msg = f"  {rate:.1f} Hz, longest stall {stall:.2f} s, hand lost >0.5 s: {len(lost)}x"
        if len(lost):
            msg += f" (longest {lost.max():.1f} s)"
        if rate < 15 or stall > 0.5:
            self.get_logger().error(
                msg + " - TRACKER STALLED: close other apps (restart realsense2_camera "
                "if the window froze), delete this trial's files, rerun")
        elif len(lost):
            self.get_logger().warn(msg + " - keep it: a lost hand is robustness data")
        else:
            self.get_logger().info(msg)

    def stop(self, proc, output_file=None, timeout=30):
        """Ctrl+C the process and wait until it has written its file, so the
        next trial never starts on top of a half-saved one."""
        if proc is None:
            return
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.get_logger().error(f"pid {proc.pid} did not exit, killing")
            proc.kill()
        if output_file and not os.path.exists(output_file):
            self.get_logger().error(f"NOT SAVED: {output_file} - redo this trial")

    def collect_trial(self, task, condition, trial_num):
        """Collect data for a single trial"""
        trial_name = f"{task}_{condition}_trial{trial_num}"
        self.get_logger().info(f"\n{'='*60}")
        self.get_logger().info(f"Starting: {trial_name}")
        self.get_logger().info(f"{'='*60}\n")
        # Recording starts on this ENTER, not on a timer: resetting Task 2
        # (unscrewing the nut) or swapping blocks takes longer than any fixed wait.
        input(f"Block for {task} in the vise, items on their marks, hand at START "
              f"-> press ENTER to start recording...")

        # Create output paths
        vicon_output = os.path.join(self.base_dir, 'trajectories', 'vicon', f'{trial_name}.npy')
        rs_output = os.path.join(self.base_dir, 'trajectories', 'realsense', f'{trial_name}.npy')
        bag_output = os.path.join(self.base_dir, 'bags', trial_name)

        # Start tracking systems
        vicon_proc = bag_proc = None
        if self.vicon:
            self.get_logger().info("Starting Vicon tracking...")
            vicon_proc = self.start_vicon_tracking(vicon_output)
            time.sleep(2)

        self.get_logger().info("Starting RealSense tracking...")
        rs_proc = self.start_realsense_tracking(rs_output)
        time.sleep(2)

        if self.bag:
            self.get_logger().info("Starting rosbag recording...")
            bag_proc = self.start_rosbag_recording(bag_output)
            time.sleep(2)

        # Wait for operator to perform demonstration
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("READY TO RECORD")
        self.get_logger().info("="*60)
        self.get_logger().info(f"Task: {task}")
        self.get_logger().info(f"Condition: {condition}")
        self.get_logger().info(f"Trial: {trial_num}")
        self.get_logger().info("\nPerform the demonstration now.")
        self.get_logger().info("Press ENTER when complete...")
        input()

        # Stop all recording
        self.get_logger().info("\nStopping recordings...")
        self.stop(vicon_proc, vicon_output)
        self.stop(rs_proc, rs_output)
        self.check_rate(rs_output)
        self.stop(bag_proc, timeout=900)  # zstd of a 5 GB bag takes minutes; killing it left no metadata.yaml

        self.get_logger().info(f"✓ Trial {trial_name} complete!")
        if self.vicon:
            self.get_logger().info(f"  Vicon data: {vicon_output}")
        self.get_logger().info(f"  RealSense data: {rs_output}")
        if self.bag:
            self.get_logger().info(f"  Bag file: {bag_output}")

    def run_experiment(self):
        """Run full experiment data collection"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("CAPSTONE B DATA COLLECTION")
        self.get_logger().info("="*60 + "\n")

        total_trials = sum(
            self.trials_per_condition[task] * len(self.conditions)
            for task in self.tasks
        )

        self.get_logger().info(f"Total trials to collect: {total_trials}")
        self.get_logger().info(f"Estimated time: ~{total_trials * 5} minutes\n")

        input("Press ENTER to start data collection...")

        trial_count = 0

        for task in self.tasks:
            num_trials = self.trials_per_condition[task]

            for condition in self.conditions:
                self.get_logger().info(f"\n{'='*60}")
                self.get_logger().info(f"TASK: {task}, CONDITION: {condition}")
                self.get_logger().info(f"{'='*60}\n")

                # Setup condition
                if condition == 'condB':
                    self.get_logger().warn("Set up REDUCED LIGHTING (~50%)")
                    input("Press ENTER when ready...")
                elif condition == 'condC':
                    self.get_logger().warn("Set up PARTIAL OCCLUSION (~30%)")
                    input("Press ENTER when ready...")
                else:
                    self.get_logger().info("Standard condition - full lighting")

                # Collect trials. Existing trials are skipped, never overwritten:
                # re-running the same command resumes; delete a file to redo it.
                for trial_num in range(1, num_trials + 1):
                    trial_count += 1
                    done = os.path.join(self.base_dir, 'trajectories', 'realsense',
                                        f'{task}_{condition}_trial{trial_num}.npy')
                    if os.path.exists(done):
                        continue
                    self.get_logger().info(f"\nProgress: {trial_count}/{total_trials}")
                    self.collect_trial(task, condition, trial_num)

        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("DATA COLLECTION COMPLETE!")
        self.get_logger().info("="*60 + "\n")
        self.get_logger().info(f"Total trials collected: {trial_count}")
        self.get_logger().info("\nNext step: Run analysis with compare_trajectories.py")


def main(args=None):
    parser = argparse.ArgumentParser(description='Collect demonstration trials')
    parser.add_argument('--task', nargs='+', choices=['task1', 'task2', 'task3'],
                        default=['task1', 'task2', 'task3'])
    parser.add_argument('--condition', nargs='+', choices=['condA', 'condB', 'condC'],
                        default=['condA', 'condB', 'condC'])
    parser.add_argument('--trials', type=int, default=20,
                        help='Trials per task per condition (default 20 = 180 total, proposal 3.5)')
    parser.add_argument('--hand', choices=['Left', 'Right'], default='Right',
                        help="Operator's physical hand to track (default Right)")
    parser.add_argument('--vicon', action='store_true', help='Also run the Vicon tracker (Week 13/14)')
    parser.add_argument('--bag', action='store_true',
                        help='Also record a rosbag (~1.3 GB per 30 s trial before compression)')
    parser.add_argument('--data-dir', default='/home/jay/lfd_project/data',
                        help='Vicon sessions go in their own dir (e.g. data/vicon_session1): '
                             'trial names repeat, and existing trials are skipped')
    cli = parser.parse_args()

    rclpy.init(args=args)
    manager = DataCollectionManager(cli.task, cli.condition, cli.trials, hand=cli.hand,
                                    vicon=cli.vicon, bag=cli.bag,
                                    data_dir=os.path.abspath(cli.data_dir))

    try:
        if cli.vicon:
            manager.check_vicon()
        manager.run_experiment()
    except KeyboardInterrupt:
        manager.get_logger().info("\nData collection interrupted!")

    manager.destroy_node()
    rclpy.try_shutdown()  # SIGINT may already have shut the context down


if __name__ == '__main__':
    main()
