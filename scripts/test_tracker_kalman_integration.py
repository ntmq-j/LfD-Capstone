#!/usr/bin/env python3
"""
Integration test for the Kalman logic inside RealSenseWristTracker, without
camera or MediaPipe: _detect_wrist is replaced by a scripted sequence and the
ROS clock by a fake one. Exercises the buffered-bridge behaviour:

  40 detections | 8-frame gap  -> bridged (<= 0.5 s), committed on re-detection
  20 detections | 40-frame gap -> too long: predictions dropped, filter reset
  20 detections | 3-frame gap  -> still pending at shutdown: dropped

Run (needs ROS sourced for rclpy):
  source /opt/ros/humble/setup.bash && venv/bin/python scripts/test_tracker_kalman_integration.py
"""
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
cv2.imshow = lambda *a, **k: None          # headless
cv2.waitKey = lambda *a, **k: -1

import rclpy  # noqa: E402
from track_wrist_realsense import RealSenseWristTracker  # noqa: E402

DT = 0.05
T0 = 1000.0
SEQ = ['D'] * 40 + ['N'] * 8 + ['D'] * 20 + ['N'] * 40 + ['D'] * 20 + ['N'] * 3


def truth(k):
    return np.array([0.25 * k * DT, 0.05, 0.60])          # 0.25 m/s along x


def measurement(k):
    return truth(k) + np.random.default_rng(k).normal(0, 0.003, 3)


def main():
    rclpy.init()
    td = tempfile.mkdtemp()
    out = os.path.join(td, 'trial.npy')
    tr = RealSenseWristTracker(output_file=out, target_hand='Right')
    tr.fx = tr.fy = 610.0
    tr.cx, tr.cy = 320.0, 240.0
    tr.color_image = np.zeros((480, 640, 3), np.uint8)
    tr.depth_image = np.full((480, 640), 600, np.uint16)
    tr.mp_drawing.draw_landmarks = lambda *a, **k: None

    frame = [0]

    def fake_detect():
        k = frame[0]
        return None if SEQ[k] == 'N' else (measurement(k), 320, 240, None)

    class FakeClock:
        def now(self):
            class T:
                nanoseconds = int((T0 + frame[0] * DT) * 1e9)

                def to_msg(self):
                    from builtin_interfaces.msg import Time
                    return Time()
            return T()

    tr._detect_wrist = fake_detect
    tr.get_clock = lambda: FakeClock()

    for k in range(len(SEQ)):
        frame[0] = k
        tr.process_frame()

    pending_at_save = len(tr.pending)
    tr.save_trajectory()

    meas = np.array(tr.is_measured)
    traj = np.array(tr.trajectory)
    ts = np.array(tr.timestamps)
    print(f"frames fed {len(SEQ)} | stored {len(traj)} | measured {meas.sum()} | predicted {(~meas).sum()}")
    print(f"predicted_count={tr.predicted_count} dropped={tr.dropped_predictions} "
          f"gap_count={tr.gap_count} pending@save={pending_at_save}\n")

    ok = True

    def check(cond, msg):
        nonlocal ok
        print(('  PASS  ' if cond else '  FAIL  ') + msg)
        ok &= bool(cond)

    check(meas.sum() == 80, "80 detections stored as measured")
    check((~meas).sum() == 8 and tr.predicted_count == 8, "8-frame gap bridged with exactly 8 predictions")
    check(len(traj) == 88, "total stored = 80 measured + 8 predicted")
    check(tr.gap_count == 1, "long gap counted once")
    check(tr.dropped_predictions == 10 + 3,
          f"dead-end predictions dropped: 10 from the 40-frame gap + 3 pending at shutdown (got {tr.dropped_predictions})")
    check(np.all(np.diff(ts) > 0), "timestamps strictly increasing (bridge committed in order)")

    pred_idx = np.where(~meas)[0]
    check(np.array_equal(pred_idx, np.arange(40, 48)), "predictions occupy frames 40..47 in the stored path")
    pred_err = np.linalg.norm(traj[pred_idx] - np.array([truth(k) for k in range(40, 48)]), axis=1) * 1000
    print("        bridged-gap prediction error (mm): " + " ".join(f"{e:.1f}" for e in pred_err))
    check(pred_err.max() < 30, "bridged predictions within 30 mm of truth")

    # After a reset the first detection must re-initialise the filter to the
    # measurement EXACTLY - no stale velocity or blending with the old state.
    first_after = int(np.where(ts > T0 + 108 * DT - 1e-6)[0][0])
    reinit_dev = np.linalg.norm(traj[first_after] - measurement(108)) * 1000
    check(reinit_dev < 1e-6, f"clean re-init: stored == first new measurement (dev {reinit_dev:.2e} mm)")

    jumps = np.linalg.norm(np.diff(traj, axis=0), axis=1) * 1000
    big = np.where(jumps > 100)[0]
    check(len(big) == 1 and abs(jumps[big[0]] - 500) < 60,
          f"exactly one large step, across the unbridged hole ({jumps[big].round(0)} mm)")

    files = sorted(os.listdir(td))
    print("        files:", files)
    m = np.load(os.path.join(td, 'trial_measured.npy'))
    r = np.load(os.path.join(td, 'trial_raw.npy'))
    check(len(m) == 88 and m.dtype == bool, "_measured.npy matches trajectory length, dtype bool")
    check(len(r) == 80, "_raw.npy holds only the 80 real measurements")

    tr.destroy_node()
    rclpy.shutdown()
    print("\n" + ("ALL PASS" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
