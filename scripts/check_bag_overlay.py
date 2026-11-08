#!/usr/bin/env python3
"""Draw the saved RealSense wrist position on the frames of the trial's rosbag,
so you can see which hand the tracker actually followed.

Needed because a wrong-hand switch is silent: in Task 3 condition C the left
forearm crosses the right wrist, MediaPipe labels it 'Right', and the speed
gate, Kalman filter and collect_data check all pass it.

Usage: python check_bag_overlay.py task3_condC_trial7 [--start 3 --end 11 --step 0.5]
Writes /tmp/<trial>_overlay.png: red dot = saved wrist (hollow = Kalman-predicted),
white skeleton = the MediaPipe hand whose wrist is at that dot, so the fingers show
which hand it is. 'NO SAMPLE' = the tracker saved nothing near that frame (hand lost).
"""
import mediapipe as mp
import argparse
import numpy as np
import cv2
from rosbag2_py import SequentialCompressionReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

DATA = '/home/jay/lfd_project/data'
LAG = 0.36  # RealSense sample stamps trail image capture by ~360 ms (estimated vs Vicon)

ap = argparse.ArgumentParser()
ap.add_argument('trial')
ap.add_argument('--start', type=float, default=None, help='s after the first sample (default: motion start)')
ap.add_argument('--end', type=float, default=None, help='default: motion end')
ap.add_argument('--step', type=float, default=0.4)
a = ap.parse_args()

p = f'{DATA}/trajectories/realsense/{a.trial}'
x = np.load(p + '.npy')
t = np.load(p + '_timestamps.npy') * 1.0
meas = np.load(p + '_measured.npy')
t0 = t[0]
t = t - t0
if a.start is None or a.end is None:  # default window: when the wrist is moving
    v = np.r_[0, np.linalg.norm(np.diff(x, axis=0), axis=1) / np.diff(t)]
    mov = np.where(np.convolve(v, np.ones(7) / 7, 'same') > 0.08)[0]
    a.start = a.start if a.start is not None else t[mov[0]]
    a.end = a.end if a.end is not None else t[mov[-1]]
want = list(np.arange(a.start, a.end + 1e-6, a.step))

reader = SequentialCompressionReader()  # decompresses next to the bag; deleted below
reader.open(StorageOptions(uri=f'{DATA}/bags/{a.trial}', storage_id='sqlite3'),
            ConverterOptions('cdr', 'cdr'))
bridge, K, frames = CvBridge(), None, []
while reader.has_next() and len(frames) < len(want):
    topic, data, _ = reader.read_next()
    if topic.endswith('/color/camera_info') and K is None:
        K = deserialize_message(data, CameraInfo).k
    elif topic.endswith('/color/image_raw'):
        msg = deserialize_message(data, Image)
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 - t0
        if ts >= want[len(frames)]:
            frames.append((ts, cv2.cvtColor(bridge.imgmsg_to_cv2(msg, 'rgb8'), cv2.COLOR_RGB2BGR)))
del reader
import glob, os
for f in glob.glob(f'{DATA}/bags/{a.trial}/*.db3'):
    if os.path.exists(f + '.zstd'):
        os.remove(f)

fx, fy, cx, cy = K[0], K[4], K[2], K[5]
hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2)
tiles = []
for ts, im in frames:
    j = np.argmin(abs(t - (ts + LAG)))  # the sample computed from this frame
    if abs(t[j] - (ts + LAG)) > 0.15:  # nearest saved sample is stale: nothing was tracked here
        cv2.putText(im, f'{ts:.1f}s NO SAMPLE', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        tiles.append(cv2.resize(im, (320, 240)))
        continue
    q = x[j]
    u, v = int(fx * q[0] / q[2] + cx), int(fy * q[1] / q[2] + cy)
    res = hands.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    near = [lm for lm in (res.multi_hand_landmarks or [])
            if np.hypot(lm.landmark[0].x * 640 - u, lm.landmark[0].y * 480 - v) < 40]
    for lm in near[:1]:
        mp.solutions.drawing_utils.draw_landmarks(im, lm, mp.solutions.hands.HAND_CONNECTIONS)
    cv2.circle(im, (u, v), 12, (0, 0, 255), -1 if meas[j] else 3)
    cv2.putText(im, f'{ts:.1f}s z={q[2]:.2f}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
    tiles.append(cv2.resize(im, (320, 240)))
while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))
out = f'/tmp/{a.trial}_overlay.png'
cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
print(out)
