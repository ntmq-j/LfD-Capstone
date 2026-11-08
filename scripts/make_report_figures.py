#!/usr/bin/env python3
"""Method-section figures for the final report.

  fig1  system architecture (diagram)
  fig3  camera view under condition C, from recorded rosbags
  fig4  coordinate frames and the comparison model (diagram)

Usage: python make_report_figures.py [fig1 fig3 fig4]   (default: all)
Writes docs/report/figures/fig*.png
"""
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle, Rectangle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'docs', 'report', 'figures')
DATA = os.path.join(ROOT, 'data')

GREY, BLUE, ORANGE, GREEN, RED = '#4d4d4d', '#2b6cb0', '#dd6b20', '#2f855a', '#c53030'
LAG = 0.36  # RealSense sample stamps trail image capture by ~360 ms (estimated vs Vicon)


# --------------------------------------------------------------------------- fig 1
def box(ax, x, y, w, h, title, lines=(), fc='white', ec=GREY, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h - 0.13, title, ha='center', va='top', fontsize=fs + 0.5, weight='bold')
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.42 - i * 0.24, line, ha='center', va='top', fontsize=fs - 0.5)


def group(ax, x, y, w, h, label, color):
    ax.add_patch(Rectangle((x, y), w, h, fc=color, ec='none', alpha=0.09))
    ax.add_patch(Rectangle((x, y), w, h, fc='none', ec=color, lw=1.0, ls='--'))
    ax.text(x + 0.08, y + h - 0.08, label, ha='left', va='top', fontsize=8.5, color=color, weight='bold')


def arrow(ax, p, q, text=None, color=GREY, off=(0, 0.1), rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=11, lw=1.2, color=color,
                                 connectionstyle=f'arc3,rad={rad}'))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, ha='center',
                va='bottom', fontsize=7.2, color=color)


def fig1():
    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.set_xlim(0, 17.8)
    ax.set_ylim(0, 8.8)
    ax.axis('off')

    group(ax, 0.2, 1.2, 3.4, 7.2, 'Demonstration', GREY)
    box(ax, 0.45, 2.9, 2.9, 3.5, 'Operator + task board',
        ['3 tasks (T1 peg, T2 bolt,', 'T3 RJ45)', 'conditions A / B / C', 'wrist marker band',
         '(7 markers)', '4 board-corner markers', '(registration)'], fs=8)

    group(ax, 4.2, 5.0, 3.6, 3.4, 'Vision system', BLUE)
    box(ax, 4.45, 5.45, 3.1, 2.35, 'Intel RealSense D435i',
        ['colour 640×480 @ 30 Hz', '+ aligned depth', 'realsense2_camera'], ec=BLUE)
    group(ax, 4.2, 1.2, 3.6, 3.4, 'Motion-capture reference', ORANGE)
    box(ax, 4.45, 1.6, 3.1, 2.4, '4 × Vicon Vero v2.2',
        ['Vicon Tracker 3.9, 100 Hz', 'objects: wrist, board', 'DataStream SDK (TCP)'], ec=ORANGE)

    group(ax, 8.4, 0.2, 4.8, 8.2, 'ROS 2 Humble host (Ubuntu 22.04 VM)', GREEN)
    box(ax, 8.65, 5.2, 4.0, 2.6, 'track_wrist_realsense.py',
        ['MediaPipe Hands, landmark 0', 'depth: median of 5×5 window', 'speed gate 3 m/s',
         'Kalman bridge ≤ 0.5 s'], ec=BLUE)
    box(ax, 8.65, 2.6, 4.0, 2.0, 'track_wrist_vicon.py',
        ['via ros2-vicon-receiver (patched)', 'drops occluded/duplicate frames', 'position + orientation'],
        ec=ORANGE)
    box(ax, 8.65, 0.45, 4.0, 1.6, 'collect_data.py',
        ['start/stop per trial (key press)', 'health check · resume · rosbag2'], ec=GREEN)

    group(ax, 13.7, 3.1, 3.9, 5.3, 'Offline analysis', GREY)
    box(ax, 13.95, 5.75, 3.4, 2.05, 'Per-trial files (.npy)',
        ['trajectory + measured mask', 'raw detections, frame times', 'Vicon pose + quaternion'], fs=8)
    box(ax, 13.95, 3.35, 3.4, 2.05, 'Sync & registration',
        ['lag correction (≈ −360 ms)', 'board-corner fit + lever arm', 'Metrics 1–3'], fs=8)
    group(ax, 13.7, 0.2, 3.9, 2.25, 'Robot verification', RED)
    box(ax, 13.95, 0.45, 3.4, 1.5, 'UR3e + OnRobot RG2',
        ['ur_robot_driver', 'replay of selected trajectories'], ec=RED, fs=8)

    arrow(ax, (3.35, 5.6), (4.45, 6.4), 'hand', BLUE, off=(-0.1, 0.12))
    arrow(ax, (3.35, 3.7), (4.45, 2.9), 'markers', ORANGE, off=(-0.1, 0.12))
    arrow(ax, (7.55, 6.6), (8.65, 6.6), 'RGB-D', BLUE, off=(0, 0.06))
    arrow(ax, (7.55, 2.8), (8.65, 3.6), 'poses', ORANGE, off=(-0.05, 0.12))
    arrow(ax, (12.65, 7.2), (13.95, 7.2), None, GREY)
    arrow(ax, (12.65, 3.6), (13.95, 6.0), None, GREY)
    arrow(ax, (15.65, 5.75), (15.65, 5.4), None, GREY)
    arrow(ax, (15.65, 3.35), (15.65, 1.95), 'trajectories', RED, off=(0.8, -0.05))
    # orchestration: collect_data starts/stops both trackers
    for y_end in (2.6, 5.2):
        x = 9.4 if y_end == 2.6 else 8.9
        ax.add_patch(FancyArrowPatch((x, 2.05), (x, y_end), arrowstyle='-|>', mutation_scale=9, lw=1.0,
                                     color=GREEN, ls='--'))
    ax.text(9.5, 2.3, 'start / stop', fontsize=7, color=GREEN, va='center')

    fig.savefig(os.path.join(OUT, 'fig1_architecture.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------- fig 3
def bag_frame(bag, t_target):
    """Colour frame nearest t_target (s, host clock) and the camera matrix."""
    from rosbag2_py import SequentialCompressionReader, SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge
    import cv2
    compressed = bool(glob.glob(f'{bag}/*.zstd'))
    reader = SequentialCompressionReader() if compressed else SequentialReader()
    reader.open(StorageOptions(uri=bag, storage_id='sqlite3'), ConverterOptions('cdr', 'cdr'))
    best, K, bridge = None, None, CvBridge()
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic.endswith('/color/camera_info') and K is None:
            K = np.array(deserialize_message(data, CameraInfo).k).reshape(3, 3)
        elif topic.endswith('/color/image_raw'):
            msg = deserialize_message(data, Image)
            ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if best is None or abs(ts - t_target) < abs(best[0] - t_target):
                best = (ts, msg)
            elif ts > t_target + 1:
                break
    del reader
    if compressed:  # SequentialCompressionReader leaves the decompressed copy behind
        for f in glob.glob(f'{bag}/*.db3'):
            os.remove(f)
    img = cv2.cvtColor(bridge.imgmsg_to_cv2(best[1], 'rgb8'), cv2.COLOR_RGB2BGR)
    return best[0], img, K


def target_time(trial):
    """Host time in the middle of the at-target phase (slow, far from START)."""
    p = f'{DATA}/trajectories/realsense/{trial}'
    x, t = np.load(p + '.npy'), np.load(p + '_timestamps.npy') * 1.0
    d = np.linalg.norm(x - x[:10].mean(0), axis=1)
    v = np.r_[0, np.linalg.norm(np.diff(x, axis=0), axis=1) / np.diff(t)]
    at = np.where((d > 0.6 * d.max()) & (np.convolve(v, np.ones(5) / 5, 'same') < 0.08))[0]
    return t[at[len(at) // 2]] if len(at) else t[np.argmax(d)], x, t


def annotate(img, K, x, t, ts, label):
    """Draw the MediaPipe skeleton of the tracked hand and the saved wrist sample."""
    import cv2
    import mediapipe as mp
    out = img.copy()
    j = np.argmin(abs(t - (ts + LAG)))  # sample computed from this frame
    res = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                   min_detection_confidence=0.5).process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if abs(t[j] - (ts + LAG)) < 0.15:
        q = x[j]
        u, v = int(K[0, 0] * q[0] / q[2] + K[0, 2]), int(K[1, 1] * q[1] / q[2] + K[1, 2])
        for lm in res.multi_hand_landmarks or []:
            if np.hypot(lm.landmark[0].x * img.shape[1] - u, lm.landmark[0].y * img.shape[0] - v) < 40:
                mp.solutions.drawing_utils.draw_landmarks(out, lm, mp.solutions.hands.HAND_CONNECTIONS)
        cv2.circle(out, (u, v), 9, (0, 200, 0), -1)
        cv2.circle(out, (u, v), 9, (255, 255, 255), 2)
    else:
        cv2.putText(out, 'no wrist detection', (12, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(out, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3)
    cv2.putText(out, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
    return out


def fig3(panels_spec=(('task1_condC_trial6', 3.45, '(a) Task 1, condition C: inserting the peg'),
                      ('task3_condC_trial12', None, '(b) Task 3, condition C: bimanual'))):
    """Frame times are seconds after the trial's first sample (None = middle of the
    at-target phase); 3.45 s is the insertion moment in task1_condC_trial6."""
    panels = []
    for trial, rel, label in panels_spec:
        tt, x, t = target_time(trial)
        if rel is not None:
            tt = t[0] + rel
        ts, img, K = bag_frame(f'{DATA}/bags/{trial}', tt)
        panels.append(annotate(img, K, x, t, ts, label))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, im in zip(axes, panels):
        ax.imshow(im[:, :, ::-1])
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig3_condition_c_view.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------- fig 4
def triad(ax, o, ang, name, color, L=0.55, label_at=None):
    """2D sketch of a frame: two axes, first along ang."""
    a = np.radians(ang)
    ex, ey = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    ax.add_patch(FancyArrowPatch(o, o + L * ex, arrowstyle='-|>', mutation_scale=9, lw=1.6, color=color))
    ax.add_patch(FancyArrowPatch(o, o + L * ey, arrowstyle='-|>', mutation_scale=9, lw=1.6, color=color))
    ax.text(*(label_at if label_at is not None else o - 0.28 * ex - 0.22 * ey), name, fontsize=11, color=color, weight='bold', ha='center', va='center')


def fig4():
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.set_xlim(-0.6, 12.6)
    ax.set_ylim(-0.2, 7.3)
    ax.set_aspect('equal')
    ax.axis('off')

    # table, board, cube, corner markers
    ax.add_patch(Rectangle((1.0, 1.5), 9.5, 0.25, fc='#e9d8a6', ec=GREY))
    ax.add_patch(Rectangle((2.2, 1.75), 6.5, 0.18, fc='#a0aec0', ec=GREY))
    ax.add_patch(Rectangle((3.4, 1.93), 0.7, 0.55, fc='#cbd5e0', ec=GREY))
    ax.text(3.75, 2.2, 'cube', fontsize=7, ha='center', va='center')
    corners = [(2.3, 2.07), (8.6, 2.07)]
    for c in corners:
        ax.add_patch(Circle(c, 0.12, fc='#d9d9d9', ec='black', lw=1, zorder=4))
    ax.text(5.45, 1.42, 'task board; corner markers seen by both systems (2 of 4 shown)', fontsize=8,
            ha='center', va='top', color=GREY)

    # RealSense camera, looking down toward the board
    cam = np.array([1.2, 5.6])
    rot = np.array([[np.cos(np.radians(-30)), -np.sin(np.radians(-30))],
                    [np.sin(np.radians(-30)), np.cos(np.radians(-30))]])
    body = np.array([[-0.35, -0.2], [0.35, -0.2], [0.35, 0.2], [-0.35, 0.2]]) @ rot.T + cam
    ax.add_patch(Polygon(body, fc='#2d3748', ec='black'))
    for end in corners:
        ax.plot([cam[0], end[0]], [cam[1], end[1]], color=BLUE, lw=0.8, ls=':')
    ax.text(-0.55, 6.05, 'RealSense D435i', fontsize=9, color=BLUE, weight='bold')
    triad(ax, cam + np.array([0.05, -0.1]), -60, '{C}', BLUE, label_at=(2.0, 5.65))

    # Vicon cameras and world frame
    for vx, vy in [(0.2, 6.65), (2.6, 6.75)]:   # behind the board, beside the RealSense
        ax.add_patch(Rectangle((vx - 0.25, vy - 0.15), 0.5, 0.3, fc='#2d3748', ec='black'))
        ax.add_patch(Circle((vx + 0.25, vy - 0.08), 0.08, fc=ORANGE, ec='none'))
        for target in ((corners[1], (6.45, 4.6)) if vx < 1 else ((6.45, 4.6),)):
            ax.plot([vx + 0.3, target[0]], [vy - 0.1, target[1]], color=ORANGE, lw=0.8, ls=':')
    ax.text(-0.55, 7.05, 'Vicon Vero cameras (same side as the RealSense)', fontsize=9, color=ORANGE,
            weight='bold')
    triad(ax, np.array([10.6, 0.5]), 0, '{V}', ORANGE)
    ax.text(11.25, 0.35, 'Vicon world frame\n(redefined at every\ncalibration)', fontsize=7.5,
            color=ORANGE, ha='left', va='top')

    # forearm, hand, marker band, landmark 0
    ax.add_patch(Polygon([[8.6, 4.3], [6.0, 3.55], [5.8, 3.05], [8.5, 3.75]], fc='#f6d5c2', ec='#9c6b53'))
    ax.add_patch(Polygon([[5.85, 3.6], [4.6, 3.0], [4.3, 2.45], [4.75, 2.35], [5.75, 3.0]],
                         fc='#f6d5c2', ec='#9c6b53'))
    band = np.array([6.45, 3.88])
    ax.add_patch(Polygon(np.array([[-0.28, -0.1], [0.28, -0.1], [0.28, 0.12], [-0.28, 0.12]]) @ np.array(
        [[np.cos(0.27), np.sin(0.27)], [-np.sin(0.27), np.cos(0.27)]]) + band, fc='black'))
    for dx, dy in [(-0.45, 0.6), (0.0, 0.75), (0.45, 0.62), (0.65, 0.25)]:
        ax.plot([band[0], band[0] + dx], [band[1] + 0.1, band[1] + 0.1 + dy], color='black', lw=1)
        ax.add_patch(Circle((band[0] + dx, band[1] + 0.1 + dy), 0.09, fc='#d9d9d9', ec='black', lw=0.8))
    ax.add_patch(FancyArrowPatch(band + [0, 0.12], band + [0.45, 0.24], arrowstyle='-|>',
                                 mutation_scale=8, lw=1.4, color=ORANGE))
    ax.add_patch(FancyArrowPatch(band + [0, 0.12], band + [-0.12, 0.55], arrowstyle='-|>',
                                 mutation_scale=8, lw=1.4, color=ORANGE))
    ax.text(7.25, 4.25, '{W} wrist band', fontsize=9, color=ORANGE, weight='bold')
    l0 = np.array([5.85, 3.25])
    ax.add_patch(Circle(l0, 0.1, fc=GREEN, ec='white', lw=1.5, zorder=5))
    ax.text(4.5, 3.4, 'landmark 0\n(MediaPipe wrist)', fontsize=8, color=GREEN, ha='right', va='top')
    ax.add_patch(FancyArrowPatch(band + [0.0, 0.05], l0 + [0.05, 0.05], arrowstyle='-|>',
                                 mutation_scale=10, lw=1.8, color=RED, zorder=6))
    ax.text(6.3, 3.3, r'$\mathbf{d}$', fontsize=14, color=RED)

    # measured vectors
    ax.add_patch(FancyArrowPatch(cam + [0.25, -0.3], l0 + [-0.08, 0.1], arrowstyle='-|>', mutation_scale=10,
                                 lw=1.3, color=BLUE, ls='--'))
    ax.text(3.0, 4.75, r'$\mathbf{p}_{RS}$ (in {C})', fontsize=10, color=BLUE)
    ax.add_patch(FancyArrowPatch([10.6, 0.5], band + [0.2, 0.0], arrowstyle='-|>', mutation_scale=10,
                                 lw=1.3, color=ORANGE, ls='--', connectionstyle='arc3,rad=0.25'))
    ax.text(6.85, 2.75, r'$\mathbf{p}_{V}, \mathbf{R}_{C}$ (in {V})', fontsize=9.5, color=ORANGE, va='top')

    # camera -> Vicon transform, routed under the table
    path = np.array([[0.95, 5.35], [0.15, 5.35], [0.15, 0.9], [10.3, 0.9]])
    ax.plot(path[:-1, 0], path[:-1, 1], color=GREY, lw=1.2)
    ax.add_patch(FancyArrowPatch(path[-2], path[-1], arrowstyle='-|>', mutation_scale=10, lw=1.2, color=GREY))
    ax.text(5.2, 0.8, r'$(\mathbf{R},\mathbf{t})$: camera → Vicon, fitted on the board-corner markers',
            fontsize=9, color=GREY, ha='center', va='top')

    ax.text(9.3, 6.95, r'$\mathbf{R}\,\mathbf{p}_{RS} + \mathbf{t} \;=\; \mathbf{p}_{V} + \mathbf{R}_{C}\,\mathbf{d}$',
            fontsize=15, ha='center', va='center', bbox=dict(fc='white', ec=GREY, boxstyle='round,pad=0.4'))
    ax.text(9.3, 6.4, 'error = norm of (left − right) at each measured RealSense sample, after lag correction',
            fontsize=7.5, ha='center', color=GREY)

    fig.savefig(os.path.join(OUT, 'fig4_frames.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    todo = sys.argv[1:] or ['fig1', 'fig3', 'fig4']
    for name in todo:
        globals()[name]()
        print('done', name)
