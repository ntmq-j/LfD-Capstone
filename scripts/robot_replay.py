#!/usr/bin/env python3
"""Replay demonstrated Task 1 (peg-in-hole) trajectories on the UR3e.

Experiment: each selected demonstration (session 2, condition A, Task 1) is replayed twice,
once from the Vicon wrist path and once from the RealSense wrist path, so the two runs
differ only by the capture system's error. The robot starts with the peg already in the
gripper, follows the demonstrated transport (pick -> insertion), then descends at the
demonstrated insertion point with a force guard. Recorded: inserted / blocked, and the
horizontal placement error against the touched hole.

Board -> robot: the TCP touches the four drawn corner circles (where the board-corner Vicon
markers stood), giving a rigid Vicon -> robot fit. RealSense points go camera -> Vicon by the
board-corner registration (analyze_results.py).
Wrist -> peg offset (plan --offset):
  mean   - mean over the OTHER trials of (touched hole - Vicon wrist at insertion). Realistic: the
           Vicon run misses the hole by the operator's trial-to-trial variation of the wrist-to-peg
           relation (~5 mm SD); the RealSense run also by its own error.
  oracle - this trial's own Vicon insertion. The Vicon run lands on the hole by construction (it
           checks the registration chain); the RealSense run misses by its sensor error alone.

Lab sequence. touch works in Local mode; run needs Remote Control (External Control NOT running).
All touches are made with the peg already gripped (vertical), so the peg tip is the reference
point and the RG2 TCP definition does not matter.
  python robot_replay.py touch --ip 192.168.0.194            # 4 corners, hole, 4 hole-wall touches
  python robot_replay.py plan --tcp-z 0.20                    # TCP z from the pendant (IK check only)
  python robot_replay.py sim                                  # optional RViz check, see sim()
  python robot_replay.py run --ip 192.168.0.194 --lift 0.05  # dry run 5 cm higher, no descent
  python robot_replay.py run --ip 192.168.0.194              # real runs, results -> data/robot/results.json
Without a robot:  touch --sim  writes a synthetic touch file so plan/sim can be tried.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_results as A                 # noqa: E402  (corner registration, samples, lever_arm)
from compare_trajectories import kabsch     # noqa: E402
from fit_vicon_calib import load            # noqa: E402

OUT = os.path.join(A.DATA, 'robot')
LAG = -0.37                                   # session-2 RealSense lag (analysis, median of calib takes)
MARKER_CENTRE_H = 0.0158 - 0.0139 / 2         # board-corner sphere centre above the board surface
CORNERS = [('near_right', 'near-right corner circle (near = START edge; left/right as the operator sees it)'),
           ('far_right', 'far-right corner circle (camera side)'),
           ('far_left', 'far-left corner circle (camera side)'),
           ('near_left', 'near-left corner circle')]     # same order as the Vicon corners board1..4
CLEARANCE = 0.0055                            # Task 1 radial clearance (23.5 mm hole, 12.5 mm peg)
SAFE_UP = 0.03                                # transport/approach band: peg tip 20-50 mm above the cube top
                                              # (the UR3e cannot reach much higher with the tool pointing down)

# ---------------------------------------------------------------------------- UR3e kinematics
DH_A = [0, -0.24355, -0.2132, 0, 0, 0]
DH_D = [0.15185, 0, 0, 0.13105, 0.08535, 0.0921]
DH_AL = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]
JOINTS = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
          'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']


def fk(q, tcp_z=0.0):
    """Base -> TCP (TCP = flange + tcp_z along tool z), standard UR DH."""
    T = np.eye(4)
    for qi, a, d, al in zip(q, DH_A, DH_D, DH_AL):
        ct, st, ca, sa = np.cos(qi), np.sin(qi), np.cos(al), np.sin(al)
        T = T @ np.array([[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st],
                          [0, sa, ca, d], [0, 0, 0, 1]])
    return T @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, tcp_z], [0, 0, 0, 1]])


def ik(p, rv, q0, tcp_z):
    """Numerical IK near q0. Returns q, position error (m), orientation error (rad)."""
    Rg = Rot.from_rotvec(rv)

    def res(q):
        T = fk(q, tcp_z)
        return np.r_[T[:3, 3] - p, 0.1 * (Rot.from_matrix(T[:3, :3]) * Rg.inv()).as_rotvec()]
    s = least_squares(res, q0, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    return s.x, np.linalg.norm(s.fun[:3]), np.linalg.norm(s.fun[3:]) / 0.1


def ik_path(P, rv, q0, tcp_z, step=0.005):
    """Follow the waypoints from configuration q0 in small Cartesian steps (as moveL keeps the
    controller's branch). Returns q per waypoint, worst position error, and singularity margins."""
    q, cur, P = np.array(q0, float), fk(q0, tcp_z)[:3, 3], np.asarray(P, float)
    out, worst, elbow, wrist = [], 0.0, np.inf, np.inf
    for p in P:
        for s in np.linspace(0, 1, max(2, int(np.linalg.norm(p - cur) / step) + 1))[1:]:
            q, ep, _ = ik(cur + s * (p - cur), rv, q, tcp_z)
            elbow, wrist = min(elbow, abs(np.sin(q[2]))), min(wrist, abs(np.sin(q[4])))
        worst, cur = max(worst, ep), p
        out.append(q)
    return np.array(out), worst, np.degrees(np.arcsin(elbow)), np.degrees(np.arcsin(wrist))


def ik_first(p, rv, tcp_z):
    """Elbow-up, wrist-down seeds around the base; best residual wins."""
    best = None
    for pan in np.linspace(-np.pi, np.pi, 12, endpoint=False):
        q, ep, eo = ik(p, rv, np.array([pan, -1.6, 1.6, -1.57, -1.57, 0.0]), tcp_z)
        if best is None or ep + 0.01 * eo < best[1] + 0.01 * best[2]:
            best = (q, ep, eo)
    return best


# ---------------------------------------------------------------------------- data
def touch_file():
    return os.path.join(OUT, 'touch.json')


def receiver(ip):
    import rtde_receive
    return rtde_receive.RTDEReceiveInterface(ip, 125.0)      # 125 Hz: the default 500 Hz stream dropped in the VM


def fresh(rr, timeout=1.0):
    """Reconnect if the RTDE stream dropped and wait for a new sample, so no reading is stale."""
    if not rr.isConnected():
        print('  (RTDE stream dropped - reconnecting)')
        rr.reconnect()
    t, t_end = rr.getTimestamp(), time.time() + timeout
    while rr.getTimestamp() == t:
        if time.time() > t_end:
            raise RuntimeError('RTDE stream stalled - check the cable / robot and restart')
        time.sleep(0.002)
    return rr


def vicon_to_robot(bag, touched):
    """Rigid fit from the Vicon corner markers (projected to the board surface) to the touched
    circle centres. Corner order: near_right, far_right, far_left, near_left."""
    V = bag['V']
    n = np.cross(V[0] - V[3], V[2] - V[3])
    n /= np.linalg.norm(n)                                     # board normal, toward the hand
    k = [i for i, (c, _) in enumerate(CORNERS) if c in touched]  # a skipped (unreachable) corner is left out
    Vs = (V - MARKER_CENTRE_H * n)[k]
    Pt = np.array([touched[CORNERS[i][0]][:3] for i in k])
    R, t = kabsch(Vs, Pt)
    ey = V[2] - V[3] - (V[2] - V[3]) @ (V[0] - V[3]) / np.linalg.norm(V[0] - V[3]) ** 2 * (V[0] - V[3])
    return R, t, np.linalg.norm(Vs @ R.T + t - Pt, axis=1), R @ n, R @ ey / np.linalg.norm(ey)


def demo(name, bags, d_pool):
    """Vicon and RealSense wrist paths (Vicon frame) between pick and insertion, and the
    wrist positions at both events. Events come from the Vicon wrist: the insertion is the
    slowest moment in the far-most zone (toward the cube), the pick the slowest moment in the
    right-most zone before it (the peg holder)."""
    D = load(A.S2, name)
    t0 = D['vt'][0]
    bag = min(bags, key=lambda b: abs(b['t0'] - t0))
    V = bag['V']
    ex = (V[0] - V[3]) / np.linalg.norm(V[0] - V[3])          # near_left -> near_right
    ey = V[2] - V[3]
    ey -= ey @ ex * ex
    ey /= np.linalg.norm(ey)                                   # near -> far (camera side)
    others = [v for k, v in d_pool.items() if k != name]
    d = sum(v[0] for v in others) / sum(v[1] for v in others)  # lever arm, leave-one-trial-out
    vt = D['vt']
    L = D['v'] + np.einsum('nij,j->ni', Rot.from_quat(D['q']).as_matrix(), d)
    bx, by = (L - V[3]) @ ex, (L - V[3]) @ ey
    sp = np.r_[0, np.linalg.norm(np.diff(L, axis=0), axis=1) / np.diff(vt)]
    sp = np.convolve(sp, np.ones(15) / 15, 'same')
    far = np.where(by >= by.max() - 0.015)[0]
    i_ins = far[np.argmin(sp[far])]
    pre = np.arange(i_ins)
    right = pre[bx[pre] >= bx[pre].max() - 0.010]
    i_pick = right[np.argmin(sp[right])]
    t_pick, t_ins = vt[i_pick], vt[i_ins]
    m = D['m']
    rs_t = D['rt'][m] + LAG
    rs_p = D['r'][m] @ bag['R'].T + bag['t']                    # camera -> Vicon
    out = {'t0': t0, 'bag': bag}
    for src, t, p in (('vicon', vt, L), ('realsense', rs_t, rs_p)):
        at = lambda te: p[np.abs(t - te) < 0.15].mean(0) if (np.abs(t - te) < 0.15).any() else p[np.argmin(abs(t - te))]
        sel = (t >= t_pick) & (t <= t_ins)
        out[src] = dict(pick=at(t_pick), ins=at(t_ins), path=p[sel])
    return out


def downsample(P, step=0.015):
    keep = [P[0]]
    for p in P[1:]:
        if np.linalg.norm(p - keep[-1]) >= step:
            keep.append(p)
    if np.linalg.norm(P[-1] - keep[-1]) > 1e-6:
        keep.append(P[-1])
    return np.array(keep)


# ---------------------------------------------------------------------------- touch
def touch(args):
    os.makedirs(OUT, exist_ok=True)
    pts = {}
    if args.sim:
        # Board in front of the robot: near-left corner at (-0.20, -0.45), board x -> robot +x,
        # board y (toward the camera side) -> robot +y, surface at z = 0. Hole from the board layout.
        b0 = A.corner_registration(f'{A.S2}/bags/board')
        V = b0['V']
        n = np.cross(V[0] - V[3], V[2] - V[3])
        n /= np.linalg.norm(n)
        ex = (V[0] - V[3]) / np.linalg.norm(V[0] - V[3])
        ey = np.cross(n, ex)
        Rb = np.c_[ex, ey, n]
        to_r = lambda pv: (pv - V[3]) @ Rb + np.array([-0.20, -0.45, 0.0])
        Vs = V - MARKER_CENTRE_H * n
        for (c, _), v in zip(CORNERS, Vs):
            pts[c] = list(to_r(v)) + [np.pi, 0, 0]
        pts['hole'] = [-0.20 + 0.190, -0.45 + 0.226, 0.085, np.pi, 0, 0]
        pts['simulated'] = True
    else:
        rr = receiver(args.ip)
        tcp_off = lambda: np.array(fresh(rr).getActualTCPPose()[:3]) - fk(rr.getActualQ())[:3, 3]
        pts['tcp_offset'] = tcp_off().tolist()
        print(f'  active TCP is {np.linalg.norm(pts["tcp_offset"]) * 1000:.0f} mm from the flange - do not change it until the runs are done')
        steps = CORNERS + [('hole', 'Task 1 hole: centred over it, level with the cube top, '
                                    'in the gripper yaw you want for the runs')]
        for name, hint in steps:
            a = input(f'Peg gripped, gripper vertical: put the peg tip on the {hint}, then press ENTER '
                      f'(s + ENTER skips an unreachable corner; at least 3 corners) ')
            if a.strip().lower() == 's' and name != 'hole':
                continue
            pts[name] = list(fresh(rr).getActualTCPPose())
            pts['q_seed'] = list(rr.getActualQ())                 # the controller's arm configuration
            if np.linalg.norm(tcp_off() - pts['tcp_offset']) > 0.010:   # nominal-DH model differs from the calibrated arm by ~1-2 mm
                sys.exit('the active TCP changed during touching (gripper URCap?) - fix it and restart touch')
            print(f'  {name}: {np.round(np.array(pts[name][:3]) * 1000, 1)} mm')
        W = []
        for k in range(4):
            input(f'Hole wall {k + 1}/4: lower the peg ~5 mm into the hole and push it gently against the wall '
                  f'(4 directions spread around the hole, translation only), ENTER ')
            W.append(fresh(rr).getActualTCPPose()[:2])
        W = np.array(W)
        a, b, c = np.linalg.lstsq(np.c_[2 * W, np.ones(4)], (W ** 2).sum(1), rcond=None)[0]   # circle fit
        print(f'  hole-wall circle radius {np.sqrt(c + a * a + b * b) * 1000:.1f} mm (expect ~5.5); centre differs '
              f'from the eye estimate by {np.hypot(a - pts["hole"][0], b - pts["hole"][1]) * 1000:.1f} mm')
        pts['hole_eye'], pts['hole_walls'] = pts['hole'][:], W.tolist()
        pts['hole'][:2] = [float(a), float(b)]
        c0 = [c for c, _ in CORNERS if c in pts][0]
        input(f'Check repeatability: put the peg tip back on the {c0} circle, then press ENTER ')
        rep = np.array(fresh(rr).getActualTCPPose()[:3])
        print(f'  {c0} repeat differs by {np.linalg.norm(rep - pts[c0][:3]) * 1000:.1f} mm')
        assert sum(c in pts for c, _ in CORNERS) >= 3, 'need at least 3 corners'
        # tilt moves the peg tip ~4.6 mm/deg relative to the flange; yaw about the tool axis does not
        tilt = max(np.degrees(np.arccos(np.clip(-Rot.from_rotvec(v[3:]).as_matrix()[2, 2], -1, 1)))
                   for k, v in pts.items() if k in dict(CORNERS) or k == 'hole')
        print(f'  largest gripper tilt from vertical: {tilt:.1f} deg (keep < 0.5; press Align and re-touch if more)')
    json.dump(pts, open(touch_file(), 'w'), indent=1)
    json.dump(pts, open(os.path.join(OUT, time.strftime('touch_%m%d_%H%M.json')), 'w'), indent=1)
    print('saved', touch_file())


# ---------------------------------------------------------------------------- plan
def plan(args):
    touched = json.load(open(touch_file()))
    hole = np.array(touched['hole'][:3])
    ori = list(touched['hole'][3:])
    bags = [A.corner_registration(f'{A.S2}/bags/{n}') for n in ('board', 'board_end', 'board_end2')]

    # lever-arm pool: session-2 trials of the second band attachment (recorded after calib5)
    t_att2 = load(f'{A.S2}/calib5')['vt'][0]
    d_pool = {}
    for n in A.trials(A.S2):
        D = load(A.S2, n)
        if D['vt'][0] < t_att2:
            continue
        b = min(bags, key=lambda bb: abs(bb['t0'] - D['vt'][0]))
        S = A.samples(D, LAG)
        r = S['P'] @ b['R'].T + b['t'] - S['Q']
        d_pool[n] = (np.einsum('nji,nj->i', S['Rv'], r), len(r))

    names = [f'task1_condA_trial{k}' for k in range(1, 11)]
    demos = {n: demo(n, bags, d_pool) for n in names}
    fits = {}
    for n, dm in demos.items():
        key = id(dm['bag'])
        if key not in fits:
            fits[key] = vicon_to_robot(dm['bag'], touched)
        dm['R'], dm['t'], dm['res'], dm['up'], dm['ey'] = fits[key]
    for R, t, res, up, ey in fits.values():
        print(f'Vicon -> robot corner fit residuals: {np.round(res * 1000, 1)} mm')
    to_r = lambda dm, p: p @ dm['R'].T + dm['t']
    ins_v = {n: to_r(dm, dm['vicon']['ins']) for n, dm in demos.items()}
    # corner-label sanity: board normal must point up, and the hole must lie beyond the demo wrist
    # (away from the operator's START edge); a left/right or near/far mix-up fails one of these
    dm0 = demos[names[0]]
    dy = (hole - np.mean(list(ins_v.values()), axis=0)) @ dm0['ey'] * 1000
    print(f'label check: board normal z {dm0["up"][2]:.2f} (expect > 0.9), hole beyond demo wrist by {dy:.0f} mm '
          f'(expect 30-200)')
    if dm0['up'][2] < 0.9 or not 30 < dy < 200:
        sys.exit('corner labels look swapped: near = START edge, left/right as seen from START facing the cube. Re-touch.')

    z_app = hole[2] + args.peg_below + 0.020          # peg tip 20 mm above the cube top
    z_tgt = hole[2] + args.peg_below - args.depth      # peg tip `depth` into the hole
    trials = [int(x) for x in args.trials.split(',')]
    runs = []
    for i, k in enumerate(trials):
        n = f'task1_condA_trial{k}'
        dm = demos[n]
        offset = (hole - ins_v[n] if args.offset == 'oracle'                 # Vicon lands on the hole by construction
                  else np.mean([hole - ins_v[m] for m in names if m != n], axis=0))
        order = ('vicon', 'realsense') if i % 2 == 0 else ('realsense', 'vicon')   # counterbalanced
        for src in order:
            P = to_r(dm, dm[src]['path']) + offset
            P[:, 2] = np.clip(P[:, 2], z_app, z_app + SAFE_UP)
            ins = to_r(dm, dm[src]['ins']) + offset
            P = np.vstack([P, [ins[0], ins[1], z_app]])
            P = downsample(P)
            err = ins[:2] - hole[:2]
            runs.append(dict(trial=k, source=src, path=P.tolist(), insert_xy=ins[:2].tolist(),
                             pred_err_mm=(err * 1000).tolist(), pred_err_abs_mm=float(np.linalg.norm(err) * 1000)))

    # kinematic check on the UR3e model (the controller re-checks before running)
    q_seed = touched.get('q_seed')
    for r in runs:
        P = np.vstack([r['path'], [*r['insert_xy'], z_tgt]])
        if q_seed is None:
            q0 = ik_first(P[0], ori, args.tcp_z)[0]
        else:                                             # from the touch configuration, via the safe height
            q0 = ik_path([[*fk(q_seed, args.tcp_z)[:2, 3], z_app + SAFE_UP], [*P[0][:2], z_app + SAFE_UP]],
                         ori, q_seed, args.tcp_z)[0][-1]
        qs, worst, el, wr = ik_path(np.vstack([[*P[0][:2], z_app + SAFE_UP], P]), ori, q0, args.tcp_z)
        qs = qs[1:]
        r['q'] = qs.tolist()
        r['ik_max_err_mm'] = float(worst * 1000)
        r['elbow_margin_deg'], r['wrist_margin_deg'] = float(el), float(wr)
        r['max_joint_step_deg'] = float(np.degrees(np.abs(np.diff(qs, axis=0)).max()))
        r['reach_m'] = float(np.linalg.norm(P[:, :2], axis=1).max())

    pl = dict(offset_mode=args.offset, hole=hole.tolist(), orientation=ori, z_app=z_app, z_target=z_tgt, tcp_z=args.tcp_z,
              peg_below=args.peg_below, depth=args.depth, simulated=bool(touched.get('simulated')), runs=runs)
    json.dump(pl, open(os.path.join(OUT, 'plan.json'), 'w'), indent=1)
    json.dump(pl, open(os.path.join(OUT, f'plan_{args.offset}.json'), 'w'), indent=1)

    print(f"\n{'#':>2} {'trial':>5} {'source':9} {'waypoints':>9} {'pred. error x,y (mm)':>22} {'|err|':>6} "
          f"{'expect':8} {'IK err':>7} {'max dq':>7} {'reach':>6}")
    for i, r in enumerate(runs):
        ok = r['pred_err_abs_mm'] < CLEARANCE * 1000
        flag = '' if (r['ik_max_err_mm'] < 1 and r['max_joint_step_deg'] < 20 and r['reach_m'] < 0.48
                      and r['elbow_margin_deg'] > 15 and r['wrist_margin_deg'] > 15) else '  CHECK'
        print(f"{i:2d} {r['trial']:5d} {r['source']:9} {len(r['path']):9d} "
              f"{r['pred_err_mm'][0]:10.1f},{r['pred_err_mm'][1]:6.1f}     {r['pred_err_abs_mm']:6.1f} "
              f"{'insert' if ok else 'blocked':8} {r['ik_max_err_mm']:6.2f}mm {r['max_joint_step_deg']:6.1f}° "
              f"{r['reach_m']:5.2f}m  sing. margin {min(r['elbow_margin_deg'], r['wrist_margin_deg']):4.0f}°{flag}")
    for src in ('vicon', 'realsense'):
        e = [r['pred_err_abs_mm'] for r in runs if r['source'] == src]
        print(f'{src:9}: predicted |error| mean {np.mean(e):.1f} mm, within the {CLEARANCE * 1000:.1f} mm clearance: '
              f'{sum(x < CLEARANCE * 1000 for x in e)}/{len(e)}')
    print('saved', os.path.join(OUT, 'plan.json'))
    figure(pl, touched)


def figure(pl, touched):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    C = np.array([touched[c][:3] for c, _ in CORNERS if c in touched])
    C = np.vstack([C, C[:1]])
    ax.plot(C[:, 0], C[:, 1], color='#52514e', lw=1)
    h = pl['hole']
    ax.add_patch(plt.Circle(h[:2], 0.01175, fill=False, color='#0b0b0b', lw=1.2))
    col = {'vicon': '#2a78d6', 'realsense': '#eb6834'}
    for src in col:
        rs = [r for r in pl['runs'] if r['source'] == src]
        for i, r in enumerate(rs):
            P = np.array(r['path'])
            ax.plot(P[:, 0], P[:, 1], color=col[src], lw=1, alpha=0.6, label=src if i == 0 else None)
            ax.plot(*r['insert_xy'], 'o', color=col[src], ms=4)
    ax.plot(0, 0, 'k+', ms=10)
    ax.text(0.005, 0.005, 'robot base', fontsize=7)
    ax.set_aspect('equal')
    ax.set_xlabel('robot x (m)')
    ax.set_ylabel('robot y (m)')
    ax.legend(frameon=False, fontsize=8)
    ax.set_title('Planned TCP paths and insertion points (circle: hole)', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'plan.png'), dpi=180)
    print('saved', os.path.join(OUT, 'plan.png'))


# ---------------------------------------------------------------------------- sim (RViz)
def sim(args):
    """Kinematic simulation on the UR3e model. In two other terminals:
      ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="$(xacro \
        /opt/ros/humble/share/ur_description/urdf/ur.urdf.xacro ur_type:=ur3e name:=ur)"
      rviz2 -d /opt/ros/humble/share/ur_description/rviz/view_robot.rviz
    then add a MarkerArray display on /replay_markers."""
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    pl = json.load(open(os.path.join(OUT, 'plan.json')))
    touched = json.load(open(touch_file()))
    rclpy.init()
    node = Node('replay_sim')
    pj = node.create_publisher(JointState, '/joint_states', 10)
    pm = node.create_publisher(MarkerArray, '/replay_markers', 1)

    def markers():
        ma, col = MarkerArray(), {'vicon': (0.16, 0.47, 0.84), 'realsense': (0.92, 0.41, 0.2)}
        m = Marker(type=Marker.LINE_STRIP, id=0, ns='board')
        m.header.frame_id = 'base'
        m.scale.x = 0.003
        m.color.r = m.color.g = m.color.b = 0.4
        m.color.a = 1.0
        cs = [touched[c][:3] for c, _ in CORNERS if c in touched]
        m.points = [Point(x=p[0], y=p[1], z=p[2]) for p in cs + cs[:1]]
        ma.markers.append(m)
        h = Marker(type=Marker.CYLINDER, id=1, ns='hole')
        h.header.frame_id = 'base'
        h.pose.position = Point(x=pl['hole'][0], y=pl['hole'][1], z=pl['hole'][2])
        h.scale.x = h.scale.y = 0.0235
        h.scale.z = 0.002
        h.color.a = 1.0
        ma.markers.append(h)
        for i, r in enumerate(pl['runs']):
            s = Marker(type=Marker.LINE_STRIP, id=10 + i, ns='paths')
            s.header.frame_id = 'base'
            s.scale.x = 0.002
            s.color.r, s.color.g, s.color.b = col[r['source']]
            s.color.a = 0.8
            s.points = [Point(x=p[0], y=p[1], z=p[2]) for p in r['path']]
            ma.markers.append(s)
        return ma

    for i, r in enumerate(pl['runs']):
        Q = np.array(r['q'])
        Q = np.vstack([Q, Q[-2]])                        # descend then retract
        node.get_logger().info(f"run {i}: trial {r['trial']} {r['source']} (pred. error {r['pred_err_abs_mm']:.1f} mm)")
        for a, b in zip(Q[:-1], Q[1:]):
            for s in np.linspace(0, 1, max(2, int(np.degrees(np.abs(b - a).max()) / 1.0))):
                js = JointState(name=JOINTS, position=list(a + s * (b - a)))
                js.header.stamp = node.get_clock().now().to_msg()
                pj.publish(js)
                pm.publish(markers())
                time.sleep(0.02 / args.speedup)
        time.sleep(0.5)
    rclpy.try_shutdown()


# ---------------------------------------------------------------------------- run (robot)
def descend(rc, rr, z_target, v=0.010, f_max=15.0, timeout=60.0):   # long: the speed slider scales v
    """Guarded insertion: a bounded straight-down move to the target depth (inserted), stopped
    early if the contact force rises by f_max (blocked) or the RTDE stream stalls."""
    rc.zeroFtSensor()
    time.sleep(0.3)
    f0 = np.array(fresh(rr).getActualTCPForce()[:3])
    p = list(rr.getActualTCPPose())
    rc.moveL(p[:2] + [z_target] + p[3:], v, 0.25, True)        # bounded: never goes below z_target
    t0, peak, outcome = time.time(), 0.0, 'timeout'
    try:
        while time.time() - t0 < timeout:
            try:
                fresh(rr, 0.1)
            except RuntimeError:
                outcome = 'stream lost'
                break
            z = rr.getActualTCPPose()[2]
            f = float(np.linalg.norm(np.array(rr.getActualTCPForce()[:3]) - f0))
            peak = max(peak, f)
            if z <= z_target + 0.0005:
                outcome = 'inserted'
                break
            if f > f_max:
                outcome = 'blocked'
                break
            if rr.isProtectiveStopped():
                outcome = 'protective stop'
                break
    finally:
        rc.stopL(0.5)
    return dict(outcome=outcome, peak_force_N=peak, tcp_final=list(rr.getActualTCPPose()))


def wait_move(rc, rr, target, timeout=60.0):
    """Poll an asynchronous move so Ctrl+C works; stop the arm on Ctrl+C, a safety stop or a timeout."""
    t0, t_slow = time.time(), None
    try:
        while True:
            time.sleep(0.01)
            fresh(rr, 0.5)
            if rr.isProtectiveStopped() or rr.isEmergencyStopped() or rr.getSafetyMode() not in (1, 2):
                raise RuntimeError('robot stopped (protective / emergency stop) - clear it on the pendant')
            d = np.linalg.norm(np.array(rr.getActualTCPPose()[:3]) - target)
            v = np.linalg.norm(rr.getActualTCPSpeed()[:3])
            if d < 0.001 and v < 0.002:
                return
            # the arm slows almost to rest at sharp corners of a path, so only a sustained stop counts
            t_slow = (t_slow or time.time()) if v < 0.002 else None
            if t_slow and time.time() - t_slow > 1.5:
                raise RuntimeError(f'arm stopped {d * 1000:.0f} mm short of its target')
            if time.time() - t0 > timeout:
                raise RuntimeError('move timed out')
    except BaseException:
        rc.stopL(1.0)
        raise


def run(args):
    import rtde_control
    pl = json.load(open(os.path.join(OUT, 'plan.json')))
    if pl['simulated']:
        sys.exit('plan.json was made from a simulated touch file - run touch and plan on the robot first')
    sel = range(len(pl['runs']))
    if args.runs:
        a, _, b = args.runs.partition('-')
        sel = range(int(a), int(b or a) + 1)
    ori = pl['orientation']
    rr = receiver(args.ip)
    rc = rtde_control.RTDEControlInterface(args.ip)
    tcp = rc.getTCPOffset()
    print(f'TCP offset on the controller: {np.round(tcp, 4)} (plan used tcp_z = {pl["tcp_z"]})')
    q, p = np.array(fresh(rr).getActualQ()), np.array(rr.getActualTCPPose())
    print(f'model check: FK vs controller TCP {np.linalg.norm(fk(q, tcp[2])[:3, 3] - p[:3]) * 1000:.1f} mm')
    touched = json.load(open(touch_file()))
    if 'tcp_offset' in touched and np.linalg.norm(p[:3] - fk(q)[:3, 3] - touched['tcp_offset']) > 0.010:
        sys.exit('active TCP differs from the one used when touching - restore it (or re-touch and re-plan)')
    pose = lambda xyz: [float(v) for v in xyz] + list(ori)
    results_path = os.path.join(OUT, 'results.json')
    results = json.load(open(results_path)) if os.path.exists(results_path) else []
    for i in sel:
        r = pl['runs'][i]
        P = np.array(r['path'])
        P[:, 2] += args.lift
        bad = [j for j, x in enumerate(P) if not rc.getInverseKinematicsHasSolution(pose(x))]
        if not rc.getInverseKinematicsHasSolution(pose([P[0][0], P[0][1], pl['z_app'] + 0.06 + args.lift])):
            bad.append('safe-height start')
        if bad:
            print(f'run {i}: waypoints {bad} unreachable - skipped')
            continue
        input(f"\nrun {i}: trial {r['trial']} from {r['source'].upper()} (predicted error {r['pred_err_abs_mm']:.1f} mm). "
              f"Peg seated as at touch (not slipped or tilted), area clear? ENTER (Ctrl+C aborts) ")
        if not rc.isConnected():
            rc.reconnect()
        # vertical to the transport height, across, down to the path start: linear moves keep the
        # arm configuration, so the whole sequence is checked on the model from the actual joints
        safe = pl['z_app'] + SAFE_UP + args.lift
        cur = fresh(rr).getActualTCPPose()
        legs = [[cur[0], cur[1], safe], [P[0][0], P[0][1], safe], *P]
        if not args.lift:
            legs.append([P[-1][0], P[-1][1], pl['z_target']])
        _, err, el, wr = ik_path(np.array(legs), ori, rr.getActualQ(), pl['tcp_z'])
        if err > 0.001 or min(el, wr) < 15:
            print(f'run {i}: model check failed from the current arm pose (IK error {err * 1000:.1f} mm, '
                  f'singularity margin {min(el, wr):.0f} deg) - skipped. Bring the arm closer to the board.')
            continue
        go = lambda xyz, v: (rc.moveL(pose(xyz), v, 0.3, True), wait_move(rc, rr, np.array(xyz, float)))
        k = args.speed / 0.05                          # approach/retract scale with --speed; descent stays 10 mm/s
        go(legs[0], 0.05 * k)
        go(legs[1], 0.08 * k)
        go(P[0], 0.03 * k)
        # blend radius below half of both neighbouring segments, else the controller halts at the blend
        keep = [P[0]]
        for x in P[1:-1]:
            if np.linalg.norm(x - keep[-1]) >= 0.005:
                keep.append(x)
        if len(keep) > 1 and np.linalg.norm(P[-1] - keep[-1]) < 0.005:
            keep.pop()                                   # the final point replaces a kept point < 5 mm before it
        P = np.array(keep + [P[-1]])
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        blend = [0.0] + [min(0.005, 0.4 * min(a, b)) for a, b in zip(seg[:-1], seg[1:])] + [0.0]
        rc.moveL([pose(x) + [args.speed, 0.3, r_b] for x, r_b in zip(P, blend)], True)
        wait_move(rc, rr, P[-1])
        if args.lift:
            res = dict(outcome='dry run')
        else:
            res = descend(rc, rr, pl['z_target'])
        go(P[-1] + [0, 0, 0.03], 0.03 * k)
        res.update(speed=args.speed, run=i, trial=r['trial'], source=r['source'], offset_mode=pl['offset_mode'], pred_err_mm=r['pred_err_mm'],
                   commanded_xy=r['insert_xy'], hole=pl['hole'], lift=args.lift, time=time.ctime())
        print(f"  -> {res['outcome']}" + (f", peak force {res['peak_force_N']:.1f} N" if 'peak_force_N' in res else ''))
        if not args.lift and not args.nosave:
            res['note'] = input('  observation (e.g. "rim 6 mm, left"; ENTER to skip): ').strip()
            results.append(res)
            json.dump(results, open(results_path, 'w'), indent=1)
    rc.stopScript()


# ---------------------------------------------------------------------------- report
def report(_):
    """Results table (docs/report/results/robot.md) and Figure 10 from data/robot/results.json."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats
    res = json.load(open(os.path.join(OUT, 'results.json')))
    err = lambda o: float(np.hypot(*o['pred_err_mm']))
    lines = ['| offset mode | source | inserted | predicted error at the hole (mm): mean, range | blocked: peak force (N) |',
             '|---|---|---|---|---|']
    for mode in ('mean', 'oracle'):
        for src in ('vicon', 'realsense'):
            rs = [o for o in res if o['offset_mode'] == mode and o['source'] == src]
            e = [err(o) for o in rs]
            fb = [o['peak_force_N'] for o in rs if o['outcome'] == 'blocked']
            lines.append(f"| {mode} | {src} | {sum(o['outcome'] == 'inserted' for o in rs)}/{len(rs)} | "
                         f"{np.mean(e):.1f} ({min(e):.1f}–{max(e):.1f}) | "
                         f"{f'{min(fb):.1f}–{max(fb):.1f}' if fb else '–'} |")
        x = {(o['trial'], o['source']): o['outcome'] == 'inserted' for o in res if o['offset_mode'] == mode}
        a = sum(x[k, 'vicon'] and not x[k, 'realsense'] for k in range(1, 11))
        b = sum(x[k, 'realsense'] and not x[k, 'vicon'] for k in range(1, 11))
        lines.append(f"\n{mode}: discordant pairs Vicon-only {a}, RealSense-only {b}, exact McNemar p = "
                     f"{stats.binomtest(a, a + b, 0.5).pvalue if a + b else 1:.3g}\n")
    ins = [err(o) for o in res if o['outcome'] == 'inserted']
    blk = [err(o) for o in res if o['outcome'] == 'blocked']
    agree = sum((o['outcome'] == 'inserted') == (err(o) < CLEARANCE * 1000) for o in res)
    lines.append(f"inserted: predicted error <= {max(ins):.1f} mm; blocked: >= {min(blk):.1f} mm; "
                 f"prediction (< {CLEARANCE * 1000:.1f} mm) agreed in {agree}/{len(res)} runs; "
                 f"inserts with rim contact (> 2 N): {sum(o['outcome'] == 'inserted' and o['peak_force_N'] > 2 for o in res)}\n")
    path = os.path.join(A.RES, 'robot.md')
    open(path, 'w').write('# Robot replay (generated by scripts/robot_replay.py report)\n\n' + '\n'.join(lines))
    print('\n'.join(lines))

    col = {'vicon': '#2a78d6', 'realsense': '#eb6834'}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.4), sharey=True)
    for ax, mode, title in zip(axes, ('mean', 'oracle'),
                               ('(a) wrist-to-peg offset from the other trials', '(b) offset from the same trial (sensor error only)')):
        ax.add_patch(plt.Circle((0, 0), 11.75, fc='#f3f2ee', ec=A.INK2, lw=0.8, zorder=0))
        ax.add_patch(plt.Circle((0, 0), CLEARANCE * 1000, fill=False, ec=A.INK, lw=1.0, ls=(0, (3, 2)), zorder=1))
        for src, name in (('vicon', 'Vicon'), ('realsense', 'RealSense')):
            rs = [o for o in res if o['offset_mode'] == mode and o['source'] == src]
            for ok, mk in ((True, 'o'), (False, 'X')):
                P = np.array([o['pred_err_mm'] for o in rs if (o['outcome'] == 'inserted') == ok]).reshape(-1, 2)
                ax.scatter(-P[:, 0], -P[:, 1], s=46, marker=mk, color=col[src], ec='white', lw=0.8, zorder=3,
                           label=f"{name}, {'inserted' if ok else 'blocked'}")
        ax.set_aspect('equal')
        ax.set_xlim(-16, 16)
        ax.set_ylim(-16, 16)
        ax.set_title(title, fontsize=9, color=A.INK, loc='left')
        ax.set_xlabel('operator right (mm)', fontsize=8.5, color=A.INK2)
        A.style(ax)
        ax.xaxis.grid(True, color=A.GRID, lw=0.6)
    axes[0].set_ylabel('toward the far side of the board (mm)', fontsize=8.5, color=A.INK2)
    axes[0].text(0, -9.6, 'clearance\n5.5 mm', ha='center', fontsize=7, color=A.INK2)
    axes[1].legend(frameon=False, fontsize=7.5, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(A.FIG, 'fig11_robot_replay.png'), dpi=220)
    print('saved', os.path.join(A.FIG, 'fig11_robot_replay.png'))


# ---------------------------------------------------------------------------- self-test
def selftest():
    T = fk(np.zeros(6))
    assert np.allclose(T[:3, 3], [-0.45675, -0.22315, 0.0665], atol=1e-4), T[:3, 3]
    q_true = np.array([0.3, -1.2, 1.5, -1.9, -1.57, 0.4])
    T = fk(q_true, 0.2)
    q, ep, eo = ik_first(T[:3, 3], Rot.from_matrix(T[:3, :3]).as_rotvec(), 0.2)
    assert ep < 1e-6 and eo < 1e-5, (ep, eo)
    P = np.array([[0, 0, 0], [0.005, 0, 0], [0.02, 0, 0], [0.021, 0, 0]])
    assert len(downsample(P)) == 3
    print('selftest PASS')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    t = sp.add_parser('touch')
    t.add_argument('--ip', default='192.168.0.194')
    t.add_argument('--sim', action='store_true')
    p = sp.add_parser('plan')
    p.add_argument('--peg-below', type=float, default=0.0,
                   help='peg tip below the touched point (m); 0 when the touches were made with the peg tip')
    p.add_argument('--tcp-z', type=float, default=0.20, help='TCP offset along tool z (m), from the pendant')
    p.add_argument('--depth', type=float, default=0.015, help='insertion depth that counts as inserted (m)')
    p.add_argument('--trials', default='1,2,3,4,5,6,7,8,9,10')
    p.add_argument('--offset', choices=['mean', 'oracle'], default='mean',
                   help='mean: wrist->peg offset averaged over the other trials (realistic); oracle: taken from '
                        'this trial\'s Vicon insertion, so only the sensor error remains')
    s = sp.add_parser('sim')
    s.add_argument('--speedup', type=float, default=2.0)
    r = sp.add_parser('run')
    r.add_argument('--ip', default='192.168.0.194')
    r.add_argument('--lift', type=float, default=0.0, help='dry run: raise everything by this (m), no descent')
    r.add_argument('--speed', type=float, default=0.05, help='transport speed (m/s)')
    r.add_argument('--runs', help='run indices from the plan table, e.g. 0-3')
    r.add_argument('--nosave', action='store_true', help='repeat runs (e.g. for video) without adding them to results.json')
    sp.add_parser('selftest')
    sp.add_parser('report')
    a = ap.parse_args()
    {'touch': touch, 'plan': plan, 'sim': sim, 'run': run, 'report': report, 'selftest': lambda _: selftest()}[a.cmd](a)
