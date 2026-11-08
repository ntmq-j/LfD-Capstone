#!/usr/bin/env python3
"""Final analysis for the report (Section 3): accuracy (Metric 1), robustness
(Metric 2), registration comparison, Kalman / gap evaluation, figures 5-9.

Registration (Method 2.7):
  corner - camera->Vicon (R, t) from the four board-corner markers seen by both
           systems (session 2 only; nearest board bag in time). Lever arm d from the
           wrist band to MediaPipe landmark 0 fitted leave-one-trial-out on the other
           task trials of the same band attachment. One RealSense lag per session,
           estimated on the calibration takes.
  wrist  - R, t, d fitted jointly on the nearest calibration take of the same Vicon
           epoch (both sessions). Kept for comparison: it can absorb tracker error.

Outputs: docs/report/results/{paired_trials,mmr_trials}.csv, tables.md
         docs/report/figures/fig5..fig9 *.png
Usage:   python analyze_results.py [--selftest]
Needs ROS 2 and ~/vicon_ws sourced (bag reading, vicon_receiver messages).
"""
import glob
import os
import re
import sys

import numpy as np
import yaml
from scipy import stats
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare_trajectories import kabsch  # noqa: E402
from fit_vicon_calib import load, pair, fit  # noqa: E402

ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
S1, S2 = os.path.join(DATA, 'vicon_session1'), os.path.join(DATA, 'vicon_session2')
RES = os.path.join(ROOT, 'docs', 'report', 'results')
FIG = os.path.join(ROOT, 'docs', 'report', 'figures')

TOL = {'task1': 5.0, 'task2': 3.0, 'task3': 1.0}   # mm, proposal Table 2
GAP = 0.5                                          # s: Kalman bridging limit = capture criterion
SPHERE_R = 0.00695                                 # m, 13.9 mm board-corner markers
# Expected sphere pixels for board1..4 (near-right, far-right, far-left, near-left) at the
# session-2 camera pose; the nearest Hough circle within 40 px is taken.
PIX = [(132, 140), (75, 446), (591, 436), (517, 135)]
# Found with check_bag_overlay.py; every other Task 3 condC bag (trials 7-20) was checked clean.
WRONG_HAND_MMR = {'task3_condC_trial6'}
# Session 1 vicon_client start times (log names, host clock). The driver was restarted after
# each Vicon recalibration, and each recalibration redefines the Vicon world frame.
S1_EPOCHS = [1793063541.2, 1793067886.8, 1793071674.4]

COND_COLOR = {'condA': '#2a78d6', 'condB': '#eb6834', 'condC': '#1baf7a'}   # validated categorical slots 1-3
COND_NAME = {'condA': 'A standard', 'condB': 'B reduced light', 'condC': 'C occlusion'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e6e5e1'
TASKS = ['task1', 'task2', 'task3']
TASK_NAME = {'task1': 'Task 1 peg-in-hole', 'task2': 'Task 2 bolt', 'task3': 'Task 3 RJ45'}
PHASES = ['hold', 'transit', 'pick', 'place']
PLACE_R = 0.05                                     # m: 'place' = slow samples within this of the block anchor


# --------------------------------------------------------------------------- helpers
def trials(base, pattern=r'task\d_cond[ABC]_trial\d+'):
    names = {os.path.basename(f)[:-4] for f in glob.glob(f'{base}/trajectories/realsense/*.npy')}
    names = [n for n in names if re.fullmatch(pattern, n)]
    return sorted(names, key=lambda n: (n[:5], n[6:11], int(n.split('trial')[1])))


def interp(D, times, max_gap=0.05):
    """Vicon cluster position and orientation at `times`; ok = inside the record and
    not bridging a Vicon dropout."""
    vt = D['vt']
    i = np.clip(np.searchsorted(vt, times), 1, len(vt) - 1)
    ok = (times >= vt[0]) & (times <= vt[-1]) & ((vt[i] - vt[i - 1]) <= max_gap)
    Q = np.column_stack([np.interp(times, vt, D['v'][:, k]) for k in range(3)])
    return Q, Rot.from_quat(D['q'][i]).as_matrix(), ok


def lever_arm(r, Rv):
    """Least-squares d for r = Rv d (Rv orthonormal): the mean of Rv^T r."""
    return np.einsum('nji,nj->i', Rv, r) / len(r)


def motion_window(x, t, pad=0.3):
    v = np.r_[0, np.linalg.norm(np.diff(x, axis=0), axis=1) / np.maximum(np.diff(t), 1e-3)]
    mov = np.where(np.convolve(v, np.ones(7) / 7, 'same') > 0.08)[0]
    return (t[mov[0]] - pad, t[mov[-1]] + pad) if len(mov) else (t[0], t[-1])


def capture(x, t, rr, rest=0.02):
    """Metric 2: longest gap between real detections during the demonstration, i.e. from
    the last sample at START (< 20 mm) before the first departure to the first sample back
    at START after the last far sample. Gaps while the hand rests at START are ignored."""
    d = np.linalg.norm(x - x[:10].mean(0), axis=1)
    far = np.where(d > rest)[0]
    if not len(far):
        return True, 0.0
    a = t[max(far[0] - 1, 0)]
    b = t[min(far[-1] + 1, len(t) - 1)]
    g = np.diff(rr)
    inwin = (rr[1:] > a) & (rr[:-1] < b)
    gmax = float(g[inwin].max()) if inwin.any() else 0.0
    return gmax <= GAP, gmax


def phases(L, ts, board=None):
    """hold (< 20 mm from START), target (beyond 60 % of the reach and < 0.08 m/s),
    transit (rest), from the Vicon landmark path L. With the board frame (origin, ex, ey), target
    splits into place (within PLACE_R in the board plane of the slowest moment in the zone
    farthest toward the block, i.e. where the part is inserted or fastened) and pick (the rest)."""
    d = np.linalg.norm(L - L[:10].mean(0), axis=1)
    v = np.r_[0, np.linalg.norm(np.diff(L, axis=0), axis=1) / np.maximum(np.diff(ts), 1e-3)]
    v = np.convolve(v, np.ones(5) / 5, 'same')
    lab = np.full(len(L), 'transit', dtype=object)
    lab[d < 0.02] = 'hold'
    tgt = (d > 0.6 * d.max()) & (v < 0.08)
    lab[tgt] = 'target'
    if board is not None:
        o, ex, ey = board
        b = np.c_[(L - o) @ ex, (L - o) @ ey]
        far = np.where(b[:, 1] >= b[:, 1].max() - 0.015)[0]
        near = np.linalg.norm(b - b[far[np.argmin(v[far])]], axis=1) < PLACE_R
        lab[tgt & near], lab[tgt & ~near] = 'place', 'pick'
    return lab


def board_axes(V):
    """Board frame from the Vicon corner markers (near_right, far_right, far_left, near_left):
    origin near_left, ex toward near_right, ey toward the far (block / camera) side."""
    ex = (V[0] - V[3]) / np.linalg.norm(V[0] - V[3])
    ey = V[2] - V[3] - (V[2] - V[3]) @ ex * ex
    return V[3], ex, ey / np.linalg.norm(ey)


# --------------------------------------------------------------------------- registration
def wrist_registration(calibdir):
    D = load(calibdir)
    lags = np.arange(-0.6, -0.15, 0.01)
    lag = lags[int(np.argmin([np.median(fit(*pair(D, l), iters=3)[3]) for l in lags]))]
    R, t, d, e = fit(*pair(D, lag))
    return dict(name=os.path.basename(calibdir), R=R, t=t, d=d, lag=float(lag), t0=float(D['vt'][0]),
                med=float(np.median(e)))


def corner_registration(bag, ref=None):
    """Camera->Vicon from the board-corner spheres (Hough + aligned depth, radius-corrected)
    and the Vicon corner markers. Cached next to the bag."""
    cache = bag + '_corner.npz'
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    import cv2
    from cv_bridge import CvBridge
    from rclpy.serialization import deserialize_message
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from sensor_msgs.msg import Image, CameraInfo
    from vicon_receiver.msg import Position
    rd = SequentialReader()
    rd.open(StorageOptions(uri=bag, storage_id='sqlite3'), ConverterOptions('cdr', 'cdr'))
    br, cols, deps, K, mk = CvBridge(), [], [], None, {}
    while rd.has_next():
        topic, data, _ = rd.read_next()
        if topic == '/camera/camera/color/image_raw' and len(cols) < 30:
            cols.append(cv2.cvtColor(br.imgmsg_to_cv2(deserialize_message(data, Image), 'rgb8'), cv2.COLOR_RGB2GRAY))
        elif topic == '/camera/camera/aligned_depth_to_color/image_raw' and len(deps) < 60:
            deps.append(br.imgmsg_to_cv2(deserialize_message(data, Image), 'passthrough').astype(float))
        elif topic == '/camera/camera/color/camera_info' and K is None:
            K = np.array(deserialize_message(data, CameraInfo).k).reshape(3, 3)
        elif '/marker_board' in topic:
            m = deserialize_message(data, Position)
            if m.x_trans or m.y_trans or m.z_trans:
                mk.setdefault(int(topic[-1]), []).append((m.x_trans, m.y_trans, m.z_trans))
    del rd
    pts = {k: np.array(v) / 1000 for k, v in mk.items()}
    if ref is None:   # first bag: Tracker labels were stable, use them
        V = np.array([np.median(pts[k], 0) for k in (1, 2, 3, 4)])
        assert max(np.linalg.norm(pts[k] - V[k - 1], axis=1).max() for k in (1, 2, 3, 4)) < 0.005, 'labels flipped'
    else:             # Tracker relabels the symmetric rectangle: assign by position
        allp = np.vstack(list(pts.values()))
        V = np.array([np.median(allp[np.linalg.norm(allp - c, axis=1) < 0.05], 0) for c in ref])
    g = cv2.GaussianBlur(np.median(cols, 0).astype(np.uint8), (5, 5), 1)   # median: robust to a passing hand
    circ = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=30, param1=80, param2=14,
                            minRadius=5, maxRadius=14)[0]
    Dd = np.array(deps)
    Dd[Dd == 0] = np.nan
    dep = np.nanmedian(Dd, 0)
    C = []
    for u0, v0 in PIX:
        c = min((x for x in circ if abs(x[0] - u0) < 40 and abs(x[1] - v0) < 40),
                key=lambda x: abs(x[0] - u0) + abs(x[1] - v0))
        u, v = int(round(c[0])), int(round(c[1]))
        ray = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
        C.append(ray * np.nanmedian(dep[v - 3:v + 4, u - 3:u + 4]) / 1000 + ray / np.linalg.norm(ray) * SPHERE_R)
    C = np.array(C)
    R, t = kabsch(C, V)
    res = np.linalg.norm(C @ R.T + t - V, axis=1)
    # gross-error guard only (a wrong circle or mirrored labelling gives tens of mm)
    assert np.sqrt(np.mean(res ** 2)) < 0.010, f'corner fit residual {res}'
    n = np.cross(V[1] - V[0], V[3] - V[0])
    n /= np.linalg.norm(n)
    meta = yaml.safe_load(open(f'{bag}/metadata.yaml'))['rosbag2_bagfile_information']
    out = dict(R=R, t=t, V=V, C=C, res=res, height=abs((t - V.mean(0)) @ n),
               t0=meta['starting_time']['nanoseconds_since_epoch'] / 1e9)
    np.savez(cache, **out)
    return out


# --------------------------------------------------------------------------- per-trial evaluation
def samples(D, lag):
    """Registered-to-be inputs at the measured and at the Kalman-predicted samples."""
    m = D['m']
    Qm, Rm, okm = interp(D, D['rt'][m] + lag)
    Qp, Rp, okp = interp(D, D['rt'][~m] + lag)
    return dict(P=D['r'][m][okm], Q=Qm[okm], Rv=Rm[okm], ts=D['rt'][m][okm] + lag,
                Pp=D['r'][~m][okp], Qp=Qp[okp], Rvp=Rp[okp])


def evaluate(S, R, t, d, board=None):
    L = S['Q'] + np.einsum('nij,j->ni', S['Rv'], d)
    e = np.linalg.norm(S['P'] @ R.T + t - L, axis=1) * 1000
    q = Rot.from_matrix(S['Rv'])
    ang = np.degrees((q * q[0].inv()).magnitude())
    ep = (np.linalg.norm(S['Pp'] @ R.T + t - S['Qp'] - np.einsum('nij,j->ni', S['Rvp'], d), axis=1) * 1000
          if len(S['Pp']) else np.zeros(0))
    return e, phases(L, S['ts'], board), ang, ep


def long_error_runs(e, ts, mask=None, thr=40.0, dur=0.3):
    """Stretches of >= dur s with error above thr mm. With mask, only samples where mask
    is true count (the wrong-hand check uses the at-target zone, where the left hand is)."""
    big_all = e > thr if mask is None else (e > thr) & mask
    runs, s = 0, None
    for j, big in enumerate(np.r_[big_all, False]):
        if big and s is None:
            s = j
        elif not big and s is not None:
            runs += ts[j - 1] - ts[s] >= dur
            s = None
    return int(runs)


def gap_bridges(D, S, R, t, d):
    """Task 3 condC: error of a straight-line bridge (and of holding the last value) over
    detection gaps > GAP, against the Vicon landmark at the Vicon samples inside the gap."""
    out = []
    P = S['P'] @ R.T + t
    ts = S['ts']
    for k in np.where(np.diff(ts) > GAP)[0]:
        a, b = ts[k], ts[k + 1]
        tv = D['vt'][(D['vt'] > a) & (D['vt'] < b)]
        Q, Rv, ok = interp(D, tv)
        L = Q[ok] + np.einsum('nij,j->ni', Rv[ok], d)
        w = ((tv[ok] - a) / (b - a))[:, None]
        out.append(dict(dur=b - a, lin=np.linalg.norm((1 - w) * P[k] + w * P[k + 1] - L, axis=1) * 1000,
                        hold=np.linalg.norm(P[k] - L, axis=1) * 1000,
                        moved=np.ptp(L, 0).max() * 1000))
    return out


def run_paired():
    # ---- corner registration (session 2) and calibration takes
    b0 = corner_registration(f'{S2}/bags/board')
    bags = [b0] + [corner_registration(f'{S2}/bags/{n}', ref=b0['V']) for n in ('board_end', 'board_end2')]
    near_bag = lambda t0: min(bags, key=lambda b: abs(b['t0'] - t0))
    calib = {s: [wrist_registration(c) for c in sorted(glob.glob(f'{base}/calib*'))] for s, base in (('S1', S1), ('S2', S2))}
    t_attach2 = next(c['t0'] for c in calib['S2'] if c['name'] == 'calib5')     # band re-attached before calib5
    s1_epoch = lambda t0: int(np.searchsorted(S1_EPOCHS, t0))
    s2_attach = lambda t0: 1 + (t0 >= t_attach2)

    # session-2 lag: per calib take, lever arm fitted at each candidate lag, corner (R, t) fixed
    lags = np.arange(-0.6, -0.15, 0.01)
    take_lags = []
    for c in sorted(glob.glob(f'{S2}/calib*')):
        D = load(c)
        b = near_bag(D['vt'][0])
        med = []
        for lag in lags:
            P, Q, Rv = pair(D, lag)
            r = P @ b['R'].T + b['t'] - Q
            med.append(np.median(np.linalg.norm(r - np.einsum('nij,j->ni', Rv, lever_arm(r, Rv)), axis=1)))
        take_lags.append(lags[int(np.argmin(med))])
    lag2 = float(np.median(take_lags))

    # ---- trials
    groups = [('S1', S1, None), ('S2', S2, None), ('S2', f'{S2}/marker_on', 'marker_on')]
    recs = []
    for sess, base, tag in groups:
        for n in trials(base):
            D = load(base, n)
            t0 = float(D['vt'][0])
            task, cond, k = n.split('_')[0], n.split('_')[1], int(n.split('trial')[1])
            rr = np.load(f'{base}/trajectories/realsense/{n}_raw_timestamps.npy') * 1.0
            cap, gmax = capture(D['r'], D['rt'], rr)
            if sess == 'S1':
                cands = [c for c in calib['S1'] if s1_epoch(c['t0']) == s1_epoch(t0)]
            else:
                cands = [c for c in calib['S2'] if s2_attach(c['t0']) == s2_attach(t0)]
            wr = min(cands, key=lambda c: abs(c['t0'] - t0))
            rec = dict(session=sess, group=tag or cond, task=task, cond=cond, trial=k, name=n, t0=t0, D=D,
                       captured=cap, max_gap=gmax, wrist=wr, S_wrist=samples(D, wr['lag']),
                       det_pct=100 * len(rr) / len(D['fr']), proc_hz=(len(D['fr']) - 1) / (D['fr'][-1] - D['fr'][0]),
                       vicon_hz=(len(D['vt']) - 1) / (D['vt'][-1] - D['vt'][0]),
                       vicon_gap_pct=100 * np.diff(D['vt'])[np.diff(D['vt']) > 0.05].sum() / (D['vt'][-1] - D['vt'][0]))
            if sess == 'S2':
                b = near_bag(t0)
                S = samples(D, lag2)
                r = S['P'] @ b['R'].T + b['t'] - S['Q']
                rec.update(bag=b, attach=s2_attach(t0), S_corner=S,
                           sumRr=np.einsum('nji,nj->i', S['Rv'], r), nr=len(r))
            recs.append(rec)

    # leave-one-trial-out lever arm per attachment (session-2 task trials)
    for a in (1, 2):
        pool = [r for r in recs if r['session'] == 'S2' and r['attach'] == a]
        tot, ntot = sum(r['sumRr'] for r in pool), sum(r['nr'] for r in pool)
        for r in pool:
            r['d_loo'] = (tot - r['sumRr']) / (ntot - r['nr'])

    rows = []
    for r in recs:
        for reg in ('corner', 'wrist'):
            if reg == 'corner' and r['session'] != 'S2':
                continue
            if reg == 'corner':
                R, t, d, S = r['bag']['R'], r['bag']['t'], r['d_loo'], r['S_corner']
            else:
                R, t, d, S = r['wrist']['R'], r['wrist']['t'], r['wrist']['d'], r['S_wrist']
            board = board_axes(r['bag']['V']) if r['session'] == 'S2' else None   # no board markers in S1
            e, ph, ang, ep = evaluate(S, R, t, d, board)
            row = dict(session=r['session'], group=r['group'], task=r['task'], cond=r['cond'], trial=r['trial'],
                       registration=reg, n=len(e), rmse=np.sqrt(np.mean(e ** 2)), median=np.median(e),
                       p95=np.percentile(e, 95), max=e.max(), n_pred=len(ep),
                       pred_median=np.median(ep) if len(ep) else np.nan, pred_max=ep.max() if len(ep) else np.nan,
                       max_rot=ang.max(), captured=r['captured'], max_gap=r['max_gap'], det_pct=r['det_pct'],
                       proc_hz=r['proc_hz'], vicon_hz=r['vicon_hz'], vicon_gap_pct=r['vicon_gap_pct'],
                       calib=r['wrist']['name'] if reg == 'wrist' else 'board', lag=r['wrist']['lag'] if reg == 'wrist' else lag2)
            for p in PHASES:
                row[f'med_{p}'] = np.median(e[ph == p]) if (ph == p).any() else np.nan
                row[f'n_{p}'] = int((ph == p).sum())
            row['err_runs'] = long_error_runs(e, S['ts'])
            L = S['Q'] + np.einsum('nij,j->ni', S['Rv'], d)
            dist = np.linalg.norm(L - L[:10].mean(0), axis=1)
            row['wrong_hand'] = (r['task'] == 'task3' and r['cond'] == 'condC' and
                                 long_error_runs(e, S['ts'], mask=dist > 0.6 * dist.max()) > 0)
            row['_e'], row['_ang'], row['_ep'] = e, ang, ep
            rows.append(row)
            if reg == 'corner' and r['task'] == 'task3' and r['cond'] == 'condC':
                row['_bridges'] = gap_bridges(r['D'], S, R, t, d)
    info = dict(lag2=lag2, take_lags=take_lags, bags=bags, calib=calib,
                d_attach={a: next(r['d_loo'] for r in recs if r.get('attach') == a) for a in (1, 2)})
    return rows, recs, info


def run_mmr():
    rows = []
    base = DATA
    for n in trials(base, r'task\d_cond[AC]_trial\d+'):
        p = f'{base}/trajectories/realsense/{n}'
        x, t = np.load(p + '.npy'), np.load(p + '_timestamps.npy') * 1.0
        m, rr = np.load(p + '_measured.npy'), np.load(p + '_raw_timestamps.npy') * 1.0
        cap, gmax = capture(x, t, rr)
        fr = np.load(p + '_frames.npy') * 1.0 if os.path.exists(p + '_frames.npy') else None
        rows.append(dict(task=n[:5], cond=n[6:11], trial=int(n.split('trial')[1]), captured=cap and n not in WRONG_HAND_MMR,
                         lost=not cap, wrong_hand=n in WRONG_HAND_MMR, max_gap=gmax, pred_pct=100 * (~m).mean(),
                         det_pct=100 * len(rr) / len(fr) if fr is not None else np.nan,
                         proc_hz=(len(fr) - 1) / (fr[-1] - fr[0]) if fr is not None else (len(t) - 1) / (t[-1] - t[0]),
                         dur=t[-1] - t[0]))
    return rows


# --------------------------------------------------------------------------- statistics
def tolerance_test(x, tol):
    """One-sided tests against the tolerance: p_meets (H1: mean < tol, the proposal's
    hypothesis) and p_exceeds (H1: mean > tol). t-test, or Wilcoxon if Shapiro rejects
    normality."""
    x = np.asarray(x)
    normal = stats.shapiro(x).pvalue >= 0.05 if len(x) >= 3 else True
    if normal:
        f, test = (lambda alt: stats.ttest_1samp(x, tol, alternative=alt).pvalue), 't'
    else:
        f, test = (lambda alt: stats.wilcoxon(x - tol, alternative=alt).pvalue), 'Wilcoxon'
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return dict(mean=x.mean(), sd=x.std(ddof=1), ci=(x.mean() - h, x.mean() + h), p_meets=f('less'),
                p_exceeds=f('greater'), test=test, d=(x.mean() - tol) / x.std(ddof=1))


def fisher(a, b):
    """a, b: lists of booleans (captured). Two-sided Fisher exact."""
    return stats.fisher_exact([[sum(a), len(a) - sum(a)], [sum(b), len(b) - sum(b)]]).pvalue


# --------------------------------------------------------------------------- figures
def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=8, width=0.6)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig5_phase(rows, plt):
    """Median error by phase, session 2, corner registration."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    conds = ['condA', 'condB', 'condC']
    w = 0.26
    for ax, task in zip(axes, TASKS):
        for i, c in enumerate(conds):
            rs = [r for r in rows if r['registration'] == 'corner' and r['group'] == c and r['task'] == task]
            for j, p in enumerate(PHASES):
                vals = np.array([r[f'med_{p}'] for r in rs if not np.isnan(r[f'med_{p}'])])
                if not len(vals):
                    continue
                x = j + (i - 1) * w
                ax.bar(x, vals.mean(), w * 0.92, color=COND_COLOR[c], label=COND_NAME[c] if j == 0 else None,
                       zorder=2)
                ax.scatter(np.full(len(vals), x), vals, s=9, color=INK, alpha=0.55, lw=0, zorder=3)
        ax.axhline(TOL[task], color=INK2, lw=1.0, zorder=1, label='task tolerance' if task == 'task1' else None)
        ax.set_title(f'{TASK_NAME[task]} (tolerance {TOL[task]:g} mm)', fontsize=9.5, color=INK, loc='left')
        ax.set_xticks(range(len(PHASES)))
        ax.set_xticklabels(['hold at\nSTART', 'transit', 'pick up\npart', 'insert / fasten\nat block'], fontsize=8)
        style(ax)
    axes[0].set_ylabel('median error per trial (mm)', fontsize=8.5, color=INK2)
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    axes[0].set_ylim(0, 50)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig5_error_by_phase.png', dpi=220)
    plt.close(fig)


def fig6_rotation(rows, calib6, plt):
    bins = np.array([0, 10, 20, 30, 45, 60, 90, 150])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    series = [('task trials (session 2)', np.concatenate([r['_ang'] for r in rows if r['registration'] == 'corner']),
               np.concatenate([r['_e'] for r in rows if r['registration'] == 'corner']), COND_COLOR['condA'])]
    if calib6 is not None:
        series.append(('calibration take with wrist rotations', calib6[0], calib6[1], COND_COLOR['condB']))
    for label, ang, e, col in series:
        cx, med, lo, hi = [], [], [], []
        for a, b in zip(bins[:-1], bins[1:]):
            sel = (ang >= a) & (ang < b)
            if sel.sum() >= 30:
                cx.append((a + b) / 2)
                med.append(np.median(e[sel]))
                lo.append(np.percentile(e[sel], 25))
                hi.append(np.percentile(e[sel], 75))
        ax.fill_between(cx, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(cx, med, color=col, lw=2, marker='o', ms=4, label=label)
    ax.set_xlabel('wrist-band rotation from its orientation at START (deg)', fontsize=8.5, color=INK2)
    ax.set_ylabel('error (mm): median, IQR band', fontsize=8.5, color=INK2)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    style(ax)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig6_error_vs_rotation.png', dpi=220)
    plt.close(fig)


def fig7_capture(mmr, paired, plt):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    panels = [('MMR, RealSense only (n = 20 per bar)', mmr, ['condA', 'condC']),
              ('Motion-capture lab, paired with Vicon', paired, ['condA', 'condB', 'condC'])]
    w = 0.26
    for ax, (title, rows, conds) in zip(axes, panels):
        for i, c in enumerate(conds):
            for j, task in enumerate(TASKS):
                rs = [r for r in rows if r['task'] == task and r['cond'] == c]
                k, n = sum(r['captured'] for r in rs), len(rs)
                x = j + (i - (len(conds) - 1) / 2) * w
                ax.bar(x, 100 * k / n, w * 0.92, color=COND_COLOR[c], label=COND_NAME[c] if j == 0 else None)
                ax.text(x, 100 * k / n + 1.5, f'{k}/{n}', ha='center', va='bottom', fontsize=7, color=INK2)
        ax.set_xticks(range(3))
        ax.set_xticklabels([TASK_NAME[t] for t in TASKS], fontsize=8)
        ax.set_title(title, fontsize=9.5, color=INK, loc='left')
        ax.set_ylim(0, 125)
        ax.set_yticks(range(0, 101, 20))
        style(ax)
        ax.legend(frameon=False, fontsize=7.5, loc='upper center', ncol=len(conds), bbox_to_anchor=(0.5, 1.0))
    axes[0].set_ylabel('trials captured (%)', fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig8_capture_rate.png', dpi=220)
    plt.close(fig)


def fig8_gap(rec, plt):
    """One paired Task 3 condC trial: distance from START, Vicon vs RealSense, with the bridge."""
    D, S, b, d = rec['D'], rec['S_corner'], rec['bag'], rec['d_loo']
    P = S['P'] @ b['R'].T + b['t']
    Lv = D['v'] + np.einsum('nij,j->ni', Rot.from_quat(D['q']).as_matrix(), d)
    t0 = S['ts'][0]
    start = Lv[np.searchsorted(D['vt'], t0):][:10].mean(0)
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    ax.plot(D['vt'] - t0, np.linalg.norm(Lv - start, axis=1) * 1000, color=INK, lw=1.4, label='Vicon (reference)')
    ax.scatter(S['ts'] - t0, np.linalg.norm(P - start, axis=1) * 1000, s=9, color=COND_COLOR['condA'],
               zorder=3, label='RealSense, measured samples')
    for k in np.where(np.diff(S['ts']) > GAP)[0]:
        a, bb = S['ts'][k] - t0, S['ts'][k + 1] - t0
        ax.axvspan(a, bb, color=GRID, lw=0, zorder=0)
        ax.plot([a, bb], [np.linalg.norm(P[k] - start) * 1000, np.linalg.norm(P[k + 1] - start) * 1000],
                color=COND_COLOR['condB'], lw=1.6, label='straight-line bridge over the gap')
        ax.text((a + bb) / 2, 8, f'no detection {bb - a:.1f} s', ha='center', fontsize=7.5, color=INK2)
    ax.set_xlim(0, S['ts'][-1] - t0)
    ax.set_xlabel('time (s)', fontsize=8.5, color=INK2)
    ax.set_ylabel('wrist distance from START (mm)', fontsize=8.5, color=INK2)
    ax.legend(frameon=False, fontsize=7.5, loc='upper right')
    style(ax)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig7_task3_gap.png', dpi=220)
    plt.close(fig)


def fig9_wrong_hand(plt):
    """MMR task3_condC_trial6: the tracker follows the left hand during insertion."""
    import make_report_figures as mrf
    trial = 'task3_condC_trial6'
    p = f'{DATA}/trajectories/realsense/{trial}'
    x, t = np.load(p + '.npy'), np.load(p + '_timestamps.npy') * 1.0
    panels = []
    for rel, label in [(4.4, '(a) right wrist tracked'), (6.6, '(b) switched to the left wrist'),
                       (9.6, '(c) right wrist again')]:
        ts, img, K = mrf.bag_frame(f'{DATA}/bags/{trial}', t[0] + rel - mrf.LAG)
        panels.append(mrf.annotate(img, K, x, t, ts, label))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, im in zip(axes, panels):
        ax.imshow(im[:, :, ::-1])
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig9_wrong_hand.png', dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- report tables
def md_table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)] +
                     ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]) + '\n'


def mean_sd(v):
    v = np.asarray(v, float)
    return f'{v.mean():.1f} ± {v.std(ddof=1):.1f}' if len(v) > 1 else f'{v.mean():.1f}'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(RES, exist_ok=True)
    rows, recs, info = run_paired()
    mmr = run_mmr()
    out = []

    out.append('## Registration and synchronisation\n')
    out.append(md_table(['board bag', 'corner residual RMSE (mm)', 'camera height above board (mm)'],
                        [[n, f"{np.sqrt(np.mean(b['res'] ** 2)) * 1000:.1f}", f"{b['height'] * 1000:.0f}"]
                         for n, b in zip(('board', 'board_end', 'board_end2'), info['bags'])]))
    B = info['bags']
    for k in (1, 2):
        out.append(f"camera pose change board -> {('board_end', 'board_end2')[k - 1]}: "
                   f"{np.degrees(Rot.from_matrix(B[k]['R'] @ B[0]['R'].T).magnitude()):.2f} deg, "
                   f"{np.linalg.norm(B[k]['t'] - B[0]['t']) * 1000:.1f} mm\n")
    out.append(f"\nsession-2 lag (median of calib takes): {info['lag2'] * 1000:.0f} ms; per take "
               f"{[round(l * 1000) for l in info['take_lags']]} ms\n")
    out.append(f"lever arm (LOO, typical) attachment 1: {np.round(info['d_attach'][1] * 1000, 1)} mm, "
               f"attachment 2: {np.round(info['d_attach'][2] * 1000, 1)} mm\n\n")
    out.append(md_table(['session', 'calib take', 'lag (ms)', 'lever arm d (mm)', 'fit median (mm)'],
                        [[s, c['name'], f"{c['lag'] * 1000:.0f}", np.round(c['d'] * 1000, 1).tolist(), f"{c['med'] * 1000:.1f}"]
                         for s in ('S1', 'S2') for c in info['calib'][s]]))

    # ---- data quality
    out.append('\n## Data quality\n')
    q = []
    for sess, grp in [('S1', 'condA'), ('S1', 'condB'), ('S2', 'condA'), ('S2', 'condB'), ('S2', 'condC'), ('S2', 'marker_on')]:
        rs = [r for r in rows if r['session'] == sess and r['group'] == grp and r['registration'] == 'wrist']
        q.append([sess, grp, len(rs), f"{np.mean([r['proc_hz'] for r in rs]):.1f} ({min(r['proc_hz'] for r in rs):.1f}–{max(r['proc_hz'] for r in rs):.1f})",
                  f"{np.mean([r['det_pct'] for r in rs]):.1f}", f"{np.mean([r['vicon_hz'] for r in rs]):.1f}",
                  f"{max(r['vicon_gap_pct'] for r in rs):.1f}"])
    for c in ('condA', 'condC'):
        rs = [r for r in mmr if r['cond'] == c]
        q.append(['MMR', c, len(rs), f"{np.mean([r['proc_hz'] for r in rs]):.1f} ({min(r['proc_hz'] for r in rs):.1f}–{max(r['proc_hz'] for r in rs):.1f})",
                  f"{np.nanmean([r['det_pct'] for r in rs]):.1f}", '–', '–'])
    out.append(md_table(['session', 'group', 'trials', 'tracker rate Hz mean (range)', 'frames with detection %',
                         'Vicon Hz', 'max Vicon gap % of trial'], q))

    # ---- Metric 1
    out.append('\n## Metric 1: accuracy (session 2, corner registration, held-out lever arm)\n')
    m1 = []
    for task in TASKS:
        for g in ('condA', 'condB', 'condC', 'marker_on'):
            rs = [r for r in rows if r['registration'] == 'corner' and r['group'] == g and r['task'] == task]
            if not rs:
                continue
            m1.append([task, g, len(rs), mean_sd([r['rmse'] for r in rs]), mean_sd([r['median'] for r in rs]),
                       mean_sd([r['p95'] for r in rs])] +
                      [mean_sd([r[f'med_{p}'] for r in rs if not np.isnan(r[f'med_{p}'])]) for p in PHASES])
    out.append(md_table(['task', 'group', 'n', 'RMSE', 'median', 'p95', 'hold', 'transit', 'pick', 'place'], m1))

    out.append('\n### Tolerance tests (session 2 condA, per-trial values)\n')
    tt = []
    for task in TASKS:
        rs = [r for r in rows if r['registration'] == 'corner' and r['group'] == 'condA' and r['task'] == task]
        for what, key in (('RMSE', 'rmse'), ('at-place median', 'med_place')):
            v = [r[key] for r in rs if not np.isnan(r[key])]
            s = tolerance_test(v, TOL[task])
            tt.append([task, what, TOL[task], f"{s['mean']:.1f} ± {s['sd']:.1f}", f"{s['ci'][0]:.1f}–{s['ci'][1]:.1f}",
                       s['test'], f"{s['p_meets']:.3g}",
                       f"{s['p_exceeds']:.2g}", f"{s['d']:.1f}"])
    out.append(md_table(['task', 'measure', 'tolerance mm', 'mean ± SD mm', '95% CI of mean', 'test', 'p (mean < tol)',
                         'p (mean > tol)', "Cohen's d"], tt))

    out.append('\n### Condition comparisons on per-trial RMSE (Welch t-test)\n')
    cc = []
    for reg, sess, a, b in [('corner', 'S2', 'condA', 'condB'), ('corner', 'S2', 'condA', 'condC'),
                            ('wrist', 'S1', 'condA', 'condB')]:
        for task in TASKS:
            xa = [r['rmse'] for r in rows if r['registration'] == reg and r['session'] == sess and r['group'] == a and r['task'] == task]
            xb = [r['rmse'] for r in rows if r['registration'] == reg and r['session'] == sess and r['group'] == b and r['task'] == task]
            cc.append([f'{sess} {reg}', f'{a} vs {b}', task, f'{np.mean(xa):.1f} (n={len(xa)})', f'{np.mean(xb):.1f} (n={len(xb)})',
                       f'{stats.ttest_ind(xa, xb, equal_var=False).pvalue:.3g}'])
    out.append(md_table(['data', 'comparison', 'task', 'mean RMSE first', 'mean RMSE second', 'p'], cc))

    out.append('\n### Registration comparison (same session-2 trials)\n')
    rc = []
    for g in ('condA', 'condB', 'condC', 'marker_on'):
        for task in TASKS:
            c = [r for r in rows if r['session'] == 'S2' and r['group'] == g and r['task'] == task and r['registration'] == 'corner']
            w = [r for r in rows if r['session'] == 'S2' and r['group'] == g and r['task'] == task and r['registration'] == 'wrist']
            if c:
                rc.append([g, task, len(c), mean_sd([r['rmse'] for r in c]), mean_sd([r['rmse'] for r in w]),
                           ', '.join(sorted({r['calib'] for r in w}))])
    out.append(md_table(['group', 'task', 'n', 'RMSE corner', 'RMSE wrist-based', 'wrist calib take'], rc))
    b0 = info['bags'][0]
    rd = [[c['name'], f"{np.degrees(Rot.from_matrix(c['R'] @ b0['R'].T).magnitude()):.1f}",
           f"{np.linalg.norm(c['t'] - b0['t']) * 1000:.0f}", np.round(c['d'] * 1000).tolist()]
          for c in info['calib']['S2']]
    out.append('\nWrist-based transforms of the session-2 calibration takes versus the corner registration:\n\n' +
               md_table(['calib take', 'rotation diff (deg)', 'translation diff (mm)', 'its lever arm d (mm)'], rd))
    out.append('\n### Session 1 (wrist-based registration only)\n')
    s1 = []
    for task in TASKS:
        for g in ('condA', 'condB'):
            rs = [r for r in rows if r['session'] == 'S1' and r['group'] == g and r['task'] == task]
            s1.append([task, g, len(rs), mean_sd([r['rmse'] for r in rs]), mean_sd([r['median'] for r in rs])])
    out.append(md_table(['task', 'condition', 'n', 'RMSE', 'median'], s1))

    # ---- Kalman bridges and Task 3 gaps
    out.append('\n## Kalman-predicted samples (paired trials; excluded from Metric 1)\n')
    kp = []
    for reg, sess in (('corner', 'S2'), ('wrist', 'S1')):
        ep = np.concatenate([r['_ep'] for r in rows if r['registration'] == reg and r['session'] == sess])
        em = np.concatenate([r['_e'] for r in rows if r['registration'] == reg and r['session'] == sess])
        kp.append([sess, reg, len(ep), f'{100 * len(ep) / (len(ep) + len(em)):.2f}',
                   f'{np.median(ep):.1f}' if len(ep) else '–', f'{np.percentile(ep, 95):.1f}' if len(ep) else '–',
                   f'{ep.max():.1f}' if len(ep) else '–', f'{np.median(em):.1f}'])
    out.append(md_table(['session', 'registration', 'predicted samples', '% of samples', 'median err', 'p95', 'max',
                         'median err of measured'], kp))
    br = [b for r in rows if '_bridges' in r for b in r['_bridges']]
    if br:
        lin, hold = np.concatenate([b['lin'] for b in br]), np.concatenate([b['hold'] for b in br])
        out.append(f"\nTask 3 condC paired gaps > {GAP} s: {len(br)} gaps, duration "
                   f"{min(b['dur'] for b in br):.1f}–{max(b['dur'] for b in br):.1f} s; true wrist movement during the gap "
                   f"{min(b['moved'] for b in br):.0f}–{max(b['moved'] for b in br):.0f} mm.\n"
                   f"straight-line bridge error: median {np.median(lin):.1f}, p95 {np.percentile(lin, 95):.1f}, max {lin.max():.1f} mm; "
                   f"hold-last-value: median {np.median(hold):.1f}, p95 {np.percentile(hold, 95):.1f} mm\n")

    # ---- Metric 2
    out.append('\n## Metric 2: robustness (captured = no detection gap > 0.5 s in the demonstration window, and on the right hand)\n')
    corner_wrong = {(r['task'], r['cond'], r['trial']) for r in rows
                    if r['registration'] == 'corner' and r['wrong_hand']}
    paired_cap = []
    for r in rows:
        if r['registration'] != 'wrist' or r['group'] == 'marker_on':
            continue
        wrong = r['session'] == 'S2' and (r['task'], r['cond'], r['trial']) in corner_wrong
        paired_cap.append(dict(session=r['session'], task=r['task'], cond=r['cond'], captured=r['captured'] and not wrong,
                               lost=not r['captured'], wrong_hand=wrong, max_gap=r['max_gap']))
    m2 = []
    for lab, rs, conds in (('MMR', mmr, ('condA', 'condC')), ('B2 paired', paired_cap, ('condA', 'condB', 'condC'))):
        for c in conds:
            for task in TASKS:
                x = [r for r in rs if r['task'] == task and r['cond'] == c]
                gaps = [r['max_gap'] for r in x if r['lost']]
                k = sum(r['captured'] for r in x)
                ci = stats.binomtest(k, len(x)).proportion_ci(0.95, 'exact')
                m2.append([lab, c, task, f"{k}/{len(x)}", f"{ci.low * 100:.0f}–{ci.high * 100:.0f}", sum(r['lost'] for r in x),
                           sum(r['wrong_hand'] for r in x),
                           f"{min(gaps):.1f}–{max(gaps):.1f}" if gaps else '–'])
    out.append(md_table(['lab', 'condition', 'task', 'captured', '95% CI (%)', 'lost > 0.5 s', 'wrong hand', 'gap range of lost (s)'], m2))
    fz = []
    s2_cap = [r for r in paired_cap if r['session'] == 'S2']
    for lab, rs, a, b in (('MMR', mmr, 'condA', 'condC'), ('B2', paired_cap, 'condA', 'condB'), ('B2', paired_cap, 'condA', 'condC'),
                          ('B2 session 2 only', s2_cap, 'condA', 'condC')):
        for task in TASKS:
            p = fisher([r['captured'] for r in rs if r['task'] == task and r['cond'] == a],
                       [r['captured'] for r in rs if r['task'] == task and r['cond'] == b])
            fz.append([lab, f'{a} vs {b}', task, f'{p:.3g}', f'{min(1, 3 * p):.3g}'])
    out.append('\nFisher exact (two-sided), Bonferroni x3 tasks:\n\n' +
               md_table(['lab', 'comparison', 'task', 'p', 'p (Bonferroni)'], fz))
    mm = [r for r in mmr]
    win = lambda n: (lambda x, t: np.subtract(*motion_window(x, t, pad=0.0)[::-1]))(
        np.load(f'{DATA}/trajectories/realsense/{n}.npy'), np.load(f'{DATA}/trajectories/realsense/{n}_timestamps.npy'))
    out.append('\nPractice effect (MMR, mean time from first to last movement, s): ' + '; '.join(
        f"{task} A1-15 {np.mean([win(f'{task}_condA_trial{k}') for k in range(1, 16)]):.1f}, "
        f"C {np.mean([win(f'{task}_condC_trial{k}') for k in range(1, 21)]):.1f}, "
        f"A16-20 {np.mean([win(f'{task}_condA_trial{k}') for k in range(16, 21)]):.1f}" for task in TASKS) + '\n')
    out.append(f"\nMMR Kalman-predicted share: mean {np.mean([r['pred_pct'] for r in mm]):.2f} %, max "
               f"{max(r['pred_pct'] for r in mm):.1f} %\n")
    e_runs = [(r['session'], r['group'], r['task'], r['trial'], r['err_runs']) for r in rows
              if r['registration'] == 'corner' and r['err_runs']]
    out.append(f"\nPaired session-2 trials with >= 0.3 s of > 40 mm error (corner): {e_runs}\n")

    # ---- error vs wrist rotation (same bins as fig 6)
    corner_task = [r for r in rows if r['registration'] == 'corner' and r['group'] != 'marker_on']
    ang = np.concatenate([r['_ang'] for r in corner_task])
    err = np.concatenate([r['_e'] for r in corner_task])
    D6 = load(f'{S2}/calib6')
    b6 = min(info['bags'], key=lambda b: abs(b['t0'] - D6['vt'][0]))
    e6, _, a6, _ = evaluate(samples(D6, info['lag2']), b6['R'], b6['t'], info['d_attach'][2])
    info['calib6'] = (a6, e6)
    rb = []
    for a, b in zip([0, 10, 20, 30, 45, 60, 90], [10, 20, 30, 45, 60, 90, 150]):
        sel, sel6 = (ang >= a) & (ang < b), (a6 >= a) & (a6 < b)
        rb.append([f'{a}–{b}', sel.sum(), f'{np.median(err[sel]):.1f}' if sel.sum() >= 30 else '–',
                   sel6.sum(), f'{np.median(e6[sel6]):.1f}' if sel6.sum() >= 30 else '–'])
    out.append('\n## Error versus wrist-band rotation from START (session 2, corner registration)\n' +
               md_table(['rotation (deg)', 'task samples', 'median error task trials (mm)', 'calib6 samples',
                         'median error calib6 (mm)'], rb) +
               f"\nSpearman rho (task trials, rotation vs error): {stats.spearmanr(ang, err)[0]:.2f}\n")

    # ---- marker on / off (Task 1, RealSense metrics only)
    mo = []
    for grp in ('marker_on', 'marker_off'):
        base = f'{S2}/{grp}'
        vals = []
        for n in trials(base):
            p = f'{base}/trajectories/realsense/{n}'
            x, t = np.load(p + '.npy'), np.load(p + '_timestamps.npy') * 1.0
            rr, fr = np.load(p + '_raw_timestamps.npy') * 1.0, np.load(p + '_frames.npy') * 1.0
            raw, rt = np.load(p + '_raw.npy'), rr - rr[0]
            hold = raw[rt < min(1.0, motion_window(x, t)[0] - t[0])]
            vals.append(dict(hz=(len(fr) - 1) / (fr[-1] - fr[0]), det=100 * len(rr) / len(fr),
                             lost=not capture(x, t, rr)[0], start=hold.mean(0) * 1000,
                             jit=np.median(np.linalg.norm(np.diff(hold, axis=0), axis=1)) * 1000))
        st = np.array([v['start'] for v in vals])
        mo.append([grp, len(vals), f"{np.mean([v['hz'] for v in vals]):.1f}", f"{np.mean([v['det'] for v in vals]):.1f}",
                   sum(v['lost'] for v in vals), np.round(st.mean(0)).astype(int).tolist(),
                   f"{np.mean([v['jit'] for v in vals]):.2f}"])
        info[grp] = st.mean(0)
    out.append('\n## Marker check: Task 1 with the wrist band on (paired) vs off (RealSense only)\n' +
               md_table(['group', 'trials', 'tracker Hz', 'frames with detection %', 'lost > 0.5 s',
                         'START position (camera frame, mm)', 'frame-to-frame jitter at START (mm)'], mo) +
               f"\nSTART position difference on vs off: {np.linalg.norm(info['marker_on'] - info['marker_off']):.1f} mm\n")

    open(f'{RES}/tables.md', 'w').write('# Analysis tables (generated by scripts/analyze_results.py)\n\n' + ''.join(out))
    keys = [k for k in rows[0] if not k.startswith('_')]
    with open(f'{RES}/paired_trials.csv', 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join(f'{r[k]:.4g}' if isinstance(r[k], float) else str(r[k]) for k in keys) + '\n')
    with open(f'{RES}/mmr_trials.csv', 'w') as f:
        f.write(','.join(mmr[0]) + '\n')
        for r in mmr:
            f.write(','.join(f'{v:.4g}' if isinstance(v, float) else str(v) for v in r.values()) + '\n')

    # ---- figures
    corner = [r for r in rows if r['registration'] == 'corner']
    fig5_phase(corner, plt)
    fig6_rotation([r for r in corner if r['group'] != 'marker_on'], info['calib6'], plt)
    fig7_capture(mmr, paired_cap, plt)
    t3 = [r for r in recs if r['session'] == 'S2' and r['task'] == 'task3' and r['cond'] == 'condC']
    fig8_gap(t3[0], plt)
    fig9_wrong_hand(plt)
    print(open(f'{RES}/tables.md').read())


def selftest():
    rng = np.random.default_rng(0)
    Rv = Rot.random(500, random_state=1).as_matrix()
    d = np.array([-0.02, -0.008, 0.032])
    r = np.einsum('nij,j->ni', Rv, d) + rng.normal(0, 0.003, (500, 3))
    assert np.allclose(lever_arm(r, Rv), d, atol=0.001)
    t = np.arange(0, 6, 0.05)
    x = np.zeros((len(t), 3))
    x[:, 0] = np.clip((t - 1) * 0.15, 0, 0.3) - np.clip((t - 3.5) * 0.15, 0, 0.3)   # out and back, 1-5.5 s
    rr = t[(t < 2.0) | (t > 2.8)]                                                       # 0.8 s gap mid-task
    assert capture(x, t, rr) == (False, capture(x, t, rr)[1]) and capture(x, t, rr)[1] > 0.75
    rr = t[(t < 0.1) | (t > 0.6)]                                                       # gap ends before the window
    assert capture(x, t, rr)[0]
    L = np.c_[x[:, 0], x[:, 1], x[:, 2]]
    ph = phases(L, t)
    assert ph[0] == 'hold' and (ph[(t > 3.0) & (t < 3.4)] == 'target').all()
    ph = phases(L, t, (np.zeros(3), np.array([0, 1.0, 0]), np.array([1.0, 0, 0])))   # far = +x: the reversal
    assert (ph[(t > 3.0) & (t < 3.4)] == 'place').all()
    print('selftest PASS')


if __name__ == '__main__':
    selftest() if '--selftest' in sys.argv else main()
