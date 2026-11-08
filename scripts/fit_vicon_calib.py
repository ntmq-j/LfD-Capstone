#!/usr/bin/env python3
"""Fit the camera->Vicon transform for one calibration take (free hand motion,
recorded with collect_data.py --vicon --data-dir <session>/calibN).

The Vicon object is a marker cluster on the back of the wrist, not MediaPipe
landmark 0, so the fit includes a body-fixed lever arm d (cluster frame):
    R @ p_realsense + t  =  p_vicon + R_cluster @ d
plus a constant time lag on the RealSense stamps (~-360 ms: the tracker's
image queue is 10 deep and it processes slower than 30 fps).
Every Vicon recalibration redefines the Vicon frame, so each epoch needs its
own take; lever arm and lag carry over within a session.

Usage: python fit_vicon_calib.py data/vicon_session2/calib [--lever-from <calibdir>] [--compare <calibdir>]
Writes <calibdir>/transform_leverarm.npz (R, t, lever_arm, lag).
"""
import argparse
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_trajectories import kabsch


def load(calibdir, name='task1_condA_trial1'):
    d = f'{calibdir}/trajectories'
    return dict(v=np.load(f'{d}/vicon/{name}.npy'), vt=np.load(f'{d}/vicon/{name}_timestamps.npy') * 1.0,
                q=np.load(f'{d}/vicon/{name}_quat.npy'), r=np.load(f'{d}/realsense/{name}.npy'),
                rt=np.load(f'{d}/realsense/{name}_timestamps.npy') * 1.0,
                m=np.load(f'{d}/realsense/{name}_measured.npy'),
                fr=np.load(f'{d}/realsense/{name}_frames.npy') * 1.0)


def pair(D, lag, max_gap=0.05):
    """Vicon position + cluster rotation at each measured RealSense sample."""
    t = D['rt'] + lag
    vt = D['vt']
    i = np.clip(np.searchsorted(vt, t), 1, len(vt) - 1)
    ok = (t >= vt[0]) & (t <= vt[-1]) & D['m'] & ((vt[i] - vt[i - 1]) <= max_gap)
    Q = np.column_stack([np.interp(t, vt, D['v'][:, k]) for k in range(3)])
    Rv = Rot.from_quat(D['q'][np.clip(i, 0, len(D['q']) - 1)]).as_matrix()
    return D['r'][ok], Q[ok], Rv[ok]


def fit(P, Q, Rv, d=None, iters=10):
    """Alternate rigid fit and (optionally) lever-arm least squares, trimming
    the worst 5 % (RealSense depth jumps) each round."""
    est = d is None
    d = np.zeros(3) if est else d
    for _ in range(iters):
        T = Q + np.einsum('nij,j->ni', Rv, d)
        R, t = kabsch(P, T)
        e = np.linalg.norm(P @ R.T + t - T, axis=1)
        k = e < np.percentile(e, 95)
        R, t = kabsch(P[k], T[k])
        if est:
            d = np.linalg.lstsq(Rv[k].reshape(-1, 3), (P @ R.T + t - Q)[k].reshape(-1), rcond=None)[0]
    e = np.linalg.norm(P @ R.T + t - Q - np.einsum('nij,j->ni', Rv, d), axis=1)
    return R, t, d, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('calibdir')
    ap.add_argument('--lever-from', help='reuse the lever arm of an earlier calib take (same session)')
    ap.add_argument('--compare', help='print rotation/translation change vs another calib take')
    a = ap.parse_args()

    D = load(a.calibdir)
    v, vt, fr = D['v'], D['vt'], D['fr']
    print(f"Vicon {(len(v) - 1) / (vt[-1] - vt[0]):.1f} Hz, gaps>50 ms {int((np.diff(vt) > 0.05).sum())}, "
          f"duration {vt[-1] - vt[0]:.0f} s, motion range {np.round(np.ptp(v, 0) * 1000).astype(int)} mm")
    print(f"RealSense {(len(fr) - 1) / (fr[-1] - fr[0]):.1f} Hz, longest stall {np.diff(fr).max():.2f} s, "
          f"{len(D['r'])} samples, {int((~D['m']).sum())} predicted")

    d0 = np.load(f'{a.lever_from}/transform_leverarm.npz')['lever_arm'] if a.lever_from else None
    lags = np.arange(-0.7, 0.0, 0.01)
    med = [np.median(fit(*pair(D, l), d=d0, iters=3)[3]) for l in lags]
    lag = float(lags[int(np.argmin(med))])
    R, t, d, e = fit(*pair(D, lag), d=d0)
    print(f"lag {lag * 1000:.0f} ms | lever arm {np.round(d * 1000, 1)} mm (|d| {np.linalg.norm(d) * 1000:.0f} mm)"
          + (' [reused]' if d0 is not None else ''))
    print(f"fit: median {np.median(e) * 1000:.1f} mm, RMSE {np.sqrt(np.mean(e ** 2)) * 1000:.1f} mm, "
          f"p95 {np.percentile(e, 95) * 1000:.0f} mm, n={len(e)}")
    if a.compare:
        c = np.load(f'{a.compare}/transform_leverarm.npz')
        print(f"vs {a.compare}: rotation {np.degrees(Rot.from_matrix(R @ c['R'].T).magnitude()):.2f} deg, "
              f"translation {np.linalg.norm(t - c['t']) * 1000:.1f} mm")
    np.savez(f'{a.calibdir}/transform_leverarm.npz', R=R, t=t, lever_arm=d, lag=lag)
    print(f"saved {a.calibdir}/transform_leverarm.npz")


if __name__ == '__main__':
    main()
