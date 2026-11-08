#!/usr/bin/env python3
"""
Validate compare_trajectories.py against synthetic data with KNOWN truth,
then run the CLI end-to-end on a real RealSense trial with a fabricated Vicon.

Part 1 - synthetic:
  A smooth wrist path is sampled as "Vicon" (100 Hz, 0.5 mm noise, one
  dropout) and as "RealSense" (~22 Hz jittered, 2/2/4 mm noise, expressed
  in a rotated+translated camera frame, timestamps 60 ms LATE, 12 frames
  marked predicted with bad positions, 25 frames missing). The comparison
  must recover the lag, the rigid transform, and an RMSE equal to the
  injected noise (~5 mm), and must exclude the predicted frames.

Part 2 - real shapes:
  Takes the newest recorded RealSense trial, fabricates a Vicon recording
  from it, and runs the CLI so real gaps/timestamps/sidecars go through
  load_trajectory -> sync -> align -> plot -> report without error.

Run:  venv/bin/python scripts/test_compare_synthetic.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_trajectories import TrajectoryComparison, rotation_angle_deg  # noqa: E402

rng = np.random.default_rng(0)
ok = True


def check(cond, msg):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    ok &= bool(cond)


# =============================================================================
print("=" * 64)
print("PART 1  synthetic trial with known truth")
print("=" * 64)

T = 60.0


def truth(tt):
    # ~0.3 m/s peak so a 60 ms lag is worth ~18 mm - must be visible
    return np.column_stack([0.20 * np.sin(1.5 * tt),
                            0.15 * np.sin(1.1 * tt + 0.4),
                            0.05 * np.cos(0.7 * tt)]) + [1.0, 0.5, 0.9]


# Vicon: 100 Hz, 0.5 mm noise, 0.3 s dropout at t=20
vt = np.arange(0, T, 0.01)
vp = truth(vt) + rng.normal(0, 0.0005, (len(vt), 3))
drop = (vt > 20.0) & (vt < 20.3)
vt, vp = vt[~drop], vp[~drop]

# Camera frame:  vicon = R_true @ cam + t_true
R_true = Rotation.from_euler('xyz', [170, 25, -40], degrees=True).as_matrix()
t_true = np.array([0.8, -0.3, 1.2])

# RealSense: ~22 Hz jittered; stamps are 60 ms LATE (capture + MediaPipe)
TRUE_LAG = -0.060                                   # add to RS stamps -> Vicon clock
rt = np.cumsum(rng.uniform(0.040, 0.050, int(T / 0.045)))
rt = rt[rt < T - 0.5]
cam = (R_true.T @ (truth(rt) - t_true).T).T
cam += rng.normal(0, [0.002, 0.002, 0.004], cam.shape)
rs_ts = rt - TRUE_LAG

# 12 frames "predicted" with poor positions, 25 frames missing entirely
measured = np.ones(len(rt), bool)
measured[300:312] = False
cam[300:312] += rng.normal(0, 0.02, (12, 3))
keep = np.ones(len(rt), bool)
keep[600:625] = False
cam, rs_ts, measured = cam[keep], rs_ts[keep], measured[keep]

expected_rmse = np.sqrt(2 ** 2 + 2 ** 2 + 4 ** 2 + 3 * 0.5 ** 2)   # mm

print("\n[A] full pipeline: lag search + time sync + Kabsch, predicted excluded")
cmp = TrajectoryComparison(vp, cam, vicon_ts=vt, realsense_ts=rs_ts, rs_measured=measured)
r = cmp.calculate_rmse()
rot_err = rotation_angle_deg(R_true.T @ cmp.R)
tr_err = np.linalg.norm(cmp.t_vec - t_true) * 1000
print(f"  RMSE {r['rmse_mm']:.2f} mm (expected ~{expected_rmse:.1f}) | lag {cmp.lag*1000:+.1f} ms "
      f"(true {TRUE_LAG*1000:+.0f}) | rot err {rot_err:.3f} deg | trans err {tr_err:.2f} mm")
check(abs(cmp.lag - TRUE_LAG) < 0.005, "lag recovered within 5 ms")
check(rot_err < 0.5, "rotation recovered within 0.5 deg")
check(tr_err < 3.0, "translation recovered within 3 mm")
check(expected_rmse - 1.0 < r['rmse_mm'] < expected_rmse + 1.0,
      f"RMSE equals injected noise ({expected_rmse:.1f} ± 1 mm)")
check(cmp.info['n_predicted_excluded'] == 12, "12 predicted samples excluded")
check(cmp.info['n_no_vicon'] >= 1, "samples across the Vicon dropout discarded")

cov = cmp.calculate_coverage()
print(f"  coverage measured {cov['coverage_measured_percent']:.1f} %, "
      f"max gap {cov['max_gap_measured_s']*1000:.0f} ms, uninterrupted={cov['uninterrupted']}")
check(cov['n_gaps_with_prediction'] == 1, "the 25-frame hole is detected as one gap > 0.5 s")

print("\n[B] what you would get WITHOUT the fixes (for the thesis discussion)")
r_lag0 = TrajectoryComparison(vp, cam, vicon_ts=vt, realsense_ts=rs_ts, rs_measured=measured,
                              lag=0.0, verbose=False).calculate_rmse()['rmse_mm']
r_incl = TrajectoryComparison(vp, cam, vicon_ts=vt, realsense_ts=rs_ts, rs_measured=None,
                              verbose=False).calculate_rmse()['rmse_mm']
import warnings  # noqa: E402
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    r_idx = TrajectoryComparison(vp, cam, verbose=False).calculate_rmse()['rmse_mm']
print(f"  lag forced to 0 (ignore 60 ms latency) : RMSE {r_lag0:6.2f} mm")
print(f"  predicted frames NOT excluded          : RMSE {r_incl:6.2f} mm")
print(f"  legacy index-based resampling          : RMSE {r_idx:6.2f} mm")
check(r_lag0 > r['rmse_mm'] * 1.5, "ignoring lag inflates RMSE by >50%")
check(r_incl > r['rmse_mm'], "including predicted frames inflates RMSE")

print("\n[C] transform reuse: fit on this trial, apply to a second trial")
with tempfile.TemporaryDirectory() as td:
    tf = Path(td) / 'cam_to_vicon.npz'
    cmp.save_transform(tf)
    from compare_trajectories import load_transform
    R2, t2, lag2 = load_transform(tf)
    # second trial: same rig, different motion
    rt2 = np.cumsum(rng.uniform(0.040, 0.050, 600)) + 5.0
    cam2 = (R_true.T @ (truth(rt2 * 0.7 + 3) - t_true).T).T + rng.normal(0, [0.002, 0.002, 0.004], (len(rt2), 3))
    vt2 = np.arange(4.0, rt2[-1] + 1, 0.01)
    vp2 = truth(vt2 * 0.7 + 3) + rng.normal(0, 0.0005, (len(vt2), 3))
    cmp2 = TrajectoryComparison(vp2, cam2, vicon_ts=vt2, realsense_ts=rt2 - TRUE_LAG,
                                transform=(R2, t2), lag=lag2, verbose=False)
    r2 = cmp2.calculate_rmse()['rmse_mm']
    print(f"  second trial with loaded transform+lag : RMSE {r2:.2f} mm")
    check(r2 < expected_rmse + 1.5, "loaded transform generalises to another trial")

# =============================================================================
print("\n" + "=" * 64)
print("PART 2  CLI end-to-end on a real RealSense trial")
print("=" * 64)

rs_dir = HERE.parent / 'data' / 'trajectories' / 'realsense'
trials = sorted([p for p in rs_dir.glob('trial_*.npy')
                 if not p.stem.endswith(('_timestamps', '_measured', '_raw'))],
                key=lambda p: p.stat().st_mtime)
trials = [p for p in trials if len(np.load(p)) > 100]
if not trials:
    print("  no real trial with >100 samples found - skipping part 2")
else:
    real = trials[-1]
    pos = np.load(real)
    ts = np.load(real.with_name(real.stem + '_timestamps.npy'))
    print(f"  using {real.name}: {len(pos)} samples, {ts[-1]-ts[0]:.1f} s")

    # fabricate Vicon: 100 Hz, interpolate the RS path, move to a Vicon frame,
    # 0.5 mm noise; RS stamps made 50 ms late
    grid = np.arange(ts[0] - 1.0, ts[-1] + 1.0, 0.01)
    interp = np.column_stack([np.interp(grid, ts, pos[:, i]) for i in range(3)])
    Rf = Rotation.from_euler('xyz', [180, 0, 90], degrees=True).as_matrix()
    tfv = np.array([2.0, 1.0, 0.8])
    vic = (Rf @ interp.T).T + tfv + rng.normal(0, 0.0005, interp.shape)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        np.save(td / 'vicon.npy', vic)
        np.save(td / 'vicon_timestamps.npy', grid)
        np.save(td / 'rs.npy', pos)
        np.save(td / 'rs_timestamps.npy', ts + 0.050)
        cmd = [sys.executable, str(HERE / 'compare_trajectories.py'),
               '--vicon', str(td / 'vicon.npy'), '--realsense', str(td / 'rs.npy'),
               '--fig', str(td / 'fig.png'), '--output', str(td / 'report.txt'),
               '--save-transform', str(td / 'tf.npz')]
        env = dict(os.environ, MPLBACKEND='Agg')
        res = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(HERE.parent))
        out = res.stdout + res.stderr
        print("  --- CLI output (tail) ---")
        print("\n".join("  | " + l for l in out.strip().splitlines()[-22:]))
        check(res.returncode == 0, "CLI exits 0 on real-shaped data")
        check((td / 'fig.png').exists() and (td / 'fig.png').stat().st_size > 10000, "figure written")
        check((td / 'report.txt').exists(), "report written")
        check((td / 'tf.npz').exists(), "transform saved")
        import re
        m = re.search(r"lag\s*:\s*([+-]?\d+\.\d+) ms", out)
        if m:
            check(abs(float(m.group(1)) + 50.0) < 5.0, f"lag on real data recovered as {m.group(1)} ms (true -50)")
        m = re.search(r"RMSE\s*:\s*(\d+\.\d+) mm", out)
        if m:
            check(float(m.group(1)) < 2.0, f"RMSE on self-derived Vicon is noise-level ({m.group(1)} mm)")

print("\n" + ("ALL PASS" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
