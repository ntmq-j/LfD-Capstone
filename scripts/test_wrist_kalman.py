#!/usr/bin/env python3
"""
Sanity test for WristKalman on a synthetic wrist path with a 10-frame
occlusion, at the ~20 Hz the ROS path delivers in the VM.

Checks:
  1. filtering reduces error on measured frames (it smooths, not just passes)
  2. predictions through the gap start close and drift by a bounded amount
  3. the filter re-acquires within a few frames after the gap

Run:  venv/bin/python scripts/test_wrist_kalman.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wrist_kalman import WristKalman  # noqa: E402

rng = np.random.default_rng(1)

# ---- synthetic wrist path: ~0.3 m/s peak, ~0.45 m/s^2 peak accel ----------
N = 400
t = np.cumsum(rng.uniform(0.045, 0.055, N))          # ~20 Hz with jitter
t -= t[0]


def truth(tt):
    return np.column_stack([0.20 * np.sin(1.5 * tt),
                            0.15 * np.sin(1.1 * tt + 0.4),
                            0.05 * np.cos(0.7 * tt) + 0.60])


gt = truth(t)
meas_sd = np.array([0.005, 0.005, 0.008])
meas = gt + rng.normal(0, meas_sd, gt.shape)

gap = np.zeros(N, bool)
gap[200:210] = True                                   # 10 frames ~ 0.5 s

# ---- run ------------------------------------------------------------------
kf = WristKalman(accel_noise=2.0, meas_noise=meas_sd)
filt = np.full((N, 3), np.nan)
predicted = np.zeros(N, bool)
for i in range(N):
    dt = t[i] - t[i - 1] if i else 0.0
    if gap[i]:
        if kf.initialised:
            filt[i] = kf.predict(dt)
            predicted[i] = True
    else:
        filt[i] = kf.step(dt, meas[i])

err = lambda a, b: np.linalg.norm(a - b, axis=1) * 1000   # noqa: E731  (mm)

measured = ~gap
raw_rms = np.sqrt(np.mean(err(meas[measured], gt[measured]) ** 2))
filt_rms = np.sqrt(np.mean(err(filt[measured], gt[measured]) ** 2))
gap_err = err(filt[gap], gt[gap])
after = np.arange(210, 215)
after_err = err(filt[after], gt[after])

print("=" * 60)
print("WristKalman synthetic test  (~20 Hz, 10-frame gap)")
print("=" * 60)
print(f"raw measurement RMS error (measured frames) : {raw_rms:6.2f} mm")
print(f"filtered RMS error (measured frames)        : {filt_rms:6.2f} mm   "
      f"({100 * (1 - filt_rms / raw_rms):.0f}% lower)")
print(f"prediction error through gap, per frame (mm): "
      + " ".join(f"{e:5.1f}" for e in gap_err))
print(f"re-acquisition error, 5 frames after gap    : "
      + " ".join(f"{e:5.1f}" for e in after_err))
print(f"position 1-sigma at end of gap (mm)         : "
      + " ".join(f"{s * 1000:5.1f}" for s in kf.position_std) if kf.position_std is not None else "")

ok = True


def check(cond, msg):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    ok &= bool(cond)


print()
check(filt_rms < raw_rms, "filter reduces error on measured frames")
check(gap_err[0] < 20, f"first predicted frame within 20 mm (got {gap_err[0]:.1f})")
check(gap_err.max() < 150, f"worst prediction in a 0.5 s gap under 150 mm (got {gap_err.max():.1f})")
check(after_err[-1] < 20, f"re-acquired within 5 frames to <20 mm (got {after_err[-1]:.1f})")
check(kf.frames_since_measurement == 0, "bookkeeping: frames_since_measurement reset after update")

print("\n" + ("ALL PASS" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
