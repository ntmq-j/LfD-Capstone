#!/usr/bin/env python3
"""
Compare Vicon and RealSense wrist trajectories (proposal sec. 3.6, Metric 1).

The two recordings are NOT directly comparable as saved:
  - different clocks/latency: RealSense stamps are taken after capture +
    MediaPipe (~50 ms); Vicon stamps at message arrival. At 0.3 m/s hand
    speed a 50 ms offset alone is 15 mm of apparent error.
  - different rates: Vicon ~100 Hz, RealSense ~20-30 Hz with dropouts.
  - different frames: RealSense points are in camera_color_optical_frame,
    Vicon in its own world frame. Raw coordinates differ by metres.

Pipeline:
  1. Lag     - estimate (or take) a constant time offset and shift the
               RealSense timestamps onto the Vicon clock.
  2. Sync    - linearly interpolate Vicon at each RealSense sample time.
               Samples outside the Vicon span, or bracketed by a Vicon
               dropout (> max_vicon_gap), are discarded.
  3. Align   - rigid transform camera -> Vicon (rotation + translation,
               Kabsch/SVD, no scale: both systems are metric). Fitted on
               this trial, or loaded from a calibration trial.
  4. Metrics - RMSE / mean / max, per-axis, coverage, tolerance test.

Kalman-predicted RealSense samples (see track_wrist_realsense.py) are
excluded from RMSE when a *_measured.npy mask exists, per proposal sec. 3.4:
"Predicted values are marked as such and do not count towards RMSE".
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from scipy import stats


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def kabsch(P, Q):
    """Rigid transform (R, t) minimising sum ||R P_i + t - Q_i||^2.
    P, Q: Nx3 corresponding points. Returns R (3x3, det=+1) and t (3,)."""
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    Pc, Qc = P.mean(axis=0), Q.mean(axis=0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d if d != 0 else 1.0])   # fix reflections
    R = Vt.T @ D @ U.T
    t = Qc - R @ Pc
    return R, t


def rotation_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def load_trajectory(path, raw=False):
    """Load positions plus the optional sidecars written by the trackers.
    Returns (positions Nx3, timestamps or None, measured-mask or None).
    raw=True loads *_raw.npy (unfiltered measurements, no Kalman)."""
    path = Path(path)
    if raw:
        path = path.with_name(path.stem + '_raw.npy')
    pos = np.load(path)
    ts_path = path.with_name(path.stem + '_timestamps.npy')
    ts = np.load(ts_path) if ts_path.exists() else None
    mask = None
    if not raw:
        m_path = path.with_name(path.stem + '_measured.npy')
        if m_path.exists():
            mask = np.load(m_path).astype(bool)
    if ts is not None and len(ts) != len(pos):
        raise ValueError(f"{path.name}: {len(pos)} positions but {len(ts)} timestamps")
    if mask is not None and len(mask) != len(pos):
        raise ValueError(f"{path.name}: {len(pos)} positions but {len(mask)} mask entries")
    return pos, ts, mask


def save_transform(path, R, t, lag, note=''):
    np.savez(path, R=np.asarray(R), t=np.asarray(t), lag=float(lag), note=str(note))


def load_transform(path):
    d = np.load(path)
    return d['R'], d['t'], float(d['lag'])


# ----------------------------------------------------------------------------
class TrajectoryComparison:
    """Vicon (ground truth) vs RealSense wrist trajectory for ONE trial."""

    def __init__(self, vicon_traj, realsense_traj, vicon_ts=None, realsense_ts=None,
                 rs_measured=None, transform=None, lag=None, lag_search=0.3,
                 max_vicon_gap=0.05, verbose=True):
        """
        vicon_traj, realsense_traj : Nx3 positions (m), each in its own frame
        vicon_ts, realsense_ts     : timestamps (s). Without BOTH, falls back to
                                     index-based resampling (legacy, inaccurate).
        rs_measured : bool mask, False = Kalman-predicted -> excluded from RMSE
        transform   : (R, t) camera->Vicon from a calibration trial. None = fit here.
        lag         : seconds to ADD to RealSense timestamps to put them on the
                      Vicon clock. None = estimate by minimising post-alignment
                      RMSE over +-lag_search.
        max_vicon_gap : RealSense samples whose bracketing Vicon samples are
                        further apart than this (Vicon dropout) are discarded.
        """
        self.vicon_raw = np.asarray(vicon_traj, float)
        self.rs_raw = np.asarray(realsense_traj, float)
        for name, a in (('vicon', self.vicon_raw), ('realsense', self.rs_raw)):
            if a.ndim != 2 or a.shape[1] != 3:
                raise ValueError(f"{name} trajectory must be Nx3, got {a.shape}")
        self.vicon_ts = None if vicon_ts is None else np.asarray(vicon_ts, float)
        self.rs_ts = None if realsense_ts is None else np.asarray(realsense_ts, float)
        self.rs_measured = (np.ones(len(self.rs_raw), bool) if rs_measured is None
                            else np.asarray(rs_measured, bool))
        self.max_vicon_gap = float(max_vicon_gap)
        self.verbose = verbose
        self.info = {}

        if self.vicon_ts is None or self.rs_ts is None:
            self._sync_by_index()
        else:
            self._sync_by_time(lag, lag_search, transform)
        self._align(transform)
        if self.verbose:
            self._print_summary()

    # ------------------------------------------------------------------ sync
    def _sync_by_index(self):
        warnings.warn("No timestamps: resampling by sample index. The two systems are "
                      "NOT time-synchronised, so RMSE will include timing error. Record "
                      "*_timestamps.npy with both trackers.", stacklevel=3)
        n = min(len(self.vicon_raw), len(self.rs_raw))

        def resample(a):
            idx = np.linspace(0, len(a) - 1, n)
            return np.column_stack([np.interp(idx, np.arange(len(a)), a[:, i]) for i in range(3)])

        self.vicon = resample(self.vicon_raw)
        self.rs_cam = resample(self.rs_raw)
        self.t = np.arange(n, dtype=float)
        self.t_label = 'Sample index'
        self.lag = None
        self.info.update(sync='index (legacy, NOT time-synchronised)', n_used=n,
                         n_rs=len(self.rs_raw), n_vicon=len(self.vicon_raw))

    def _interp_vicon(self, t):
        """Vicon position at times t (on the Vicon clock) and a validity mask:
        inside the Vicon span AND not bridging a Vicon dropout."""
        order = np.argsort(self.vicon_ts)
        vt = self.vicon_ts[order]
        vp = self.vicon_raw[order]
        inside = (t >= vt[0]) & (t <= vt[-1])
        idx = np.clip(np.searchsorted(vt, t, side='right'), 1, len(vt) - 1)
        gap = vt[idx] - vt[idx - 1]
        ok = inside & (gap <= self.max_vicon_gap)
        interp = np.column_stack([np.interp(t, vt, vp[:, i]) for i in range(3)])
        return interp, ok

    def _rmse_at_lag(self, lag, transform):
        vic, ok = self._interp_vicon(self.rs_ts + lag)
        keep = ok & self.rs_measured
        if keep.sum() < 20:
            return np.inf
        P, Q = self.rs_raw[keep], vic[keep]
        if transform is None:
            R, t = kabsch(P, Q)
        else:
            R, t = transform
        e = np.linalg.norm((R @ P.T).T + t - Q, axis=1)
        return float(np.sqrt(np.mean(e ** 2)))

    def _estimate_lag(self, search, transform):
        """Grid-search the constant time offset that minimises post-alignment
        RMSE. Coarse 5 ms, then 1 ms around the best. Reported in info so it
        goes in the thesis as a pipeline-latency figure, not hidden."""
        coarse = np.arange(-search, search + 1e-9, 0.005)
        r = np.array([self._rmse_at_lag(l, transform) for l in coarse])
        if not np.isfinite(r).any():
            warnings.warn("Lag search found no overlap at any offset - using lag=0. "
                          "Do the two recordings overlap in time?", stacklevel=4)
            return 0.0
        best = coarse[int(np.argmin(r))]
        fine = np.arange(best - 0.005, best + 0.005 + 1e-9, 0.001)
        rf = np.array([self._rmse_at_lag(l, transform) for l in fine])
        lag = float(fine[int(np.argmin(rf))])
        self.info['lag_curve'] = (coarse, r)
        if search - abs(lag) < 0.006:
            warnings.warn(f"Estimated lag {lag*1000:+.0f} ms sits at the edge of the "
                          f"+-{search*1000:.0f} ms search window - widen --lag-search.",
                          stacklevel=4)
        return lag

    def _sync_by_time(self, lag, lag_search, transform):
        if lag is None:
            lag = self._estimate_lag(lag_search, transform)
            self.info['lag_source'] = f'estimated (search +-{lag_search*1000:.0f} ms)'
        else:
            self.info['lag_source'] = 'given'
        self.lag = float(lag)

        vic, ok = self._interp_vicon(self.rs_ts + self.lag)
        keep = ok & self.rs_measured
        if keep.sum() < 3:
            raise ValueError("Fewer than 3 overlapping samples after sync. Check that the "
                             "recordings overlap in time and the lag window is wide enough.")
        self.vicon = vic[keep]
        self.rs_cam = self.rs_raw[keep]
        self.t = self.rs_ts[keep] - self.rs_ts[0]
        self.t_label = 'Time (s)'

        def rate(ts):
            return (len(ts) - 1) / (ts.max() - ts.min()) if len(ts) > 1 else float('nan')

        self.info.update(sync='time (Vicon interpolated at RealSense sample times)',
                         lag_ms=self.lag * 1000,
                         n_rs=len(self.rs_raw), n_vicon=len(self.vicon_raw),
                         n_predicted_excluded=int((~self.rs_measured).sum()),
                         n_no_vicon=int((~ok & self.rs_measured).sum()),
                         n_used=int(keep.sum()),
                         rs_rate_hz=rate(self.rs_ts), vicon_rate_hz=rate(self.vicon_ts))

    # ----------------------------------------------------------------- align
    def _align(self, transform):
        if transform is None:
            if len(self.rs_cam) < 3:
                raise ValueError("Need at least 3 points to fit a rigid transform")
            self.R, self.t_vec = kabsch(self.rs_cam, self.vicon)
            self.info['alignment'] = ('rigid transform fitted on THIS trial '
                                      '(self-aligned: optimistic - use a calibration trial for the report)')
        else:
            self.R = np.asarray(transform[0], float)
            self.t_vec = np.asarray(transform[1], float)
            self.info['alignment'] = 'rigid transform loaded from calibration'
        # RealSense points expressed in the Vicon frame
        self.realsense = (self.R @ self.rs_cam.T).T + self.t_vec
        self.info['rotation_deg'] = rotation_angle_deg(self.R)
        self.info['translation_m'] = float(np.linalg.norm(self.t_vec))

    def save_transform(self, path, note=''):
        save_transform(path, self.R, self.t_vec, self.lag if self.lag is not None else 0.0, note)

    def _print_summary(self):
        i = self.info
        print(f"  sync : {i['sync']}")
        if self.lag is not None:
            print(f"  lag  : {i['lag_ms']:+.1f} ms added to RealSense stamps ({i['lag_source']})")
            print(f"  rates: RealSense {i['rs_rate_hz']:.1f} Hz, Vicon {i['vicon_rate_hz']:.1f} Hz")
            print(f"  used : {i['n_used']} of {i['n_rs']} RealSense samples "
                  f"({i['n_predicted_excluded']} predicted excluded, "
                  f"{i['n_no_vicon']} without Vicon coverage)")
        print(f"  align: {i['alignment']}")
        print(f"         rotation {i['rotation_deg']:.1f} deg, translation {i['translation_m']:.3f} m")

    # --------------------------------------------------------------- metrics
    def errors_mm(self):
        return np.linalg.norm(self.vicon - self.realsense, axis=1) * 1000

    def calculate_rmse(self):
        errors = self.errors_mm()
        return {
            'rmse_mm': float(np.sqrt(np.mean(errors ** 2))),
            'mean_error_mm': float(np.mean(errors)),
            'std_error_mm': float(np.std(errors)),
            'max_error_mm': float(np.max(errors)),
            'median_error_mm': float(np.median(errors)),
            'p95_error_mm': float(np.percentile(errors, 95)),
            'n_points': int(len(errors)),
            'errors': errors,
        }

    def calculate_robustness(self, tolerance_mm=5.0):
        """Per-point view: share of compared samples within tolerance."""
        errors = self.errors_mm()
        within = int(np.sum(errors <= tolerance_mm))
        return {
            'tolerance_mm': tolerance_mm,
            'within_tolerance': within,
            'total_points': int(len(errors)),
            'success_rate_percent': 100.0 * within / len(errors),
            'num_failures': int(len(errors) - within),
        }

    def calculate_coverage(self, max_gap_s=0.5):
        """Metric 2 support (proposal sec. 3.6): was the wrist path captured
        without interruption? Evaluated over the window where BOTH systems were
        recording. A gap is a stretch with no RealSense sample longer than
        max_gap_s (0.5 s = 15 frames at 30 fps = the Kalman bridging limit)."""
        if self.lag is None:
            return {'available': False}
        t0 = max(self.rs_ts.min() + self.lag, self.vicon_ts.min())
        t1 = min(self.rs_ts.max() + self.lag, self.vicon_ts.max())
        window = t1 - t0
        if window <= 0:
            return {'available': False}

        def gap_stats(ts):
            ts = np.sort(ts + self.lag)
            ts = ts[(ts >= t0) & (ts <= t1)]
            g = np.diff(np.concatenate([[t0], ts, [t1]]))
            lost = np.clip(g - max_gap_s, 0, None).sum()
            return g, lost

        g_all, lost_all = gap_stats(self.rs_ts)
        g_meas, lost_meas = gap_stats(self.rs_ts[self.rs_measured])
        return {
            'available': True,
            'window_s': float(window),
            'max_gap_s': max_gap_s,
            'max_gap_measured_s': float(g_meas.max()),
            'max_gap_with_prediction_s': float(g_all.max()),
            'n_gaps_measured': int((g_meas > max_gap_s).sum()),
            'n_gaps_with_prediction': int((g_all > max_gap_s).sum()),
            'coverage_measured_percent': 100.0 * (1 - lost_meas / window),
            'coverage_with_prediction_percent': 100.0 * (1 - lost_all / window),
            'uninterrupted': bool(g_all.max() <= max_gap_s),
        }

    def statistical_analysis(self, tolerance_mm=5.0):
        """One-sided one-sample t-test on the per-sample errors:
        H0: mean error >= tolerance, H1: mean error < tolerance.
        (Testing against 0 would always be 'significant' - errors are norms.)
        Cohen's d is (tolerance - mean) / std: positive = below tolerance."""
        rmse_result = self.calculate_rmse()
        errors = rmse_result['errors']
        t_stat, p_two = stats.ttest_1samp(errors, tolerance_mm)
        p_value = p_two / 2 if t_stat < 0 else 1 - p_two / 2
        sd = np.std(errors, ddof=1)
        cohens_d = (tolerance_mm - np.mean(errors)) / sd if sd > 0 else float('inf')
        return {
            'test': 'one-sided one-sample t-test, H1: mean error < tolerance',
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'within_tolerance': bool(rmse_result['rmse_mm'] <= tolerance_mm),
            'rmse_vs_tolerance': f"{rmse_result['rmse_mm']:.2f} mm vs {tolerance_mm:.2f} mm",
        }

    # ------------------------------------------------------------------ plot
    def plot_comparison(self, save_path=None):
        fig = plt.figure(figsize=(16, 10))
        rmse_result = self.calculate_rmse()
        errors = rmse_result['errors']
        x = self.t

        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot(*self.vicon.T, 'b-', label='Vicon (ground truth)', linewidth=2)
        ax1.plot(*self.realsense.T, 'r--', label='RealSense (aligned)', linewidth=2)
        ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
        ax1.set_title('3D trajectory (Vicon frame)')
        ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(self.vicon[:, 0], self.vicon[:, 1], 'b-', label='Vicon', linewidth=2)
        ax2.plot(self.realsense[:, 0], self.realsense[:, 1], 'r--', label='RealSense', linewidth=2)
        ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)'); ax2.set_title('X-Y projection')
        ax2.legend(); ax2.grid(True, alpha=0.3); ax2.axis('equal')

        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(x, errors, 'r-', linewidth=1)
        ax3.axhline(rmse_result['mean_error_mm'], color='b', linestyle='--',
                    label=f"Mean: {rmse_result['mean_error_mm']:.2f} mm")
        ax3.axhline(rmse_result['rmse_mm'], color='g', linestyle='--',
                    label=f"RMSE: {rmse_result['rmse_mm']:.2f} mm")
        ax3.set_xlabel(self.t_label); ax3.set_ylabel('Error (mm)')
        ax3.set_title('Positional error over time'); ax3.legend(); ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(2, 3, 4)
        ax4.hist(errors, bins=30, color='red', alpha=0.7, edgecolor='black')
        ax4.axvline(rmse_result['mean_error_mm'], color='b', linestyle='--', linewidth=2,
                    label=f"Mean: {rmse_result['mean_error_mm']:.2f} mm")
        ax4.axvline(rmse_result['rmse_mm'], color='g', linestyle='--', linewidth=2,
                    label=f"RMSE: {rmse_result['rmse_mm']:.2f} mm")
        ax4.set_xlabel('Error (mm)'); ax4.set_ylabel('Frequency')
        ax4.set_title('Error distribution'); ax4.legend(); ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(2, 3, 5)
        comp = (self.realsense - self.vicon) * 1000
        for i, (c, lab) in enumerate(zip('rgb', 'XYZ')):
            ax5.plot(x, comp[:, i], c + '-', alpha=0.7,
                     label=f'{lab} (bias {np.mean(comp[:, i]):+.2f}, sd {np.std(comp[:, i]):.2f} mm)')
        ax5.axhline(0, color='k', linewidth=0.5)
        ax5.set_xlabel(self.t_label); ax5.set_ylabel('RealSense - Vicon (mm)')
        ax5.set_title('Signed per-axis error'); ax5.legend(); ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(2, 3, 6)
        s = np.sort(errors)
        ax6.plot(s, np.arange(1, len(s) + 1) / len(s) * 100, 'b-', linewidth=2)
        for tol, c in ((5.0, 'r'), (3.0, 'orange'), (1.0, 'purple')):
            ax6.axvline(tol, color=c, linestyle='--', label=f'{tol:g} mm tolerance')
        ax6.set_xlabel('Error (mm)'); ax6.set_ylabel('Cumulative (%)')
        ax6.set_title('Cumulative error distribution'); ax6.legend(); ax6.grid(True, alpha=0.3)

        sub = self.info['sync']
        if self.lag is not None:
            sub += f" | lag {self.info['lag_ms']:+.0f} ms | n={self.info['n_used']}"
        fig.suptitle(f"RMSE {rmse_result['rmse_mm']:.2f} mm  -  {sub}", fontsize=11)
        plt.tight_layout(rect=(0, 0, 1, 0.97))

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Figure saved to: {save_path}")
        else:
            plt.show()
        return fig


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description='Compare Vicon and RealSense wrist trajectories')
    p.add_argument('--vicon', required=True, help='Vicon trajectory (.npy); *_timestamps.npy alongside')
    p.add_argument('--realsense', required=True, help='RealSense trajectory (.npy); sidecars alongside')
    p.add_argument('--raw', action='store_true', help='Use the unfiltered *_raw.npy RealSense measurements')
    p.add_argument('--task', help='Task name (task1, task2, task3)')
    p.add_argument('--condition', help='Condition (condA, condB, condC)')
    p.add_argument('--tolerance', type=float, default=5.0, help='Tolerance in mm (default 5.0)')
    p.add_argument('--lag', type=float, help='Fixed lag in seconds to add to RealSense stamps (skip search)')
    p.add_argument('--lag-search', type=float, default=0.3, help='Half-width of lag search window, s (default 0.3)')
    p.add_argument('--transform', help='Load camera->Vicon transform (+lag) from a calibration .npz')
    p.add_argument('--save-transform', help='Save the fitted transform (+lag) to this .npz for other trials')
    p.add_argument('--max-vicon-gap', type=float, default=0.05, help='Discard samples across Vicon gaps > this, s')
    p.add_argument('--fig', help='Figure path (default data/figures/<task>_<condition>.png)')
    p.add_argument('--output', help='Text report path')
    args = p.parse_args()

    print("\n" + "=" * 60 + "\nTRAJECTORY COMPARISON\n" + "=" * 60)

    vicon_traj, vicon_ts, _ = load_trajectory(args.vicon)
    rs_traj, rs_ts, rs_measured = load_trajectory(args.realsense, raw=args.raw)
    print(f"\nVicon    : {args.vicon}  shape {vicon_traj.shape}"
          f"{'' if vicon_ts is not None else '  (no timestamps!)'}")
    print(f"RealSense: {args.realsense}{' [raw]' if args.raw else ''}  shape {rs_traj.shape}"
          f"{'' if rs_ts is not None else '  (no timestamps!)'}"
          f"{'' if rs_measured is None else f'  ({int((~rs_measured).sum())} predicted)'}")

    transform, lag = None, args.lag
    if args.transform:
        R, t, lag_saved = load_transform(args.transform)
        transform = (R, t)
        if lag is None:
            lag = lag_saved
        print(f"Transform: loaded from {args.transform} (lag {lag_saved*1000:+.1f} ms)")

    print("\n" + "-" * 60 + "\nSYNC & ALIGNMENT\n" + "-" * 60)
    cmp = TrajectoryComparison(vicon_traj, rs_traj, vicon_ts=vicon_ts, realsense_ts=rs_ts,
                               rs_measured=rs_measured, transform=transform, lag=lag,
                               lag_search=args.lag_search, max_vicon_gap=args.max_vicon_gap)
    if args.save_transform:
        cmp.save_transform(args.save_transform, note=f"fitted on {Path(args.realsense).name}")
        print(f"✓ Transform saved to: {args.save_transform}")

    print("\n" + "-" * 60 + "\nACCURACY (Metric 1)\n" + "-" * 60)
    r = cmp.calculate_rmse()
    print(f"RMSE       : {r['rmse_mm']:.2f} mm   (n = {r['n_points']})")
    print(f"Mean error : {r['mean_error_mm']:.2f} ± {r['std_error_mm']:.2f} mm")
    print(f"Median/p95 : {r['median_error_mm']:.2f} / {r['p95_error_mm']:.2f} mm")
    print(f"Max error  : {r['max_error_mm']:.2f} mm")

    print("\n" + "-" * 60 + "\nCOVERAGE (Metric 2 support)\n" + "-" * 60)
    cov = cmp.calculate_coverage()
    if cov['available']:
        print(f"Overlap window          : {cov['window_s']:.1f} s")
        print(f"Coverage (measured)     : {cov['coverage_measured_percent']:.1f} %   "
              f"max gap {cov['max_gap_measured_s']*1000:.0f} ms, {cov['n_gaps_measured']} gaps > {cov['max_gap_s']*1000:.0f} ms")
        print(f"Coverage (w/ prediction): {cov['coverage_with_prediction_percent']:.1f} %   "
              f"max gap {cov['max_gap_with_prediction_s']*1000:.0f} ms, {cov['n_gaps_with_prediction']} gaps")
        print(f"Uninterrupted path      : {'YES' if cov['uninterrupted'] else 'NO'}")
    else:
        print("not available without timestamps")

    rob = cmp.calculate_robustness(tolerance_mm=args.tolerance)
    print(f"\nSamples within {args.tolerance:g} mm: {rob['within_tolerance']}/{rob['total_points']} "
          f"({rob['success_rate_percent']:.1f} %)")

    print("\n" + "-" * 60 + "\nSTATISTICS\n" + "-" * 60)
    st = cmp.statistical_analysis(tolerance_mm=args.tolerance)
    print(f"{st['test']}")
    print(f"t = {st['t_statistic']:.3f}, p = {st['p_value']:.4g}, Cohen's d = {st['cohens_d']:.3f}")
    print(f"RMSE vs tolerance: {st['rmse_vs_tolerance']}  ->  {'WITHIN' if st['within_tolerance'] else 'EXCEEDS'}")

    fig_name = f"{args.task}_{args.condition}" if args.task and args.condition else "comparison"
    fig_path = args.fig or f"data/figures/{fig_name}.png"
    cmp.plot_comparison(save_path=fig_path)
    plt.close('all')

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            w = f.write
            w("=" * 60 + "\nTRAJECTORY COMPARISON REPORT\n" + "=" * 60 + "\n\n")
            if args.task:
                w(f"Task: {args.task}\n")
            if args.condition:
                w(f"Condition: {args.condition}\n")
            w(f"Vicon: {args.vicon}\nRealSense: {args.realsense}{' [raw]' if args.raw else ''}\n")
            w(f"Tolerance: {args.tolerance} mm\n\n")
            w("-" * 60 + "\nSYNC & ALIGNMENT\n" + "-" * 60 + "\n")
            w(f"Sync: {cmp.info['sync']}\n")
            if cmp.lag is not None:
                w(f"Lag added to RealSense stamps: {cmp.info['lag_ms']:+.1f} ms ({cmp.info['lag_source']})\n")
                w(f"Samples used: {cmp.info['n_used']} of {cmp.info['n_rs']} "
                  f"({cmp.info['n_predicted_excluded']} predicted excluded)\n")
            w(f"Alignment: {cmp.info['alignment']}\n")
            w(f"Rotation {cmp.info['rotation_deg']:.2f} deg, translation {cmp.info['translation_m']:.4f} m\n\n")
            w("-" * 60 + "\nACCURACY\n" + "-" * 60 + "\n")
            w(f"RMSE: {r['rmse_mm']:.2f} mm (n={r['n_points']})\n")
            w(f"Mean Error: {r['mean_error_mm']:.2f} mm\nStd Error: {r['std_error_mm']:.2f} mm\n")
            w(f"Median Error: {r['median_error_mm']:.2f} mm\np95 Error: {r['p95_error_mm']:.2f} mm\n")
            w(f"Max Error: {r['max_error_mm']:.2f} mm\n\n")
            if cov['available']:
                w("-" * 60 + "\nCOVERAGE\n" + "-" * 60 + "\n")
                w(f"Coverage measured: {cov['coverage_measured_percent']:.1f} %\n")
                w(f"Coverage with prediction: {cov['coverage_with_prediction_percent']:.1f} %\n")
                w(f"Max gap (with prediction): {cov['max_gap_with_prediction_s']*1000:.0f} ms\n")
                w(f"Uninterrupted: {'YES' if cov['uninterrupted'] else 'NO'}\n\n")
            w("-" * 60 + "\nROBUSTNESS (per-sample)\n" + "-" * 60 + "\n")
            w(f"Success Rate: {rob['success_rate_percent']:.1f}%\nFailures: {rob['num_failures']}\n\n")
            w("-" * 60 + "\nSTATISTICS\n" + "-" * 60 + "\n")
            w(f"{st['test']}\n")
            w(f"t-statistic: {st['t_statistic']:.4f}\np-value: {st['p_value']:.6f}\n")
            w(f"Cohen's d: {st['cohens_d']:.4f}\n")
            w(f"Within Tolerance: {'YES' if st['within_tolerance'] else 'NO'}\n")
        print(f"✓ Report saved to: {args.output}")

    print("\n" + "=" * 60 + "\nCOMPARISON COMPLETE\n" + "=" * 60 + "\n")


if __name__ == '__main__':
    main()
