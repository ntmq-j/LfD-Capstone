#!/usr/bin/env python3
"""
Constant-velocity Kalman filter for 3D wrist tracking.

Implements Strategy 2 ("Temporal Prediction via Kalman Filter") from the
research proposal, sec. 3.4: during short occlusions (<= 15 frames at 30 fps,
i.e. <= 0.5 s) the wrist position is propagated with a constant-velocity model
so the trajectory stays continuous. The tracker flags predicted samples so the
comparison stage can exclude them from RMSE while keeping them for continuity.

State        x = [px, py, pz, vx, vy, vz]   (m, m/s)
Measurement  z = [px, py, pz]               (m)
"""

import numpy as np


class WristKalman:
    def __init__(self, accel_noise=2.0, meas_noise=(0.005, 0.005, 0.008),
                 init_vel_std=1.0):
        """
        accel_noise  : std of unmodelled acceleration, m/s^2. Larger = trusts
                       measurements more, follows fast moves, smooths less.
        meas_noise   : per-axis std of the position measurement, m. Defaults
                       come from the D435i depth-noise measurement (0.7 mm
                       frame-to-frame at 0.6 m) plus MediaPipe landmark jitter,
                       with Z loosened because depth holes bias it.
        init_vel_std : std of the velocity prior at (re)initialisation, m/s.
        """
        self.sigma_a = float(accel_noise)
        self.R = np.diag(np.asarray(meas_noise, float) ** 2)
        self.init_vel_var = float(init_vel_std) ** 2
        self.H = np.hstack([np.eye(3), np.zeros((3, 3))])   # z = H x
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.x = None
        self.P = None
        self.frames_since_measurement = 0

    @property
    def initialised(self):
        return self.x is not None

    @property
    def position(self):
        return None if self.x is None else self.x[:3].copy()

    @property
    def velocity(self):
        return None if self.x is None else self.x[3:].copy()

    @property
    def position_std(self):
        """Per-axis 1-sigma position uncertainty, m. Grows while predicting."""
        return None if self.P is None else np.sqrt(np.diag(self.P)[:3])

    # ------------------------------------------------------------------
    def initialise(self, z):
        z = np.asarray(z, float)
        self.x = np.concatenate([z, np.zeros(3)])
        self.P = np.zeros((6, 6))
        self.P[:3, :3] = self.R
        self.P[3:, 3:] = np.eye(3) * self.init_vel_var
        self.frames_since_measurement = 0

    def predict(self, dt):
        """Propagate the state by dt seconds under constant velocity."""
        if self.x is None:
            return None
        dt = float(dt)
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        # Discrete white-noise-acceleration model, per axis:
        #   Q = sigma_a^2 * [[dt^4/4, dt^3/2],
        #                    [dt^3/2, dt^2  ]]
        q = self.sigma_a ** 2
        Qpp = np.eye(3) * (q * dt ** 4 / 4)
        Qpv = np.eye(3) * (q * dt ** 3 / 2)
        Qvv = np.eye(3) * (q * dt ** 2)
        Q = np.block([[Qpp, Qpv], [Qpv, Qvv]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.frames_since_measurement += 1
        return self.position

    def update(self, z):
        """Fuse a position measurement. Initialises on the first call."""
        z = np.asarray(z, float)
        if self.x is None:
            self.initialise(z)
            return self.position
        y = z - self.H @ self.x                          # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ self.H
        # Joseph form: stays symmetric positive-definite under rounding
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.frames_since_measurement = 0
        return self.position

    def step(self, dt, z=None):
        """predict(dt), then update(z) if a measurement is available."""
        if z is None:
            return self.predict(dt)
        if self.x is None:
            return self.update(z)
        self.predict(dt)
        return self.update(z)
