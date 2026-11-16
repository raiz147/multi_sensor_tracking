
import numpy as np


class KalmanFilter2D:
    """
    Constant-velocity Kalman filter.

    State: [x, y, vx, vy]
    Measurement: [x, y]
    """

    def __init__(
        self,
        initial_state,
        dt=1.0,
        process_acceleration_std=0.2
    ):
        self.state = np.array(
            initial_state, dtype=float
        )

        self.dt = dt

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        self.P = np.diag([10, 10, 25, 25]).astype(float)

        acceleration_variance = process_acceleration_std ** 2

        q = acceleration_variance

        self.Q = q * np.array([
            [dt**4 / 4, 0, dt**3 / 2, 0],
            [0, dt**4 / 4, 0, dt**3 / 2],
            [dt**3 / 2, 0, dt**2, 0],
            [0, dt**3 / 2, 0, dt**2]
        ])

    def predict(self):
        self.state = self.F @ self.state

        self.P = (
            self.F @ self.P @ self.F.T
            + self.Q
        )

        return self.state

    def update(self, measurement, measurement_std):
        measurement = np.asarray(
            measurement, dtype=float
        )

        R = np.eye(2) * measurement_std**2

        innovation = (
            measurement - self.H @ self.state
        )

        S = (
            self.H @ self.P @ self.H.T
            + R
        )

        K = (
            self.P @ self.H.T
            @ np.linalg.inv(S)
        )

        self.state = (
            self.state + K @ innovation
        )

        # Joseph-form covariance update.
        identity = np.eye(4)
        residual_matrix = identity - K @ self.H

        self.P = (
            residual_matrix @ self.P @ residual_matrix.T
            + K @ R @ K.T
        )

        # Reduce numerical asymmetry.
        self.P = (self.P + self.P.T) / 2

        return self.state
