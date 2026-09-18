import math

import numpy as np


class KalmanGazeFilter:
    """Two-dimensional constant-velocity Kalman filter for screen gaze."""

    def __init__(
        self,
        process_noise=800.0,
        measurement_noise=225.0,
        dead_zone=0.0,
        max_movement=None,
    ):
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.dead_zone = float(dead_zone)
        self.max_movement = max_movement
        self.state = None
        self.covariance = None

    def reset(self):
        self.state = None
        self.covariance = None

    def update(self, x, y, dt=1.0 / 30.0):
        measurement = np.array([float(x), float(y)], dtype=float)
        dt = float(np.clip(dt, 0.001, 0.2))

        if self.state is None:
            self.state = np.array(
                [measurement[0], measurement[1], 0.0, 0.0],
                dtype=float,
            )
            self.covariance = np.diag([100.0, 100.0, 1000.0, 1000.0])
            return float(measurement[0]), float(measurement[1])

        transition = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        measurement_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        process = self.process_noise
        process_covariance = process * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=float,
        )

        predicted_state = transition @ self.state
        predicted_covariance = (
            transition @ self.covariance @ transition.T
            + process_covariance
        )

        innovation = measurement - measurement_matrix @ predicted_state
        innovation_covariance = (
            measurement_matrix
            @ predicted_covariance
            @ measurement_matrix.T
            + np.eye(2) * self.measurement_noise
        )
        kalman_gain = (
            predicted_covariance
            @ measurement_matrix.T
            @ np.linalg.inv(innovation_covariance)
        )

        next_state = predicted_state + kalman_gain @ innovation
        identity = np.eye(4)
        next_covariance = (
            identity - kalman_gain @ measurement_matrix
        ) @ predicted_covariance

        previous_x, previous_y = self.state[:2]
        next_x, next_y = next_state[:2]
        movement = math.hypot(next_x - previous_x, next_y - previous_y)

        if movement < self.dead_zone:
            next_state[0] = previous_x
            next_state[1] = previous_y

        elif self.max_movement is not None and movement > self.max_movement:
            scale = self.max_movement / movement
            next_state[0] = previous_x + (next_x - previous_x) * scale
            next_state[1] = previous_y + (next_y - previous_y) * scale

        self.state = next_state
        self.covariance = next_covariance

        return float(next_state[0]), float(next_state[1])
