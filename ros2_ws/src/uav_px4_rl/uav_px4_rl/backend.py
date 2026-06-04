"""Backend contracts for PX4/Gazebo and local diagnostics."""
from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class VehicleState:
    """Vehicle estimate returned by a flight backend, expressed in ENU."""

    position: np.ndarray
    velocity: np.ndarray
    timestamp_us: int = 0
    armed: bool = True
    offboard: bool = True


class KinematicDiagnosticBackend:
    """Small test backend; it is deliberately not a PX4 training substitute."""

    def __init__(self):
        self._state = VehicleState(
            np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        )
        self.scenario = None
        self._latest_lidar_points = None

    def prepare_episode(self, scenario):
        self.scenario = scenario
        self._state = VehicleState(
            scenario.start.copy(), np.zeros(3, dtype=np.float32), timestamp_us=0
        )
        return self._state

    def advance_velocity(self, velocity_enu, duration_seconds):
        velocity = np.asarray(velocity_enu, dtype=np.float32)
        position = self._state.position + velocity * duration_seconds
        timestamp = self._state.timestamp_us + int(duration_seconds * 1_000_000)
        self._state = VehicleState(position, velocity, timestamp)
        return self._state

    @property
    def latest_lidar_points(self):
        return self._latest_lidar_points

    def get_lidar_points(self):
        return self.latest_lidar_points

    def close(self):
        """No runtime resources are held by the diagnostic backend."""
