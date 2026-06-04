"""Scenario generation for multi-wire avoidance tasks."""

from dataclasses import dataclass

import numpy as np

from .geometry import segment_model_pose


@dataclass(frozen=True)
class WireSegment:
    """One finite wire segment in ENU world coordinates."""

    start: np.ndarray
    end: np.ndarray


@dataclass(frozen=True)
class Scenario:
    """Episode geometry for a start, goal, and one or more wires."""

    start: np.ndarray
    goal: np.ndarray
    wires: tuple[WireSegment, ...]

    def __post_init__(self):
        if not self.wires:
            raise ValueError("Scenario requires at least one wire segment.")

    @property
    def wire_a(self):
        """Backward-compatible first wire start."""

        return self.wires[0].start

    @property
    def wire_b(self):
        """Backward-compatible first wire end."""

        return self.wires[0].end

    @property
    def wire_poses(self):
        """Return (center, quaternion) pairs for all wire model instances."""

        return tuple(segment_model_pose(wire.start, wire.end)[:2] for wire in self.wires)

    @property
    def wire_pose(self):
        """Backward-compatible first wire pose."""

        return self.wire_poses[0]


class ScenarioSampler:
    """Generate fixed diagnostic or randomized multi-wire scenarios."""

    WIRE_LENGTH = 2.4
    BOUNDS_LOW = np.array([-5.5, -3.5, 0.15], dtype=np.float32)
    BOUNDS_HIGH = np.array([5.5, 3.5, 3.2], dtype=np.float32)

    def __init__(self, mode="random", seed=None, num_wires=3):
        if mode not in {"fixed", "random"}:
            raise ValueError("Scenario mode must be 'fixed' or 'random'.")
        if int(num_wires) < 1:
            raise ValueError("num_wires must be at least 1.")
        self.mode = mode
        self.num_wires = int(num_wires)
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed):
        """Reset the deterministic scenario sequence."""

        self.rng = np.random.default_rng(seed)

    def sample(self):
        """Create one fixed diagnostic scene or one randomized training scene."""

        if self.mode == "fixed":
            return self._fixed()
        return self._random()

    def _fixed(self):
        start = np.array([-4.0, 0.0, 1.0], dtype=np.float32)
        goal = np.array([4.0, 0.0, 1.0], dtype=np.float32)
        centers = (
            np.array([-1.2, -0.18, 1.0], dtype=np.float32),
            np.array([0.0, 0.22, 1.15], dtype=np.float32),
            np.array([1.2, -0.12, 0.95], dtype=np.float32),
        )
        directions = (
            np.array([0.05, 1.0, 0.55], dtype=np.float32),
            np.array([-0.08, 1.0, -0.35], dtype=np.float32),
            np.array([0.12, 1.0, 0.45], dtype=np.float32),
        )
        wires = [
            self._wire_from_center_direction(centers[i % len(centers)], directions[i % len(directions)])
            for i in range(self.num_wires)
        ]
        return Scenario(start=start, goal=goal, wires=tuple(wires))

    def _random(self):
        for _ in range(100):
            start = np.array(
                [
                    self.rng.uniform(-4.4, -3.6),
                    self.rng.uniform(-0.8, 0.8),
                    self.rng.uniform(0.9, 1.5),
                ],
                dtype=np.float32,
            )
            goal = np.array(
                [
                    self.rng.uniform(3.6, 4.4),
                    self.rng.uniform(-0.8, 0.8),
                    self.rng.uniform(0.9, 1.5),
                ],
                dtype=np.float32,
            )
            wires = []
            for wire_index in range(self.num_wires):
                wire = self._sample_wire_between_start_goal(start, goal, wire_index)
                if wire is None:
                    break
                wires.append(wire)
            if len(wires) == self.num_wires:
                return Scenario(start=start, goal=goal, wires=tuple(wires))
        raise RuntimeError("Failed to generate a valid randomized multi-wire scenario.")

    def _sample_wire_between_start_goal(self, start, goal, wire_index):
        path = goal - start
        for _ in range(50):
            fraction = (wire_index + 1) / (self.num_wires + 1)
            fraction += self.rng.uniform(-0.08, 0.08)
            center = start + np.clip(fraction, 0.18, 0.82) * path
            center += np.array(
                [
                    self.rng.uniform(-0.35, 0.35),
                    self.rng.uniform(-0.55, 0.55),
                    self.rng.uniform(-0.35, 0.35),
                ],
                dtype=np.float32,
            )
            center[2] = np.clip(center[2], 0.65, 2.2)
            direction = np.array(
                [
                    self.rng.uniform(-0.18, 0.18),
                    1.0,
                    self.rng.uniform(-0.65, 0.85),
                ],
                dtype=np.float32,
            )
            wire = self._wire_from_center_direction(center, direction)
            if self._wire_is_valid(wire, start, goal):
                return wire
        return None

    def _wire_from_center_direction(self, center, direction):
        direction = np.asarray(direction, dtype=np.float32)
        direction /= np.linalg.norm(direction)
        half_segment = direction * (self.WIRE_LENGTH * 0.5)
        start = np.asarray(center, dtype=np.float32) - half_segment
        end = np.asarray(center, dtype=np.float32) + half_segment
        return WireSegment(start.astype(np.float32), end.astype(np.float32))

    def _wire_is_valid(self, wire, start, goal):
        endpoints = np.vstack([wire.start, wire.end])
        if np.any(endpoints < self.BOUNDS_LOW) or np.any(endpoints > self.BOUNDS_HIGH):
            return False
        if not (0.4 <= wire.start[2] <= 2.5 and 0.4 <= wire.end[2] <= 2.5):
            return False
        center = (wire.start + wire.end) * 0.5
        if np.linalg.norm(center - start) < 1.0 or np.linalg.norm(center - goal) < 1.0:
            return False
        return True
