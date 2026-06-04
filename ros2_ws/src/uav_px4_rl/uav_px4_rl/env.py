"""Gymnasium wrapper for PX4/Gazebo multi-wire LiDAR-feature training."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .backend import KinematicDiagnosticBackend
from .geometry import point_to_segments_distance
from .perception import LiDARFeatureExtractor, SyntheticLidarSimulator
from .rewards import RewardConfig, evaluate_transition
from .scenario import ScenarioSampler


class Px4GazeboWireEnv(gym.Env):
    """Online multi-wire navigation task using flight backend states."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        backend=None,
        scenario_mode="random",
        seed=None,
        action_dt=0.1,
        max_steps=250,
        max_velocity=(1.5, 1.5, 1.0),
        reward_config=None,
        num_wires=3,
        perception_mode="lidar",
        lidar_config=None,
    ):
        super().__init__()
        if backend is None:
            from .px4_backend import Px4RosBackend

            backend = Px4RosBackend()
        if perception_mode not in {"lidar", "empty", "none"}:
            raise ValueError("perception_mode must be 'lidar', 'empty', or 'none'.")

        self.backend = backend
        self.sampler = ScenarioSampler(mode=scenario_mode, seed=seed, num_wires=num_wires)
        self.action_dt = float(action_dt)
        self.max_steps = int(max_steps)
        self.max_velocity = np.asarray(max_velocity, dtype=np.float32)
        self.reward_config = reward_config or RewardConfig()
        self.num_wires = int(num_wires)
        self.perception_mode = perception_mode
        self.lidar_extractor = LiDARFeatureExtractor(
            **_config_subset(lidar_config, "extractor", {"max_range", "min_range", "confidence_points"})
        )
        simulator_config = _config_subset(
            lidar_config,
            "simulator",
            {"max_range", "points_per_wire", "noise_std", "dropout_prob", "seed"},
        )
        simulator_config.setdefault("max_range", self.lidar_extractor.max_range)
        simulator_config.setdefault("seed", seed)
        self.lidar_simulator = SyntheticLidarSimulator(**simulator_config)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        observation_dim = 3 + 3 + self.lidar_extractor.feature_dim + 3
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(observation_dim,), dtype=np.float32
        )
        self.scenario = None
        self.state = None
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.previous_goal_distance = 0.0
        self.steps = 0
        self._last_lidar_features = self.lidar_extractor.empty_features()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        _ = options
        if seed is not None:
            self.sampler.reseed(seed)
            self.lidar_simulator.reseed(seed)
        self.scenario = self.sampler.sample()
        self.state = self.backend.prepare_episode(self.scenario)
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.previous_goal_distance = self._goal_distance()
        self.steps = 0
        observation = self._observation()
        return observation, self._info()

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high
        )
        velocity_setpoint = action * self.max_velocity
        self.state = self.backend.advance_velocity(velocity_setpoint, self.action_dt)
        self.steps += 1

        goal_distance = self._goal_distance()
        wire = self._wire_result()
        reward, reached_goal, collision, out_of_bounds = evaluate_transition(
            self.previous_goal_distance,
            goal_distance,
            wire.distances,
            action,
            self.previous_action,
            self.state.position,
            self.reward_config,
            start=self.scenario.start,
            goal=self.scenario.goal,
        )
        terminated = reached_goal or collision or out_of_bounds
        truncated = self.steps >= self.max_steps and not terminated
        self.previous_goal_distance = goal_distance
        self.previous_action = action.copy()

        return self._observation(), reward, terminated, truncated, self._info()

    def close(self):
        self.backend.close()

    def _wire_result(self):
        return point_to_segments_distance(self.state.position, self.scenario.wires)

    def _goal_distance(self):
        return float(np.linalg.norm(self.scenario.goal - self.state.position))

    def _observation(self):
        lidar_features = self._perception_features()
        self._last_lidar_features = lidar_features
        return np.concatenate(
            [
                self.scenario.goal - self.state.position,
                self.state.velocity,
                lidar_features,
                self.previous_action,
            ]
        ).astype(np.float32)

    def _perception_features(self):
        if self.perception_mode in {"empty", "none"}:
            return self.lidar_extractor.empty_features()
        points = self._backend_lidar_points()
        if points is None and isinstance(self.backend, KinematicDiagnosticBackend):
            points = self.lidar_simulator.simulate(self.scenario, self.state.position)
        if points is None:
            return self.lidar_extractor.empty_features()
        return self.lidar_extractor.extract(points)

    def _backend_lidar_points(self):
        if hasattr(self.backend, "get_lidar_points"):
            return self.backend.get_lidar_points()
        if hasattr(self.backend, "latest_lidar_points"):
            return self.backend.latest_lidar_points
        return None

    def _info(self):
        wire = self._wire_result()
        goal_distance = self._goal_distance()
        reward_config = self.reward_config
        wires = np.asarray([[w.start, w.end] for w in self.scenario.wires], dtype=np.float32)
        lidar_features = np.asarray(self._last_lidar_features, dtype=np.float32)
        return {
            "position": self.state.position.copy(),
            "velocity": self.state.velocity.copy(),
            "goal": self.scenario.goal.copy(),
            "wire_a": self.scenario.wire_a.copy(),
            "wire_b": self.scenario.wire_b.copy(),
            "wires": wires.copy(),
            "wire_distance": wire.distance,
            "min_true_wire_distance": wire.distance,
            "wire_distances": wire.distances.copy(),
            "nearest_wire_index": wire.wire_index,
            "goal_distance": goal_distance,
            "lidar_features": lidar_features.copy(),
            "lidar_confidence": float(lidar_features[-1]),
            "num_wires": len(self.scenario.wires),
            "perception_mode": self.perception_mode,
            "collision": wire.distance <= reward_config.collision_radius,
            "reached_goal": goal_distance <= reward_config.goal_radius,
            "step": self.steps,
            "timestamp_us": self.state.timestamp_us,
            "armed": self.state.armed,
            "offboard": self.state.offboard,
        }


def _config_subset(config, section, allowed_keys):
    if config is None:
        return {}
    source = config.get(section, config) if isinstance(config, dict) else {}
    return {key: value for key, value in source.items() if key in allowed_keys}
