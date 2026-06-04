from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    goal_radius: float = 0.35
    collision_radius: float = 0.35
    safe_distance: float = 0.8
    progress_scale: float = 5.0
    safe_distance_scale: float = 3.0
    action_delta_scale: float = 0.03
    step_penalty: float = 0.02
    success_bonus: float = 120.0
    collision_penalty: float = 120.0
    out_of_bounds_penalty: float = 60.0
    bounds_low: tuple = (-5.5, -3.5, 0.15)
    bounds_high: tuple = (5.5, 3.5, 3.2)
    altitude_scale: float = 0.8
    altitude_tolerance: float = 0.35
    corridor_scale: float = 0.8
    corridor_radius: float = 1.5
    desired_altitude: Optional[float] = None


def evaluate_transition(
    previous_goal_distance,
    goal_distance,
    wire_distances,
    action,
    previous_action,
    position,
    reward_config,
    start=None,
    goal=None,
):
    """Compute reward and terminal flags from simulator truth feedback."""

    position = np.asarray(position, dtype=np.float32)
    distances = np.asarray(wire_distances, dtype=np.float32).reshape(-1)
    if distances.size == 0:
        distances = np.asarray([np.inf], dtype=np.float32)
    nearest_wire_distance = float(np.min(distances))

    reached_goal = bool(goal_distance <= reward_config.goal_radius)
    collision = bool(nearest_wire_distance <= reward_config.collision_radius)
    out_of_bounds = bool(
        np.any(position < np.asarray(reward_config.bounds_low, dtype=np.float32))
        or np.any(position > np.asarray(reward_config.bounds_high, dtype=np.float32))
    )

    reward = reward_config.progress_scale * (previous_goal_distance - goal_distance)
    reward -= reward_config.step_penalty
    reward -= reward_config.action_delta_scale * float(
        np.linalg.norm(np.asarray(action) - np.asarray(previous_action))
    )

    safe_violations = np.maximum(0.0, reward_config.safe_distance - distances)
    reward -= reward_config.safe_distance_scale * float(np.sum(safe_violations))
    reward -= _altitude_penalty(position, start, goal, reward_config)
    reward -= _corridor_penalty(position, start, goal, reward_config)

    if reached_goal:
        reward += reward_config.success_bonus
    if collision:
        reward -= reward_config.collision_penalty
    if out_of_bounds:
        reward -= reward_config.out_of_bounds_penalty

    return float(reward), reached_goal, collision, out_of_bounds


def _altitude_penalty(position, start, goal, config):
    if config.altitude_scale <= 0.0:
        return 0.0
    if config.desired_altitude is not None:
        desired_altitude = float(config.desired_altitude)
    elif start is not None and goal is not None:
        desired_altitude = float((np.asarray(start)[2] + np.asarray(goal)[2]) * 0.5)
    else:
        return 0.0
    excess = max(0.0, abs(float(position[2]) - desired_altitude) - config.altitude_tolerance)
    return config.altitude_scale * excess


def _corridor_penalty(position, start, goal, config):
    if config.corridor_scale <= 0.0 or start is None or goal is None:
        return 0.0
    start_xy = np.asarray(start, dtype=np.float32)[:2]
    goal_xy = np.asarray(goal, dtype=np.float32)[:2]
    position_xy = np.asarray(position, dtype=np.float32)[:2]
    segment = goal_xy - start_xy
    denominator = float(np.dot(segment, segment))
    if denominator <= 1e-12:
        return 0.0
    t = float(np.clip(np.dot(position_xy - start_xy, segment) / denominator, 0.0, 1.0))
    closest = start_xy + t * segment
    corridor_distance = float(np.linalg.norm(position_xy - closest))
    excess = max(0.0, corridor_distance - config.corridor_radius)
    return config.corridor_scale * excess
