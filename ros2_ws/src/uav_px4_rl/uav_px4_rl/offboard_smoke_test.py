"""Verify PX4 Offboard and Gazebo services before launching PPO training."""

import argparse
import time

import numpy as np

from .px4_backend import Px4RosBackend
from .scenario import ScenarioSampler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync", action="store_true", help="Step Gazebo synchronously.")
    return parser.parse_args()


def main():
    args = parse_args()
    backend = Px4RosBackend(synchronous=args.sync)
    scenario = ScenarioSampler(mode="fixed").sample()
    waypoints = [
        np.array([-1.0, -1.2, 1.3], dtype=np.float32),
        np.array([1.0, -1.2, 1.3], dtype=np.float32),
        scenario.goal,
    ]
    try:
        state = backend.prepare_episode(scenario)
        print(f"Takeoff/start acquired at ENU position {state.position}.")
        for waypoint in waypoints:
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                state = backend.get_state()
                offset = waypoint - state.position
                if np.linalg.norm(offset) < 0.25:
                    print(f"Reached waypoint {waypoint}.")
                    break
                velocity = np.clip(offset * 0.8, -0.8, 0.8)
                backend.advance_velocity(velocity, 0.1)
            else:
                raise TimeoutError(f"Did not reach smoke-test waypoint {waypoint}.")
        print("PX4 Offboard and Gazebo scene smoke test passed.")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
