"""Run a trained policy online through PX4/Gazebo and record measured tracks."""

from pathlib import Path
import argparse
import csv
import json

from stable_baselines3 import PPO

from uav_px4_rl.env import Px4GazeboWireEnv
from uav_px4_rl.px4_backend import Px4RosBackend


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_FIELDS = [
    "episode",
    "step",
    "timestamp_us",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "goal_x",
    "goal_y",
    "goal_z",
    "wire_ax",
    "wire_ay",
    "wire_az",
    "wire_bx",
    "wire_by",
    "wire_bz",
    "wire_distance",
    "num_wires",
    "min_true_wire_distance",
    "nearest_wire_index",
    "lidar_confidence",
    "goal_distance",
    "reward",
    "collision",
    "reached_goal",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "ppo_px4_3d_lidar_multiwire.zip",
    )
    parser.add_argument(
        "--scenario",
        choices=["fixed", "random"],
        default="random",
        help="'random' is the 3B-lite curriculum; 'fixed' is the pipeline check.",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num-wires", type=int, default=3)
    parser.add_argument("--perception", default="lidar", choices=["lidar", "empty", "none"])
    parser.add_argument("--lidar-topic", default=None)
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use synchronous headless stepping instead of real-time visual evaluation.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "evaluation"
    )
    return parser.parse_args()


def row_from_info(episode, info, reward):
    position = info["position"]
    velocity = info["velocity"]
    goal = info["goal"]
    wire_a = info["wire_a"]
    wire_b = info["wire_b"]
    return {
        "episode": episode,
        "step": info["step"],
        "timestamp_us": info["timestamp_us"],
        "x": position[0],
        "y": position[1],
        "z": position[2],
        "vx": velocity[0],
        "vy": velocity[1],
        "vz": velocity[2],
        "goal_x": goal[0],
        "goal_y": goal[1],
        "goal_z": goal[2],
        "wire_ax": wire_a[0],
        "wire_ay": wire_a[1],
        "wire_az": wire_a[2],
        "wire_bx": wire_b[0],
        "wire_by": wire_b[1],
        "wire_bz": wire_b[2],
        "wire_distance": info["wire_distance"],
        "num_wires": info["num_wires"],
        "min_true_wire_distance": info["min_true_wire_distance"],
        "nearest_wire_index": info["nearest_wire_index"],
        "lidar_confidence": info["lidar_confidence"],
        "goal_distance": info["goal_distance"],
        "reward": reward,
        "collision": info["collision"],
        "reached_goal": info["reached_goal"],
    }


def main():
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Policy checkpoint not found: {args.model}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "actual_px4_trajectory.csv"
    summary_path = args.output_dir / "summary.json"

    model = PPO.load(str(args.model))
    backend = Px4RosBackend(synchronous=args.sync, lidar_topic=args.lidar_topic)
    env = Px4GazeboWireEnv(
        backend=backend,
        scenario_mode=args.scenario,
        num_wires=args.num_wires,
        perception_mode=args.perception,
    )
    rows = []
    summaries = []
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            rows.append(row_from_info(episode, info, 0.0))
            done = False
            total_reward = 0.0
            min_wire_distance = info["min_true_wire_distance"]
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                rows.append(row_from_info(episode, info, reward))
                total_reward += reward
                min_wire_distance = min(min_wire_distance, info["min_true_wire_distance"])
                done = terminated or truncated
            summaries.append(
                {
                    "episode": episode,
                    "seed": args.seed + episode,
                    "num_wires": args.num_wires,
                    "steps": info["step"],
                    "total_reward": total_reward,
                    "min_true_wire_distance": min_wire_distance,
                    "nearest_wire_index": info["nearest_wire_index"],
                    "lidar_confidence": info["lidar_confidence"],
                    "collision": info["collision"],
                    "reached_goal": info["reached_goal"],
                    "final_goal_distance": info["goal_distance"],
                }
            )
    finally:
        env.close()

    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summaries, stream, indent=2)
    print(f"Measured PX4 trajectory saved to {csv_path}")
    print(f"Evaluation summary saved to {summary_path}")


if __name__ == "__main__":
    main()
