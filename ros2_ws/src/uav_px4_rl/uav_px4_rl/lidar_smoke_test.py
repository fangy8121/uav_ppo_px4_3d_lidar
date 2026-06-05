"""Verify that the PX4 backend receives PointCloud2 and extracts LiDAR features."""

import argparse
import time

import numpy as np

from .perception import LiDARFeatureExtractor
from .px4_backend import Px4RosBackend
from .scenario import ScenarioSampler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lidar-topic", default="/x500/lidar/points")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--max-range", type=float, default=8.0)
    parser.add_argument("--min-usable-points", type=int, default=1)
    parser.add_argument("--min-confidence", type=float, default=0.01)
    parser.add_argument("--num-wires", type=int, default=3)
    parser.add_argument(
        "--prepare-fixed-scene",
        action="store_true",
        help="Move through Px4RosBackend.prepare_episode before checking the point cloud.",
    )
    return parser.parse_args()


def _usable_points(points, extractor):
    points = np.asarray(points, dtype=np.float32).reshape((-1, 3))
    if points.size == 0:
        return points, np.empty((0, 3), dtype=np.float32)

    finite = points[np.all(np.isfinite(points), axis=1)]
    if finite.size == 0:
        return points, np.empty((0, 3), dtype=np.float32)

    ranges = np.linalg.norm(finite, axis=1)
    usable = finite[(ranges >= extractor.min_range) & (ranges <= extractor.max_range)]
    return points, usable


def main():
    args = parse_args()
    extractor = LiDARFeatureExtractor(max_range=args.max_range)
    backend = Px4RosBackend(synchronous=False, lidar_topic=args.lidar_topic)
    best_sample = None

    try:
        if args.prepare_fixed_scene:
            scenario = ScenarioSampler(mode="fixed", num_wires=args.num_wires).sample()
            state = backend.prepare_episode(scenario)
            print(
                "Prepared fixed LiDAR smoke scene at ENU position "
                f"{np.round(state.position, 3)}."
            )

        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            points = backend.get_lidar_points()
            if points is not None:
                raw_points, usable = _usable_points(points, extractor)
                features = extractor.extract(raw_points)
                best_sample = (raw_points.shape[0], usable.shape[0], features)
                if (
                    usable.shape[0] >= args.min_usable_points
                    and features.shape == (extractor.feature_dim,)
                    and np.isfinite(features).all()
                    and float(features[-1]) >= args.min_confidence
                ):
                    print(
                        "LiDAR smoke test passed: "
                        f"topic={args.lidar_topic}, "
                        f"raw_points={raw_points.shape[0]}, "
                        f"usable_points={usable.shape[0]}, "
                        f"nearest_obstacle_distance={features[0]:.3f}, "
                        f"detection_confidence={features[-1]:.3f}."
                    )
                    return
            time.sleep(0.1)

        if best_sample is None:
            raise TimeoutError(f"No PointCloud2 sample received from {args.lidar_topic}.")
        raw_count, usable_count, features = best_sample
        raise TimeoutError(
            "No usable LiDAR feature sample received before timeout: "
            f"topic={args.lidar_topic}, raw_points={raw_count}, "
            f"usable_points={usable_count}, detection_confidence={features[-1]:.3f}."
        )
    finally:
        backend.close()


if __name__ == "__main__":
    main()
