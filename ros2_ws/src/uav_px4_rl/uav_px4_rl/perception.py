"""Lightweight 3D LiDAR perception features and synthetic test point clouds."""

import numpy as np


class LiDARFeatureExtractor:
    """Compress local 3D points into fixed-length MLP-friendly features."""

    feature_names = (
        "nearest_obstacle_distance",
        "nearest_obstacle_direction_x",
        "nearest_obstacle_direction_y",
        "nearest_obstacle_direction_z",
        "front_min_range",
        "front_left_min_range",
        "front_right_min_range",
        "left_min_range",
        "right_min_range",
        "up_min_range",
        "down_min_range",
        "valid_point_ratio",
        "detection_confidence",
    )

    def __init__(self, max_range=8.0, min_range=0.03, confidence_points=24):
        self.max_range = float(max_range)
        self.min_range = float(min_range)
        self.confidence_points = max(1, int(confidence_points))

    @property
    def feature_dim(self):
        return len(self.feature_names)

    def empty_features(self):
        return np.asarray(
            [
                self.max_range,
                0.0,
                0.0,
                0.0,
                self.max_range,
                self.max_range,
                self.max_range,
                self.max_range,
                self.max_range,
                self.max_range,
                self.max_range,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

    def extract(self, points):
        if points is None:
            return self.empty_features()
        points = np.asarray(points, dtype=np.float32)
        if points.size == 0:
            return self.empty_features()
        points = points.reshape((-1, 3))
        total_count = points.shape[0]
        finite_mask = np.all(np.isfinite(points), axis=1)
        points = points[finite_mask]
        if points.size == 0:
            return self.empty_features()

        ranges = np.linalg.norm(points, axis=1)
        range_mask = (ranges >= self.min_range) & (ranges <= self.max_range)
        points = points[range_mask]
        ranges = ranges[range_mask]
        if points.size == 0:
            features = self.empty_features()
            features[-2] = 0.0 if total_count == 0 else float(np.count_nonzero(finite_mask) / total_count)
            return features

        nearest_index = int(np.argmin(ranges))
        nearest_distance = float(ranges[nearest_index])
        nearest_direction = points[nearest_index] / max(nearest_distance, 1e-6)
        valid_ratio = float(points.shape[0] / max(1, total_count))
        confidence = float(np.clip(points.shape[0] / self.confidence_points, 0.0, 1.0))

        features = np.asarray(
            [
                nearest_distance,
                nearest_direction[0],
                nearest_direction[1],
                nearest_direction[2],
                self._sector_min(points, ranges, self._front_mask(points)),
                self._sector_min(points, ranges, (points[:, 0] > 0.0) & (points[:, 1] >= 0.0)),
                self._sector_min(points, ranges, (points[:, 0] > 0.0) & (points[:, 1] <= 0.0)),
                self._sector_min(points, ranges, (points[:, 1] > 0.0) & (np.abs(points[:, 1]) >= np.abs(points[:, 0]))),
                self._sector_min(points, ranges, (points[:, 1] < 0.0) & (np.abs(points[:, 1]) >= np.abs(points[:, 0]))),
                self._sector_min(points, ranges, points[:, 2] > 0.0),
                self._sector_min(points, ranges, points[:, 2] < 0.0),
                valid_ratio,
                confidence,
            ],
            dtype=np.float32,
        )
        return features

    def _front_mask(self, points):
        return (points[:, 0] > 0.0) & (np.abs(points[:, 1]) <= np.maximum(points[:, 0], 1e-6))

    def _sector_min(self, points, ranges, mask):
        _ = points
        if not np.any(mask):
            return self.max_range
        return float(np.min(ranges[mask]))


class SyntheticLidarSimulator:
    """Generate diagnostic local point clouds from truth wire segments."""

    def __init__(
        self,
        max_range=8.0,
        points_per_wire=32,
        noise_std=0.015,
        dropout_prob=0.05,
        seed=None,
    ):
        self.max_range = float(max_range)
        self.points_per_wire = max(2, int(points_per_wire))
        self.noise_std = float(noise_std)
        self.dropout_prob = float(np.clip(dropout_prob, 0.0, 1.0))
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed):
        self.rng = np.random.default_rng(seed)

    def simulate(self, scenario, vehicle_position):
        vehicle_position = np.asarray(vehicle_position, dtype=np.float32)
        points = []
        fractions = np.linspace(0.0, 1.0, self.points_per_wire, dtype=np.float32)
        for wire in scenario.wires:
            start = np.asarray(wire.start, dtype=np.float32)
            end = np.asarray(wire.end, dtype=np.float32)
            segment_points = start[None, :] + fractions[:, None] * (end - start)[None, :]
            points.append(segment_points)
        if not points:
            return np.empty((0, 3), dtype=np.float32)

        cloud = np.vstack(points).astype(np.float32)
        if self.dropout_prob > 0.0:
            keep = self.rng.random(cloud.shape[0]) >= self.dropout_prob
            cloud = cloud[keep]
        if cloud.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        local_cloud = cloud - vehicle_position[None, :]
        if self.noise_std > 0.0:
            local_cloud += self.rng.normal(0.0, self.noise_std, local_cloud.shape).astype(np.float32)
        ranges = np.linalg.norm(local_cloud, axis=1)
        local_cloud = local_cloud[ranges <= self.max_range]
        return local_cloud.astype(np.float32).reshape((-1, 3))
