import numpy as np

from uav_px4_rl.backend import KinematicDiagnosticBackend
from uav_px4_rl.env import Px4GazeboWireEnv
from uav_px4_rl.frames import enu_to_ned, ned_to_enu
from uav_px4_rl.geometry import (
    point_to_segment_distance,
    point_to_segments_distance,
    segment_model_pose,
)
from uav_px4_rl.perception import LiDARFeatureExtractor, SyntheticLidarSimulator
from uav_px4_rl.scenario import ScenarioSampler, WireSegment


def test_coordinate_conversions_round_trip():
    vector = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(ned_to_enu(enu_to_ned(vector)), vector)


def test_fixed_multi_wire_pose_matches_world_length():
    scenario = ScenarioSampler(mode="fixed", num_wires=3).sample()
    assert len(scenario.wires) == 3
    assert len(scenario.wire_poses) == 3
    for wire in scenario.wires:
        center, quaternion, length = segment_model_pose(wire.start, wire.end)
        assert center.shape == (3,)
        assert np.isclose(length, ScenarioSampler.WIRE_LENGTH, atol=1e-3)
        assert np.isclose(np.linalg.norm(quaternion), 1.0)


def test_random_multi_wire_scenarios_are_valid_and_vary():
    sampler = ScenarioSampler(mode="random", seed=11, num_wires=3)
    first = sampler.sample()
    second = sampler.sample()
    assert not np.allclose(first.wires[0].start, second.wires[0].start)
    for scenario in (first, second):
        assert scenario.start[0] < -3.5
        assert scenario.goal[0] > 3.5
        assert len(scenario.wires) == 3
        for wire in scenario.wires:
            assert 0.4 <= wire.start[2] <= 2.5
            assert 0.4 <= wire.end[2] <= 2.5
            assert np.isclose(
                np.linalg.norm(wire.end - wire.start),
                ScenarioSampler.WIRE_LENGTH,
                atol=1e-3,
            )


def test_segment_distance_uses_finite_wire():
    result = point_to_segment_distance(
        [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [0.0, 1.0, 2.0]
    )
    assert np.isclose(result.distance, 0.0)


def test_point_to_segments_distance_finds_nearest_and_all_distances():
    wires = (
        WireSegment(
            np.array([0.0, -1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        ),
        WireSegment(
            np.array([2.0, -1.0, 0.0], dtype=np.float32),
            np.array([2.0, 1.0, 0.0], dtype=np.float32),
        ),
    )
    result = point_to_segments_distance([1.8, 0.0, 0.0], wires)
    assert np.isclose(result.distance, 0.2, atol=1e-6)
    assert result.wire_index == 1
    assert result.distances.shape == (2,)
    np.testing.assert_allclose(result.distances, [1.8, 0.2], atol=1e-6)


def test_lidar_feature_extractor_empty_point_cloud_has_fixed_dimension():
    extractor = LiDARFeatureExtractor(max_range=6.0)
    features = extractor.extract(np.empty((0, 3), dtype=np.float32))
    assert features.shape == (extractor.feature_dim,)
    assert extractor.feature_dim == 13
    assert features[-1] == 0.0


def test_lidar_feature_extractor_valid_point_cloud_has_fixed_dimension():
    extractor = LiDARFeatureExtractor(max_range=6.0)
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.2],
            [0.5, -0.2, 0.1],
            [np.nan, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    features = extractor.extract(points)
    assert features.shape == (extractor.feature_dim,)
    assert np.isfinite(features).all()
    assert features[0] < 6.0
    assert features[-1] > 0.0


def test_synthetic_lidar_simulator_generates_multi_wire_points():
    scenario = ScenarioSampler(mode="fixed", num_wires=3).sample()
    simulator = SyntheticLidarSimulator(
        max_range=8.0, points_per_wire=8, noise_std=0.0, dropout_prob=0.0, seed=3
    )
    points = simulator.simulate(scenario, scenario.start)
    assert points.shape == (24, 3)
    assert np.isfinite(points).all()


def test_environment_accepts_backend_states_without_px4_runtime():
    env = Px4GazeboWireEnv(
        backend=KinematicDiagnosticBackend(),
        scenario_mode="random",
        seed=4,
        max_steps=2,
        num_wires=3,
    )
    obs, info = env.reset(seed=4)
    assert obs.shape == env.observation_space.shape == (22,)
    assert info["step"] == 0
    assert info["num_wires"] == 3
    assert info["wire_distances"].shape == (3,)
    assert not np.isclose(obs[9], info["min_true_wire_distance"])
    obs, reward, terminated, truncated, info = env.step(
        np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )
    assert obs.shape == (22,)
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    _, _, _, truncated, _ = env.step(np.zeros(3, dtype=np.float32))
    assert truncated
    env.close()
