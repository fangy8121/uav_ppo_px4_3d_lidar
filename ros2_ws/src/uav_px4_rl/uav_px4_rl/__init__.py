"""PX4/Gazebo online reinforcement-learning environment for wire avoidance."""

from .perception import LiDARFeatureExtractor, SyntheticLidarSimulator
from .scenario import Scenario, ScenarioSampler, WireSegment

__all__ = [
    "LiDARFeatureExtractor",
    "Px4GazeboWireEnv",
    "Scenario",
    "ScenarioSampler",
    "SyntheticLidarSimulator",
    "WireSegment",
]


def __getattr__(name):
    if name == "Px4GazeboWireEnv":
        from .env import Px4GazeboWireEnv

        return Px4GazeboWireEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
