"""Geometry helpers for finite wire segments."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentDistance:
    """Distance from a point to one finite segment."""

    distance: float
    closest: np.ndarray
    normal: np.ndarray


@dataclass(frozen=True)
class SegmentSetDistance:
    """Nearest segment result plus per-segment distances."""

    distance: float
    closest: np.ndarray
    normal: np.ndarray
    wire_index: int
    distances: np.ndarray


def point_to_segment_distance(point, start, end):
    """Return distance, closest point, and outward normal for one segment."""

    point = np.asarray(point, dtype=np.float32)
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    delta = end - start
    denominator = float(np.dot(delta, delta))
    if denominator <= 1e-12:
        closest = start
    else:
        projection = float(np.dot(point - start, delta) / denominator)
        closest = start + np.clip(projection, 0.0, 1.0) * delta
    offset = point - closest
    distance = float(np.linalg.norm(offset))
    normal = offset / distance if distance > 1e-8 else np.zeros(3, dtype=np.float32)
    return SegmentDistance(distance, closest.astype(np.float32), normal.astype(np.float32))


def point_to_segments_distance(point, wires):
    """Return nearest distance result for a set of wire-like segments."""

    point = np.asarray(point, dtype=np.float32)
    results = []
    for wire in wires:
        start, end = _segment_endpoints(wire)
        results.append(point_to_segment_distance(point, start, end))
    if not results:
        raise ValueError("point_to_segments_distance requires at least one segment.")
    distances = np.asarray([result.distance for result in results], dtype=np.float32)
    wire_index = int(np.argmin(distances))
    nearest = results[wire_index]
    return SegmentSetDistance(
        distance=nearest.distance,
        closest=nearest.closest,
        normal=nearest.normal,
        wire_index=wire_index,
        distances=distances,
    )


def segment_model_pose(start, end):
    """Return cylinder model center, quaternion, and length for a segment."""

    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length <= 1e-8:
        raise ValueError("A wire segment must have non-zero length.")
    direction = axis / length
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    dot = float(np.dot(z_axis, direction))
    if dot < -0.999999:
        quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    else:
        xyz = np.cross(z_axis, direction)
        quaternion = np.array([xyz[0], xyz[1], xyz[2], 1.0 + dot], dtype=np.float32)
        quaternion /= np.linalg.norm(quaternion)
    return ((start + end) * 0.5).astype(np.float32), quaternion, length


def _segment_endpoints(segment):
    if hasattr(segment, "start") and hasattr(segment, "end"):
        return segment.start, segment.end
    if isinstance(segment, (tuple, list)) and len(segment) == 2:
        return segment[0], segment[1]
    raise TypeError("Segments must provide start/end attributes or be (start, end) pairs.")
