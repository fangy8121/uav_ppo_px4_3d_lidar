from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest

from uav_px4_rl.px4_backend import Px4RosBackend


class FakeFuture:
    def __init__(self, response):
        self._response = response

    def done(self):
        return True

    def result(self):
        return self._response


class PendingFuture:
    def done(self):
        return False

    def result(self):
        return None


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def call_async(self, request):
        self.requests.append(request)
        response = self.responses.pop(0)
        if hasattr(response, "done") and hasattr(response, "result"):
            return response
        return FakeFuture(response)


class FakeLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)


def make_backend(logger):
    backend = object.__new__(Px4RosBackend)
    backend.setup_timeout = 1.0
    backend.service_timeout = 1.0
    backend.service_retries = 3
    backend.service_retry_delay = 0.0
    backend.node = SimpleNamespace(get_logger=lambda: logger)
    return backend


def test_gazebo_service_retries_rejected_response():
    logger = FakeLogger()
    backend = make_backend(logger)
    client = FakeClient(
        [
            SimpleNamespace(success=False, status_message="busy"),
            SimpleNamespace(success=True),
        ]
    )

    response = backend._call_gazebo_service(
        client,
        lambda: SimpleNamespace(payload="request"),
        "setting pose for start_marker",
    )

    assert response.success
    assert len(client.requests) == 2
    assert len(logger.warnings) == 1


def test_latest_lidar_points_returns_none_before_first_sample():
    backend = object.__new__(Px4RosBackend)
    backend._lock = Lock()
    backend._latest_lidar_points = None

    assert backend.latest_lidar_points is None


def test_gazebo_service_raises_after_retry_limit():
    logger = FakeLogger()
    backend = make_backend(logger)
    client = FakeClient(
        [
            SimpleNamespace(success=False, status_message="busy"),
            SimpleNamespace(success=False, status_message="busy"),
            SimpleNamespace(success=False, status_message="busy"),
        ]
    )

    with pytest.raises(RuntimeError, match="setting pose for wire_obstacle"):
        backend._call_gazebo_service(
            client,
            lambda: SimpleNamespace(payload="request"),
            "setting pose for wire_obstacle",
        )

    assert len(client.requests) == 3
    assert len(logger.warnings) == 2


def test_gazebo_service_retries_timed_out_future():
    logger = FakeLogger()
    backend = make_backend(logger)
    backend.service_timeout = 0.0
    client = FakeClient(
        [
            PendingFuture(),
            SimpleNamespace(success=True),
        ]
    )

    response = backend._call_gazebo_service(
        client,
        lambda: SimpleNamespace(payload="request"),
        "controlling Gazebo simulation time",
    )

    assert response.success
    assert len(client.requests) == 2
    assert len(logger.warnings) == 1


def test_move_to_position_reasserts_offboard_when_state_drops():
    backend = object.__new__(Px4RosBackend)
    backend.setup_timeout = 1.0
    backend.synchronous = False
    backend.VehicleCommand = SimpleNamespace(
        VEHICLE_CMD_DO_SET_MODE=176,
        VEHICLE_CMD_COMPONENT_ARM_DISARM=400,
    )
    backend.commands = []
    backend.targets = []
    start = np.array([-4.0, 0.0, 1.0], dtype=np.float32)
    states = [
        SimpleNamespace(
            position=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            timestamp_us=0,
            armed=False,
            offboard=False,
        ),
        SimpleNamespace(
            position=start.copy(),
            velocity=np.zeros(3, dtype=np.float32),
            timestamp_us=100_000,
            armed=True,
            offboard=True,
        ),
    ]

    def get_state():
        return states.pop(0)

    backend.get_state = get_state
    backend._set_position_target = lambda target: backend.targets.append(target.copy())
    backend._publish_setpoint = lambda: None
    backend._vehicle_command = lambda command, *params: backend.commands.append(
        (command, params)
    )

    backend._move_to_position(
        start,
        "moving to test target",
        settle_radius=0.18,
        settle_speed=0.25,
    )

    assert backend.commands == [(176, (1.0, 6.0)), (400, (1.0,))]
    assert len(backend.targets) == 2


def test_go_to_start_recovers_low_vehicle_before_start():
    backend = object.__new__(Px4RosBackend)
    backend.reset_safe_altitude = 1.2
    backend.targets = []
    backend.state = SimpleNamespace(
        position=np.array([-2.0, -2.0, -0.01], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        timestamp_us=0,
        armed=True,
        offboard=True,
    )
    start = np.array([-4.4, 0.57, 0.92], dtype=np.float32)

    def get_state():
        return backend.state

    def move_to_position(target, *_args, **_kwargs):
        backend.targets.append(target.copy())
        backend.state = SimpleNamespace(
            position=target.copy(),
            velocity=np.zeros(3, dtype=np.float32),
            timestamp_us=0,
            armed=True,
            offboard=True,
        )

    backend.get_state = get_state
    backend._move_to_position = move_to_position

    backend._go_to_start(start)

    np.testing.assert_allclose(backend.targets[0], [-2.0, -2.0, 1.2])
    np.testing.assert_allclose(backend.targets[1], [-4.4, 0.57, 1.2])
    np.testing.assert_allclose(backend.targets[2], start)


def test_activate_offboard_uses_safe_hold_when_low():
    backend = object.__new__(Px4RosBackend)
    backend.setup_timeout = 1.0
    backend.synchronous = False
    backend.reset_safe_altitude = 1.2
    backend.VehicleCommand = SimpleNamespace(VEHICLE_CMD_DO_SET_MODE=176)
    backend.targets = []
    backend.commands = []
    states = [
        SimpleNamespace(
            position=np.array([0.0, 0.0, -0.01], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            timestamp_us=0,
            armed=True,
            offboard=False,
        ),
        SimpleNamespace(
            position=np.array([0.0, 0.0, 1.2], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            timestamp_us=100_000,
            armed=True,
            offboard=False,
        ),
        SimpleNamespace(
            position=np.array([0.0, 0.0, 1.2], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            timestamp_us=200_000,
            armed=True,
            offboard=True,
        ),
    ]

    def get_state():
        return states.pop(0)

    backend.get_state = get_state
    backend._set_position_target = lambda target: backend.targets.append(target.copy())
    backend._publish_setpoint = lambda: None
    backend._vehicle_command = lambda command, *params: backend.commands.append(
        (command, params)
    )

    backend._activate_offboard()

    np.testing.assert_allclose(backend.targets[0], [0.0, 0.0, 1.2])
    assert backend.commands == [(176, (1.0, 6.0))]
