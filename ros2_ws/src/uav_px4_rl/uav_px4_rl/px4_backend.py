"""ROS 2 backend that closes the loop through PX4 SITL and Gazebo."""

from threading import Lock, Thread
import time

import numpy as np

from .backend import VehicleState
from .frames import enu_to_ned, nan_vector, ned_to_enu
from .geometry import segment_model_pose


class Px4RosBackend:
    """驱动 PX4 Offboard期望点并观测仿真的 x500 无人机.

    Gazebo 必须正在运行提供的world文件
    PX4 SITL 必须已生成一架 x500 无人机
    Micro XRCE-DDS Agent 必须已连接，
    且 Gazebo bridge 必须已导出该世界的 set_pose 和控制服务。
    """

    def __init__(
        self,
        world_name="wire_training_world",
        synchronous=False,
        physics_step_size=0.004,
        heartbeat_hz=20.0,
        setup_timeout=35.0,
        service_timeout=7.0,
        service_retries=12,
        service_retry_delay=0.1,
        reset_safe_altitude=1.2,
        reset_stale_after=0.5,
        lidar_topic=None,
        wire_model_names=None,
    ):
        try:
            import rclpy  # ROS 2 库
            from geometry_msgs.msg import Pose
            from px4_msgs.msg import (
                OffboardControlMode,  # 控制模式
                TrajectorySetpoint,   # 轨迹期望点
                VehicleCommand,       # 无人机命令
                VehicleOdometry,      # 里程计数据
                VehicleStatus,        # 无人机状态
            )
            from rclpy.executors import MultiThreadedExecutor  # ROS2线程执行器
            from rclpy.node import Node  # ROS2节点
            from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy  # QoS策略
            from ros_gz_interfaces.msg import Entity  # 世界实体
            from ros_gz_interfaces.srv import ControlWorld, SetEntityPose  # 世界控制服务
        except ImportError as exc:
            raise RuntimeError(
                "Px4RosBackend requires a sourced ROS 2 workspace with px4_msgs "
                "and ros_gz_interfaces installed."
            ) from exc
        try:
            from sensor_msgs.msg import PointCloud2
            from sensor_msgs_py import point_cloud2
        except ImportError:
            PointCloud2 = None
            point_cloud2 = None

        self.rclpy = rclpy 
        self.Pose = Pose 
        self.OffboardControlMode = OffboardControlMode 
        self.TrajectorySetpoint = TrajectorySetpoint
        self.VehicleCommand = VehicleCommand
        self.VehicleStatus = VehicleStatus
        self.Entity = Entity 
        self.ControlWorld = ControlWorld 
        self.SetEntityPose = SetEntityPose 
        self.PointCloud2 = PointCloud2
        self.point_cloud2 = point_cloud2
        self.synchronous = bool(synchronous) # 是否同步执行
        self.physics_step_size = float(physics_step_size) 
        self.setup_timeout = float(setup_timeout)
        self.service_timeout = float(service_timeout)
        self.service_retries = max(1, int(service_retries))
        self.service_retry_delay = float(service_retry_delay)
        self.reset_safe_altitude = float(reset_safe_altitude)
        self.reset_stale_after = float(reset_stale_after)
        self.lidar_topic = lidar_topic
        self.wire_model_names = tuple(
            wire_model_names or ("wire_obstacle", "wire_obstacle_1", "wire_obstacle_2")
        )
        self._lock = Lock()
        self._last_odometry = None 
        self._last_status = None
        self._latest_lidar_points = None
        self._mode = "position"
        self._position_ned = nan_vector()
        self._velocity_ned = nan_vector()
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()  # 初始化ROS2

        self.node = Node("px4_gazebo_wire_backend")
        px4_qos = QoSProfile( #只保留最新一条消息；允许丢包；不关心旧消息。
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1, 
        )
        self.offboard_publisher = self.node.create_publisher( # 使用Offboard控制模式
            OffboardControlMode, "/fmu/in/offboard_control_mode", px4_qos
        )
        self.setpoint_publisher = self.node.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", px4_qos
        )
        self.command_publisher = self.node.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", px4_qos
        )
        self.node.create_subscription( #订阅无人机位置速度
            VehicleOdometry, "/fmu/out/vehicle_odometry", self._on_odometry, px4_qos
        )
        self.node.create_subscription( #订阅无人机飞行状态
            VehicleStatus, "/fmu/out/vehicle_status_v1", self._on_status, px4_qos
        )
        self.node.create_subscription( #兼容不带消息版本后缀的PX4话题
            VehicleStatus, "/fmu/out/vehicle_status", self._on_status, px4_qos
        )
        if lidar_topic is not None:
            if PointCloud2 is None or point_cloud2 is None:
                raise RuntimeError(
                    "lidar_topic requires sensor_msgs and sensor_msgs_py in the ROS 2 environment."
                )
            lidar_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self.node.create_subscription(
                PointCloud2, lidar_topic, self._on_lidar, lidar_qos
            )
        self.set_pose_client = self.node.create_client( #直接设置无人机位置
            SetEntityPose, f"/world/{world_name}/set_pose"
        )
        self.control_client = self.node.create_client( #控制Gazebo世界
            ControlWorld, f"/world/{world_name}/control"
        )
        self.node.create_timer(1.0 / heartbeat_hz, self._publish_setpoint)

        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        try:
            self._wait_for_stack()
        except Exception:
            self.close()
            raise

    def prepare_episode(self, scenario):
        """Move to a random start safely, then put the episode wire in place."""
        if self.synchronous:
            self._control_world(pause=False)
        wire_endpoints = self._scenario_wire_endpoints(scenario)
        wire_count = len(wire_endpoints)
        for index in range(max(wire_count, len(self.wire_model_names))):
            self._set_model_pose(
                self._wire_model_name(index),
                np.array([0.0, 0.0, -5.0], dtype=np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            )
        self._set_model_pose(
            "goal_marker",
            scenario.goal,
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        self._set_model_pose(
            "start_marker",
            scenario.start,
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        self._activate_offboard()
        self._go_to_start(scenario.start)
        for index, (start, end) in enumerate(wire_endpoints):
            center, quaternion, _ = segment_model_pose(start, end)
            self._set_model_pose(self._wire_model_name(index), center, quaternion)
        self._set_position_target(scenario.start)
        time.sleep(0.15)
        if self.synchronous:
            self._control_world(pause=True)
        return self.get_state()

    def advance_velocity(self, velocity_enu, duration_seconds):
        """Apply one policy action and return measured PX4 state after it."""
        start_timestamp = self.get_state().timestamp_us
        self._set_velocity_target(np.asarray(velocity_enu, dtype=np.float32))
        if self.synchronous:
            steps = max(1, round(float(duration_seconds) / self.physics_step_size))
            self._control_world(pause=True, multi_step=steps)
            deadline = time.monotonic() + 2.0
            expected = start_timestamp + int(duration_seconds * 1_000_000 * 0.8)
            while self.get_state().timestamp_us < expected and time.monotonic() < deadline:
                time.sleep(0.002)
        else:
            time.sleep(float(duration_seconds))
        return self.get_state()

    def get_state(self):
        with self._lock:
            odometry = self._last_odometry
            status = self._last_status
        if odometry is None:
            raise RuntimeError("PX4 vehicle odometry has not been received.")
        position = ned_to_enu(odometry.position)
        velocity = ned_to_enu(odometry.velocity)
        armed = bool(
            status is not None
            and status.arming_state == self.VehicleStatus.ARMING_STATE_ARMED
        )
        offboard = bool(
            status is not None
            and status.nav_state == self.VehicleStatus.NAVIGATION_STATE_OFFBOARD
        )
        return VehicleState(position, velocity, int(odometry.timestamp), armed, offboard)

    @property
    def latest_lidar_points(self):
        with self._lock:
            if self._latest_lidar_points is None:
                return None
            return self._latest_lidar_points.copy()

    def get_lidar_points(self):
        return self.latest_lidar_points

    def close(self):
        try:
            if self._last_odometry is not None:
                try:
                    self._set_velocity_target(np.zeros(3, dtype=np.float32))
                    time.sleep(0.1)
                except Exception as exc:
                    if self.rclpy.ok():
                        self.node.get_logger().warning(
                            f"Failed to publish stop setpoint during close: {exc}"
                        )
        finally:
            self.executor.shutdown()
            self.thread.join(timeout=2.0)
            self.node.destroy_node()
            if self._owns_rclpy and self.rclpy.ok():
                self.rclpy.shutdown()

    def _wait_for_stack(self):
        deadline = time.monotonic() + self.setup_timeout
        while self._last_odometry is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if self._last_odometry is None:
            raise TimeoutError("No /fmu/out/vehicle_odometry received from PX4.")
        for client, label in (
            (self.set_pose_client, "Gazebo set_pose"),
            (self.control_client, "Gazebo control"),
        ):
            remaining = max(0.0, deadline - time.monotonic())
            if not client.wait_for_service(timeout_sec=remaining):
                raise TimeoutError(f"{label} service is not available through ros_gz_bridge.")

    def _on_odometry(self, message):
        with self._lock:
            self._last_odometry = message

    def _on_status(self, message):
        with self._lock:
            self._last_status = message

    def _on_lidar(self, message):
        points = self._pointcloud2_to_xyz(message)
        with self._lock:
            self._latest_lidar_points = points

    def _pointcloud2_to_xyz(self, message):
        raw_points = self.point_cloud2.read_points(
            message, field_names=("x", "y", "z"), skip_nans=True
        )
        if hasattr(raw_points, "dtype") and raw_points.dtype.names:
            points = np.stack(
                [raw_points["x"], raw_points["y"], raw_points["z"]], axis=-1
            )
        else:
            points = np.asarray(list(raw_points), dtype=np.float32)
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(points, dtype=np.float32).reshape((-1, 3))

    def _wire_model_name(self, index):
        if index < len(self.wire_model_names):
            return self.wire_model_names[index]
        return f"wire_obstacle_{index}"

    def _scenario_wire_endpoints(self, scenario):
        if hasattr(scenario, "wires"):
            return tuple((wire.start, wire.end) for wire in scenario.wires)
        return ((scenario.wire_a, scenario.wire_b),)

    def _timestamp(self):
        return int(self.node.get_clock().now().nanoseconds / 1000)

    def _publish_setpoint(self):
        with self._lock:
            mode = self._mode
            position = self._position_ned.copy()
            velocity = self._velocity_ned.copy()
        control_mode = self.OffboardControlMode()
        control_mode.timestamp = self._timestamp()
        control_mode.position = mode == "position"
        control_mode.velocity = mode == "velocity"
        self.offboard_publisher.publish(control_mode)

        setpoint = self.TrajectorySetpoint()
        setpoint.timestamp = self._timestamp()
        setpoint.position = position.tolist()
        setpoint.velocity = velocity.tolist()
        setpoint.yaw = float("nan")
        self.setpoint_publisher.publish(setpoint)

    def _set_position_target(self, position_enu):
        with self._lock:
            self._mode = "position"
            self._position_ned = enu_to_ned(position_enu)
            self._velocity_ned = nan_vector()
        self._publish_setpoint()

    def _set_velocity_target(self, velocity_enu):
        with self._lock:
            self._mode = "velocity"
            self._position_ned = nan_vector()
            self._velocity_ned = enu_to_ned(velocity_enu)
        self._publish_setpoint()

    def _activate_offboard(self):
        hold_position = self.get_state().position.copy()
        if hold_position[2] < self.reset_safe_altitude:
            hold_position[2] = self.reset_safe_altitude
        self._set_position_target(hold_position)
        if self.synchronous:
            self._control_world(pause=False)
        deadline = time.monotonic() + self.setup_timeout
        next_command_time = 0.0
        last_advance_time = time.monotonic()
        last_timestamp = None
        while time.monotonic() < deadline:
            self._publish_setpoint()
            state = self.get_state()
            if last_timestamp is None or state.timestamp_us != last_timestamp:
                last_timestamp = state.timestamp_us
                last_advance_time = time.monotonic()
            if state.armed and state.offboard:
                return
            now = time.monotonic()
            if self.synchronous and now - last_advance_time > self.reset_stale_after:
                self._control_world(pause=False)
                self._set_position_target(hold_position)
                last_advance_time = now
            if now >= next_command_time:
                if not state.offboard:
                    self._vehicle_command(self.VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                if not state.armed:
                    self._vehicle_command(self.VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                next_command_time = now + 0.5
            time.sleep(0.05)
        state = self.get_state()
        raise TimeoutError(
            "PX4 failed to enter armed Offboard mode "
            f"(armed={state.armed}, offboard={state.offboard})."
        )

    def _go_to_start(self, start):
        start = np.asarray(start, dtype=np.float32)
        state = self.get_state()
        safe_altitude = max(float(start[2]), self.reset_safe_altitude)

        if state.position[2] < safe_altitude - 0.2:
            climb_target = state.position.copy()
            climb_target[2] = safe_altitude
            self._move_to_position(
                climb_target,
                "recovering to safe altitude before episode start",
                settle_radius=0.28,
                settle_speed=0.35,
            )

        cruise_target = start.copy()
        cruise_target[2] = safe_altitude
        self._move_to_position(
            cruise_target,
            "moving above sampled episode start",
            settle_radius=0.28,
            settle_speed=0.35,
        )
        self._move_to_position(
            start,
            "settling at sampled episode start",
            settle_radius=0.18,
            settle_speed=0.25,
        )

    def _move_to_position(self, target, operation, settle_radius, settle_speed):
        target = np.asarray(target, dtype=np.float32)
        self._set_position_target(target)
        if self.synchronous:
            self._control_world(pause=False)

        deadline = time.monotonic() + self.setup_timeout
        next_command_time = 0.0
        last_advance_time = time.monotonic()
        last_timestamp = None
        last_state = None
        last_distance = float("inf")
        last_speed = float("inf")
        while time.monotonic() < deadline:
            self._publish_setpoint()
            state = self.get_state()
            last_state = state
            if last_timestamp is None or state.timestamp_us != last_timestamp:
                last_timestamp = state.timestamp_us
                last_advance_time = time.monotonic()
            distance = float(np.linalg.norm(state.position - target))
            speed = float(np.linalg.norm(state.velocity))
            last_distance = distance
            last_speed = speed
            if distance < settle_radius and speed < settle_speed:
                return
            now = time.monotonic()
            if self.synchronous and now - last_advance_time > self.reset_stale_after:
                self._control_world(pause=False)
                self._set_position_target(target)
                last_advance_time = now
            if (not state.armed or not state.offboard) and now >= next_command_time:
                if not state.offboard:
                    self._vehicle_command(
                        self.VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                        1.0,
                        6.0,
                    )
                if not state.armed:
                    self._vehicle_command(
                        self.VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                        1.0,
                    )
                next_command_time = now + 0.5
                self._set_position_target(target)
            time.sleep(0.05)
        if last_state is None:
            raise TimeoutError(f"PX4 failed to provide state while {operation}.")
        target_text = np.array2string(target, precision=2)
        position = np.array2string(last_state.position, precision=2)
        raise TimeoutError(
            f"PX4 failed while {operation} "
            f"(target={target_text}, position={position}, distance={last_distance:.2f}, "
            f"speed={last_speed:.2f}, armed={last_state.armed}, "
            f"offboard={last_state.offboard})."
        )

    def _vehicle_command(self, command, param1=0.0, param2=0.0):
        message = self.VehicleCommand()
        message.timestamp = self._timestamp()
        message.param1 = float(param1)
        message.param2 = float(param2)
        message.command = command
        message.target_system = 1
        message.target_component = 1
        message.source_system = 1
        message.source_component = 1
        message.from_external = True
        self.command_publisher.publish(message)

    def _set_model_pose(self, model_name, position, quaternion):
        def make_request():
            request = self.SetEntityPose.Request()
            request.entity.name = model_name
            request.entity.type = self.Entity.MODEL
            request.pose.position.x = float(position[0])
            request.pose.position.y = float(position[1])
            request.pose.position.z = float(position[2])
            request.pose.orientation.x = float(quaternion[0])
            request.pose.orientation.y = float(quaternion[1])
            request.pose.orientation.z = float(quaternion[2])
            request.pose.orientation.w = float(quaternion[3])
            return request

        self._call_gazebo_service(
            self.set_pose_client,
            make_request,
            f"setting pose for {model_name}",
        )

    def _control_world(self, pause, multi_step=0):
        def make_request():
            request = self.ControlWorld.Request()
            request.world_control.pause = bool(pause)
            request.world_control.multi_step = int(multi_step)
            return request

        self._call_gazebo_service(
            self.control_client,
            make_request,
            "controlling Gazebo simulation time",
        )

    def _call_gazebo_service(self, client, make_request, operation):
        response = None
        last_error = None
        for attempt in range(1, self.service_retries + 1):
            future = client.call_async(make_request())
            try:
                response = self._wait_future(
                    future,
                    operation,
                    timeout=self.service_timeout,
                )
            except (RuntimeError, TimeoutError) as exc:
                last_error = exc
                if attempt < self.service_retries:
                    self.node.get_logger().warning(
                        f"Gazebo service failed while {operation}: {exc}; "
                        f"retrying {attempt}/{self.service_retries}."
                    )
                    time.sleep(self.service_retry_delay)
                    continue
                raise RuntimeError(
                    f"Gazebo service failed while {operation}: {exc}"
                ) from exc
            if not hasattr(response, "success") or response.success:
                return response
            if attempt < self.service_retries:
                self.node.get_logger().warning(
                    f"Gazebo rejected request while {operation}; "
                    f"retrying {attempt}/{self.service_retries}."
                )
                time.sleep(self.service_retry_delay)

        message = getattr(response, "status_message", "")
        detail = f" ({message})" if message else ""
        if response is None and last_error is not None:
            raise RuntimeError(f"Gazebo service failed while {operation}: {last_error}.")
        raise RuntimeError(f"Gazebo rejected request while {operation}{detail}.")

    def _wait_future(self, future, operation, timeout=None):
        deadline = time.monotonic() + (
            self.setup_timeout if timeout is None else float(timeout)
        )
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.005)
        if not future.done():
            raise TimeoutError(f"Timed out while {operation}.")
        response = future.result()
        if response is None:
            raise RuntimeError(f"ROS service returned no response while {operation}.")
        return response
