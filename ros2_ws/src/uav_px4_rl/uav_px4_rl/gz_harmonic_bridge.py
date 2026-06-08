"""ROS services backed directly by Gazebo Harmonic transport."""

import argparse

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose
from rosgraph_msgs.msg import Clock as RosClock

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.clock_pb2 import Clock as GzClock
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.world_control_pb2 import WorldControl
from gz.transport13 import Node as GzNode


class GzHarmonicBridge(Node):
    """Expose the small Gazebo API surface used by the PX4 backend."""

    def __init__(self, world_name: str, timeout_ms: int):
        super().__init__("wire_world_bridge")
        self.world_name = world_name
        self.timeout_ms = int(timeout_ms)
        self.gz_service_node = GzNode()
        self.gz_clock_node = GzNode()

        self.clock_pub = self.create_publisher(RosClock, "/clock", 10)
        self.gz_clock_node.subscribe(
            GzClock, f"/world/{self.world_name}/clock", self._publish_clock
        )

        self.create_service(
            SetEntityPose,
            f"/world/{self.world_name}/set_pose",
            self._set_entity_pose,
        )
        self.create_service(
            ControlWorld,
            f"/world/{self.world_name}/control",
            self._control_world,
        )

    def _publish_clock(self, gz_clock):
        message = RosClock()
        message.clock.sec = int(gz_clock.sim.sec)
        message.clock.nanosec = int(gz_clock.sim.nsec)
        self.clock_pub.publish(message)

    def _set_entity_pose(self, request, response):
        gz_request = Pose()
        if request.entity.id:
            gz_request.id = int(request.entity.id)
        if request.entity.name:
            gz_request.name = request.entity.name
        elif request.entity.type != Entity.NONE:
            self.get_logger().warning("set_pose request has entity type but no name/id")

        gz_request.position.x = float(request.pose.position.x)
        gz_request.position.y = float(request.pose.position.y)
        gz_request.position.z = float(request.pose.position.z)
        gz_request.orientation.x = float(request.pose.orientation.x)
        gz_request.orientation.y = float(request.pose.orientation.y)
        gz_request.orientation.z = float(request.pose.orientation.z)
        gz_request.orientation.w = float(request.pose.orientation.w)

        ok, gz_response = self.gz_service_node.request(
            f"/world/{self.world_name}/set_pose",
            gz_request,
            Pose,
            Boolean,
            self.timeout_ms,
        )
        if not ok:
            self.get_logger().warning(
                f"Gazebo transport timed out while setting pose after "
                f"{self.timeout_ms} ms."
            )
        elif not gz_response.data:
            self.get_logger().warning("Gazebo rejected set_pose request.")
        response.success = bool(ok and gz_response.data)
        return response

    def _control_world(self, request, response):
        source = request.world_control
        gz_request = WorldControl()
        gz_request.pause = bool(source.pause)
        if source.step:
            gz_request.step = True
        if source.multi_step:
            gz_request.multi_step = int(source.multi_step)
        if source.reset.all or source.reset.time_only or source.reset.model_only:
            gz_request.reset.all = bool(source.reset.all)
            gz_request.reset.time_only = bool(source.reset.time_only)
            gz_request.reset.model_only = bool(source.reset.model_only)
        if source.seed:
            gz_request.seed = int(source.seed)
        if source.run_to_sim_time.sec or source.run_to_sim_time.nanosec:
            gz_request.run_to_sim_time.sec = int(source.run_to_sim_time.sec)
            gz_request.run_to_sim_time.nsec = int(source.run_to_sim_time.nanosec)

        ok, gz_response = self.gz_service_node.request(
            f"/world/{self.world_name}/control",
            gz_request,
            WorldControl,
            Boolean,
            self.timeout_ms,
        )
        if not ok:
            self.get_logger().warning(
                f"Gazebo transport timed out while controlling the world after "
                f"{self.timeout_ms} ms."
            )
        elif not gz_response.data:
            self.get_logger().warning("Gazebo rejected world control request.")
        response.success = bool(ok and gz_response.data)
        return response


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", default="wire_training_world")
    parser.add_argument("--timeout-ms", type=int, default=5000)
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    rclpy.init()
    node = GzHarmonicBridge(args.world, args.timeout_ms)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
