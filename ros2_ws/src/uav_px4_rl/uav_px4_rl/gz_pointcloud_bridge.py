"""Bridge Gazebo PointCloudPacked messages to ROS 2 PointCloud2."""

import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField

from gz.msgs10.pointcloud_packed_pb2 import PointCloudPacked
from gz.transport13 import Node as GzNode


GZ_TO_ROS_FIELD_TYPES = {
    PointCloudPacked.Field.INT8: PointField.INT8,
    PointCloudPacked.Field.UINT8: PointField.UINT8,
    PointCloudPacked.Field.INT16: PointField.INT16,
    PointCloudPacked.Field.UINT16: PointField.UINT16,
    PointCloudPacked.Field.INT32: PointField.INT32,
    PointCloudPacked.Field.UINT32: PointField.UINT32,
    PointCloudPacked.Field.FLOAT32: PointField.FLOAT32,
    PointCloudPacked.Field.FLOAT64: PointField.FLOAT64,
}


def _header_value(gz_msg, key, default=""):
    for item in gz_msg.header.data:
        if item.key == key:
            return item.value[0] if item.value else default
    return default


def convert_pointcloud(gz_msg):
    ros_msg = PointCloud2()
    ros_msg.header.stamp.sec = int(gz_msg.header.stamp.sec)
    ros_msg.header.stamp.nanosec = int(gz_msg.header.stamp.nsec)
    ros_msg.header.frame_id = _header_value(gz_msg, "frame_id")
    ros_msg.height = int(gz_msg.height)
    ros_msg.width = int(gz_msg.width)
    ros_msg.is_bigendian = bool(gz_msg.is_bigendian)
    ros_msg.point_step = int(gz_msg.point_step)
    ros_msg.row_step = int(gz_msg.row_step)
    ros_msg.is_dense = bool(gz_msg.is_dense)
    ros_msg.fields = [
        PointField(
            name=field.name,
            offset=int(field.offset),
            datatype=GZ_TO_ROS_FIELD_TYPES[int(field.datatype)],
            count=int(field.count),
        )
        for field in gz_msg.field
    ]
    ros_msg.data = bytes(gz_msg.data)
    return ros_msg


class GzPointCloudBridge(Node):
    """One-way Gazebo to ROS 2 point cloud bridge."""

    def __init__(self, gz_topic: str, ros_topic: str):
        super().__init__("gz_pointcloud_bridge")
        self.gz_topic = gz_topic
        self.ros_topic = ros_topic
        self.gz_node = GzNode()
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.publisher = self.create_publisher(PointCloud2, self.ros_topic, qos)
        self.gz_node.subscribe(PointCloudPacked, self.gz_topic, self._on_pointcloud)
        self.get_logger().info(
            f"Bridging Gazebo {self.gz_topic} to ROS 2 {self.ros_topic}"
        )

    def _on_pointcloud(self, gz_msg):
        self.publisher.publish(convert_pointcloud(gz_msg))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gz-topic", default="/x500/lidar/points")
    parser.add_argument("--ros-topic", default="/x500/lidar/points")
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    rclpy.init()
    node = GzPointCloudBridge(args.gz_topic, args.ros_topic)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
