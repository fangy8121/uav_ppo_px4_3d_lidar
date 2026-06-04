"""在 PX4 边界处使用的坐标系转换。

学习环境与 Gazebo 场景均采用 ENU(东北天)坐标系表示。
PX4 本地位置与 Offboard轨迹期望点则采用 NED(北东地)坐标系表示。"""
import numpy as np


def enu_to_ned(vector): 
    """将ENU [east, north, up] 转换为 NED[north, east, down]."""
    east, north, up = np.asarray(vector, dtype=np.float32)
    return np.array([north, east, -up], dtype=np.float32)


def ned_to_enu(vector):
    north, east, down = np.asarray(vector, dtype=np.float32)
    return np.array([east, north, -down], dtype=np.float32)


def nan_vector(): #返回一个未设置的三维PX4设定点字段。
    return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

