# uav_ppo_px4_3d_lidar

这是 **3B-lite 多电线 3D LiDAR 分支**。项目由 `D:\UAV\uav_ppo_px4_demo` 复制而来，当前目录用于独立开发，不应再直接修改原 1B 项目。

本阶段目标是把原来的单电线真值几何避障升级为：

```text
PPO -> Gymnasium Px4GazeboWireEnv
    -> 3D LiDAR perception features observation
    -> ROS 2 Offboard -> PX4 SITL -> Gazebo x500 + 多电线场景
    -> PX4 里程计反馈 + 仿真真值 reward/collision/info
```

## 3B-lite 任务定义

| 项目 | 定义 |
| --- | --- |
| 场景 | 默认 3 根长度 `2.4 m` 的电线，`fixed` 用于链路排障，`random` 每局随机生成多电线 |
| 动作 | PPO 输出 `[vx, vy, vz]` 归一化值，映射为 ENU 最大速度 `[1.5, 1.5, 1.0] m/s` |
| 观测 | 目标相对位置 3、实测速度 3、LiDAR perception features 13、上一动作 3，共 22 维 |
| 策略输入 | 不再直接包含真值 `wire.closest` 或真值 `wire.distance` |
| 奖励/终止/info | 仍使用仿真真值距离，用于训练反馈、碰撞判定、日志和诊断 |
| 当前目标 | 安全绕开多根电线，不主动设计从两根电线之间穿过的窄缝任务 |
| 逃课约束 | reward 中加入高度、走廊、边界和路径效率相关约束，抑制飞高或绕大圈 |

reward 使用真值不是作弊：reward 是训练环境反馈，部署时策略网络只能看到 observation。本分支的 observation 已经从真值电线几何切换到 LiDAR perception features。

## LiDAR Features

`LiDARFeatureExtractor.feature_dim == 13`，特征含义如下：

1. `nearest_obstacle_distance`
2. `nearest_obstacle_direction_x`
3. `nearest_obstacle_direction_y`
4. `nearest_obstacle_direction_z`
5. `front_min_range`
6. `front_left_min_range`
7. `front_right_min_range`
8. `left_min_range`
9. `right_min_range`
10. `up_min_range`
11. `down_min_range`
12. `valid_point_ratio`
13. `detection_confidence`

点云第一版假设已经在无人机局部坐标或 LiDAR 坐标中。完整点云不会端到端喂给 PPO，而是先压缩成固定长度特征，继续使用 MLP PPO。

`SyntheticLidarSimulator` 只用于无 ROS/PX4 时的测试和诊断 fallback：它沿多根电线真值采样少量点、加入轻量噪声和 dropout，再按 `max_range` 过滤。它不代表真实 3D LiDAR，也不包含扫描线、ring 或 intensity 模型。

## 目录结构

```text
uav_ppo_px4_3d_lidar/
  README.md
  requirements.txt
  docs/
    USAGE_GUIDE_UBUNTU22.md
  ros2_ws/src/uav_px4_rl/
    launch/sim_bridge.launch.py
    worlds/wire_training_world.sdf
    uav_px4_rl/
      backend.py
      env.py
      geometry.py
      perception.py
      px4_backend.py
      rewards.py
      scenario.py
  train/train_ppo_px4.py
  eval/evaluate_online.py
  tests/
```

## 核心源码

| 文件 | 作用 |
| --- | --- |
| `scenario.py` | `WireSegment`、多电线 `Scenario`、固定/随机 `ScenarioSampler` |
| `geometry.py` | 单线距离、`point_to_segments_distance`、多线最近距离和所有距离 |
| `perception.py` | `LiDARFeatureExtractor` 与 `SyntheticLidarSimulator` |
| `env.py` | 22 维 observation，LiDAR 特征获取，真值 reward/info |
| `rewards.py` | 多线安全距离惩罚、高度/走廊/边界约束 |
| `backend.py` | 无 ROS/PX4 的诊断后端 |
| `px4_backend.py` | PX4 Offboard 后端，多电线模型摆放，预留 PointCloud2 接口 |
| `wire_training_world.sdf` | 默认包含 `wire_obstacle`、`wire_obstacle_1`、`wire_obstacle_2` 三个可移动电线模型 |

## 安装与构建

正式闭环运行建议使用：

```text
Ubuntu 22.04
ROS 2 Humble
PX4 Autopilot v1.16.2
Gazebo Harmonic
Micro XRCE-DDS Agent v2.x
Python 3.10
```

准备 `px4_msgs`：

```bash
cd ~/uav_ppo_px4_3d_lidar/ros2_ws/src
git clone --branch release/1.16 https://github.com/PX4/px4_msgs.git
cd ../..
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

创建 Python 环境：

```bash
cd ~/uav_ppo_px4_3d_lidar
/usr/bin/python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 在线训练

启动 PX4/Gazebo 栈后运行：

```bash
python train/train_ppo_px4.py \
  --scenario random \
  --num-wires 3 \
  --perception lidar \
  --timesteps 300000
```

默认模型名：

```text
models/ppo_px4_3d_lidar_multiwire.zip
```

如果已经 bridge 出真实 3D LiDAR 的 `sensor_msgs/msg/PointCloud2` topic，可传入：

```bash
python train/train_ppo_px4.py --lidar-topic /your/pointcloud2/topic
```

没有真实 LiDAR topic 时，PX4 后端返回 `None`，环境使用空 LiDAR 特征；只有 `KinematicDiagnosticBackend` 本地测试会启用 `SyntheticLidarSimulator` fallback。

## 在线评估

```bash
python eval/evaluate_online.py \
  --model models/ppo_px4_3d_lidar_multiwire.zip \
  --episodes 10 \
  --num-wires 3 \
  --perception lidar
```

CSV 和 summary 记录包括：

- `num_wires`
- `min_true_wire_distance`
- `nearest_wire_index`
- `lidar_confidence`
- `goal_distance`
- `collision`
- `reached_goal`

## 测试

纯 Python 测试不依赖 ROS/PX4：

```bash
python -m pytest
```

测试覆盖固定/随机多电线场景、有限线段距离、多线最近距离与所有距离、LiDAR 特征、synthetic 点云、环境 observation shape，以及诊断后端 reset/step。

## 真实 3D LiDAR 接入位置

当前真实 LiDAR 接口是 `Px4RosBackend(lidar_topic=...)` 和 `get_lidar_points()/latest_lidar_points`。真正接入 Gazebo 3D LiDAR 还需要：

1. 在 Gazebo/PX4 `x500` 模型中挂载 3D LiDAR 传感器插件。
2. 将 Gazebo 点云 bridge 为 ROS 2 `sensor_msgs/msg/PointCloud2` topic。
3. 训练或评估时把该 topic 传给 `--lidar-topic`。
4. 必要时扩展 `sim_bridge.launch.py` 或外部 `ros_gz_bridge` 配置。

不要在代码中写死用户本机 PX4 路径。PX4 路径继续通过环境变量和启动脚本管理。
