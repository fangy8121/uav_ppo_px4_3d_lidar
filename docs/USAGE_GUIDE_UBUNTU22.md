# Ubuntu 22.04 使用指南

## 1. 项目目标

本项目是 **3B-lite：多电线 + 3D LiDAR perception features** 分支。策略 observation 不再直接使用真值电线几何，而是使用固定长度 LiDAR 感知特征；reward、collision 和 info 仍可使用仿真真值距离作为训练反馈和诊断数据。

```text
PPO policy
  -> Px4GazeboWireEnv.step(action)
  -> ROS 2 /fmu/in/* Offboard messages
  -> PX4 SITL flight controller
  -> Gazebo Harmonic x500 physics + multi-wire world
  -> PX4 VehicleOdometry + optional PointCloud2
  -> LiDAR features observation, truth-based reward, termination
```

reward 使用真值不是作弊，因为 reward 是训练时环境反馈；部署时网络只接收 observation。本分支已经移除了 observation 中的真值 `wire.closest` 和 `wire.distance`。

## 2. 固定软件组合

```text
Ubuntu 22.04 LTS
ROS 2 Humble
Gazebo Harmonic
PX4 Autopilot v1.16.2
px4_msgs release/1.16
Micro XRCE-DDS Agent v2.x
Python 3.10
```

需要安装：

```bash
sudo apt update
sudo apt install ros-humble-desktop ros-dev-tools python3-colcon-common-extensions python3-venv
sudo apt install ros-humble-ros-gzharmonic
```

## 3. 文件作用

| 文件 | 作用 |
| --- | --- |
| `train/train_ppo_px4.py` | PPO 在线训练入口，支持 `--num-wires`、`--perception`、`--lidar-topic` |
| `eval/evaluate_online.py` | 在线评估入口，记录多线距离和 LiDAR confidence |
| `uav_px4_rl/scenario.py` | `WireSegment`、多电线 `Scenario`、固定/随机采样 |
| `uav_px4_rl/geometry.py` | 单线和多线距离，返回最近线与所有线距离 |
| `uav_px4_rl/perception.py` | `LiDARFeatureExtractor` 和 `SyntheticLidarSimulator` |
| `uav_px4_rl/env.py` | 22 维 observation，真值 reward/info，synthetic fallback 管理 |
| `uav_px4_rl/rewards.py` | progress、所有线安全距离、高度、走廊、边界、成功/碰撞奖励 |
| `uav_px4_rl/backend.py` | 无 ROS/PX4 的本地诊断后端 |
| `uav_px4_rl/px4_backend.py` | PX4 Offboard 后端，多线模型摆放，PointCloud2 订阅接口 |
| `launch/sim_bridge.launch.py` | 启动 Gazebo world 和当前服务 bridge，预留 LiDAR topic 参数 |
| `worlds/wire_training_world.sdf` | 默认 3 根可移动电线模型和起点/目标标记 |

## 4. 任务定义

| 项目 | 内容 |
| --- | --- |
| 场景 | 默认 `num_wires=3`，每根线长 `2.4 m`，随机线大致位于 start-goal 之间 |
| 动作 | `[vx, vy, vz]` 归一化速度指令，最大 ENU 速度 `[1.5, 1.5, 1.0] m/s` |
| 观测 | goal relative position 3、velocity 3、LiDAR features 13、previous action 3，共 22 维 |
| 成功 | 机体中心到目标小于 `RewardConfig.goal_radius`，默认 `0.35 m` |
| 碰撞 | 最近电线真值距离小于 `RewardConfig.collision_radius`，默认 `0.35 m` |
| 安全惩罚 | 对所有电线计算 `max(0, safe_distance - distance_i)` 并聚合惩罚 |
| 逃课抑制 | 高度偏差、start-goal 走廊偏差、边界越界和每步惩罚 |

当前阶段不主动设计窄缝穿越任务，目标是安全绕开多根电线。

## 5. LiDAR Features

`LiDARFeatureExtractor.feature_dim == 13`：

| 序号 | 特征 |
| --- | --- |
| 1 | `nearest_obstacle_distance` |
| 2 | `nearest_obstacle_direction_x` |
| 3 | `nearest_obstacle_direction_y` |
| 4 | `nearest_obstacle_direction_z` |
| 5 | `front_min_range` |
| 6 | `front_left_min_range` |
| 7 | `front_right_min_range` |
| 8 | `left_min_range` |
| 9 | `right_min_range` |
| 10 | `up_min_range` |
| 11 | `down_min_range` |
| 12 | `valid_point_ratio` |
| 13 | `detection_confidence` |

完整点云不会直接喂给 PPO。空点云、NaN、Inf 和无检测都会返回稳定的固定维度占位特征。

## 6. Synthetic Fallback 边界

`SyntheticLidarSimulator` 只在无 ROS/PX4 的测试和诊断中使用，主要配合 `KinematicDiagnosticBackend` 让 `python -m pytest` 能覆盖 observation 接口。它基于真值电线采样局部点云，加轻量噪声与 dropout，不代表真实 3D LiDAR。

正式 PX4/Gazebo 运行时：

- 如果 `Px4RosBackend` 收到真实 PointCloud2，则提取真实点云特征。
- 如果没有真实 PointCloud2，则返回空 LiDAR 特征。
- synthetic fallback 不用于伪装真实 Gazebo LiDAR。

## 7. 安装 PX4 与消息包

```bash
cd ~
git clone --recursive https://github.com/PX4/PX4-Autopilot.git
cd PX4-Autopilot
git checkout v1.16.2
git submodule update --init --recursive
bash ./Tools/setup/ubuntu.sh
make px4_sitl gz_x500
```

准备 Micro XRCE-DDS Agent v2.x 后，在工程中放置 `px4_msgs`：

```bash
cd ~/uav_ppo_px4_3d_lidar/ros2_ws/src
git clone --branch release/1.16 https://github.com/PX4/px4_msgs.git
```

## 8. Python 环境与构建

```bash
cd ~/uav_ppo_px4_3d_lidar
/usr/bin/python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

source /opt/ros/humble/setup.bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## 9. 启动闭环栈

终端 A：

```bash
cd ~/uav_ppo_px4_3d_lidar
export PX4_AUTOPILOT_DIR=~/PX4-Autopilot
GUI=false bash tools/start_stack.sh
```

脚本启动 Agent、Gazebo world、bridge 和 PX4 SITL。不要在代码中写死本机 PX4 路径。

终端 B 验证：

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 topic echo /fmu/out/vehicle_odometry --once
ros2 service list | grep wire_training_world
ros2 run uav_px4_rl offboard_smoke_test
```

## 10. 在线 PPO 训练

```bash
cd ~/uav_ppo_px4_3d_lidar
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
source .venv/bin/activate
python train/train_ppo_px4.py \
  --scenario random \
  --num-wires 3 \
  --perception lidar \
  --timesteps 300000
```

常用参数：

```bash
# 固定多电线场景，仅用于链路排障
python train/train_ppo_px4.py --scenario fixed --num-wires 3 --timesteps 20000

# 指定模型名
python train/train_ppo_px4.py --model-name ppo_px4_3d_lidar_multiwire_seed7 --seed 7

# 接入已 bridge 的真实 PointCloud2 topic
python train/train_ppo_px4.py --lidar-topic /your/pointcloud2/topic
```

默认模型输出：

```text
models/ppo_px4_3d_lidar_multiwire.zip
models/checkpoints/
logs/monitor.csv
logs/tensorboard/
```

## 11. 在线评估

GUI 展示时先启动：

```bash
GUI=true bash tools/start_stack.sh
```

运行策略：

```bash
python eval/evaluate_online.py \
  --model models/ppo_px4_3d_lidar_multiwire.zip \
  --episodes 10 \
  --num-wires 3 \
  --perception lidar
```

输出：

```text
outputs/evaluation/actual_px4_trajectory.csv
outputs/evaluation/summary.json
```

CSV/summary 会记录 `num_wires`、`min_true_wire_distance`、`nearest_wire_index`、`lidar_confidence`、`goal_distance`、`collision` 和 `reached_goal`。

## 12. 真实 3D LiDAR 接入步骤

当前代码已经有真实接口：

- `Px4RosBackend(lidar_topic=...)`
- `Px4RosBackend.get_lidar_points()`
- `Px4RosBackend.latest_lidar_points`
- `Px4GazeboWireEnv` 中的 LiDAR feature extraction

仍需手动完成：

1. 在 Gazebo/PX4 `x500` 模型中挂载 3D LiDAR 传感器插件。
2. 将 Gazebo 点云 bridge 到 ROS 2 `sensor_msgs/msg/PointCloud2`。
3. 必要时扩展 `sim_bridge.launch.py` 或使用外部 `ros_gz_bridge` 配置。
4. 训练和评估时传入 `--lidar-topic /your/pointcloud2/topic`。

`sim_bridge.launch.py` 目前只桥接本项目必须的 `/clock`、`SetEntityPose` 和 `ControlWorld` 服务，并预留了 `lidar_topic` launch 参数说明。

## 13. 测试

不依赖 ROS/PX4 的测试：

```bash
cd ~/uav_ppo_px4_3d_lidar
source .venv/bin/activate
python -m pytest
```

覆盖内容：

- fixed 多电线场景生成。
- random 多电线场景生成且变化。
- 每根电线长度正确。
- `point_to_segments_distance` 找到最近电线并返回所有电线距离。
- LiDAR 特征对空点云和正常点云输出固定维度。
- `SyntheticLidarSimulator` 基于多电线场景生成点云。
- 环境 observation shape 为 22。
- observation 不再按旧结构直接包含真值 wire distance。
- `KinematicDiagnosticBackend` 在无 ROS/PX4 下仍能 reset 和 step。

PX4/Gazebo 集成验收仍需在 Ubuntu 仿真环境完成，包括 Offboard 冒烟、随机多线重置、真实 PointCloud2 接入和在线评估视频记录。
