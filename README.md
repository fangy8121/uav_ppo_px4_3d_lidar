# uav_ppo_px4_3d_lidar

这是 **3B-lite 多电线 3D LiDAR 分支**。项目由 `D:\UAV\uav_ppo_px4_demo` 复制而来，当前目录用于独立开发，不应再直接修改原 1B 项目。

本阶段目标是把原来的单电线真值几何避障升级为：

```text
PPO -> Gymnasium Px4GazeboWireEnv
    -> 3D LiDAR perception features observation
    -> ROS 2 Offboard -> PX4 SITL -> Gazebo x500_3d_lidar + 多电线场景
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
    models/x500_3d_lidar/model.sdf
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
| `px4_backend.py` | PX4 Offboard 后端，多电线模型摆放，PointCloud2 订阅接口 |
| `gz_pointcloud_bridge.py` | Gazebo `PointCloudPacked` 到 ROS 2 `PointCloud2` 的点云 bridge |
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

安装或检查 Micro XRCE-DDS Agent：

```bash
command -v MicroXRCEAgent
```

如果没有输出，安装 v2.x：

```bash
sudo apt install cmake g++ git
cd ~
git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent
git checkout v2.4.3
mkdir -p build
cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig /usr/local/lib
```

Micro XRCE-DDS Agent 是 PX4 SITL 与 ROS 2 的通信代理。没有它，ROS 2 后端收不到 `/fmu/out/*` 里程计，也无法把 `/fmu/in/*` Offboard 指令送进 PX4。正式启动时由 `tools/start_stack.sh` 自动运行，不需要手动常驻启动。

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

## 首次启动与冒烟验证

本仓库默认模型就是带 3D LiDAR 的 `x500_3d_lidar`。首次启动建议打开 GUI，确认 Gazebo 世界、完整无人机模型和后续冒烟测试运动过程：

```bash
cd ~/uav_ppo_px4_3d_lidar
export PX4_AUTOPILOT_DIR=~/PX4-Autopilot
GUI=true bash tools/start_stack.sh
```

启动脚本默认会设置 `PX4_SIM_MODEL=gz_x500_3d_lidar` 并开启点云 bridge。日常训练时关闭 GUI：

```bash
GUI=false bash tools/start_stack.sh
```

默认 ROS 2 点云 topic：

```text
/x500/lidar/points
```

训练前在另一个终端完成验收：

```bash
cd ~/uav_ppo_px4_3d_lidar
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 topic echo /fmu/out/vehicle_odometry --once
ros2 topic echo /x500/lidar/points --once
ros2 run uav_px4_rl lidar_smoke_test
ros2 run uav_px4_rl offboard_smoke_test
```

`lidar_smoke_test` 会通过 `Px4RosBackend(lidar_topic="/x500/lidar/points")` 接收真实 `PointCloud2`，转换为 numpy 点云，并用 `LiDARFeatureExtractor` 压缩成 13 维特征。该测试验证点云链路和特征提取，不做语义分类；需要额外走固定场景准备流程时再运行 `ros2 run uav_px4_rl lidar_smoke_test --prepare-fixed-scene`。

基本 `gz_x500` 只作为排障/兼容覆盖项。可以直接执行下面这一行，它只对这一次启动临时设置环境变量；如果已有仿真栈在运行，先按 `Ctrl-C` 停止：

```bash
PX4_SIM_MODEL=gz_x500 BRIDGE_LIDAR=false GUI=false bash tools/start_stack.sh
```

这会启动基本 `x500` 并关闭 LiDAR bridge，因此 `/x500/lidar/points` 不会作为可用训练点云。

## 在线训练

如果首次验证时使用的是 `GUI=true`，训练前先在启动栈的终端按 `Ctrl-C` 停止，再用无 GUI 模式重启，减少负担：

```bash
cd ~/uav_ppo_px4_3d_lidar
export PX4_AUTOPILOT_DIR=~/PX4-Autopilot
GUI=false bash tools/start_stack.sh
```

如果当前已经是 `GUI=false` 的默认 `gz_x500_3d_lidar` 栈，并且冒烟验证已通过，则不用重启。保持启动栈终端打开，在另一个终端快速确认点云链路：

```bash
cd ~/uav_ppo_px4_3d_lidar
mkdir -p logs/ros logs/matplotlib
export ROS_LOG_DIR=$PWD/logs/ros
export MPLCONFIGDIR=$PWD/logs/matplotlib
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 topic echo /fmu/out/vehicle_odometry --once
ros2 run uav_px4_rl lidar_smoke_test
```

如果里程计没有输出，先不要训练，回到启动栈终端确认 PX4 已启动完成。确认链路正常后直接启动正式训练：

```bash
cd ~/uav_ppo_px4_3d_lidar
mkdir -p logs/ros logs/matplotlib
export ROS_LOG_DIR=$PWD/logs/ros
export MPLCONFIGDIR=$PWD/logs/matplotlib
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
source .venv/bin/activate
python train/train_ppo_px4.py \
  --scenario random \
  --num-wires 3 \
  --perception lidar \
  --timesteps 100000
```

正式训练默认使用实时闭环，避免 PX4 lockstep 下频繁暂停/步进 Gazebo 导致服务阻塞。`--synchronous` 仅用于短时同步链路诊断。

默认模型名：

```text
models/ppo_px4_3d_lidar_multiwire.zip
```

默认训练入口订阅 `/x500/lidar/points`。如果改接其他真实 3D LiDAR 的 `sensor_msgs/msg/PointCloud2` topic，再覆盖：

```bash
python train/train_ppo_px4.py --lidar-topic /your/pointcloud2/topic
```

没有真实 LiDAR 样本时，PX4 后端返回 `None`，环境使用空 LiDAR 特征；只有 `KinematicDiagnosticBackend` 本地测试会启用 `SyntheticLidarSimulator` fallback。这个边界由纯 Python 测试覆盖，正式训练前仍应运行 `lidar_smoke_test` 确认真实点云链路可用。

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

测试覆盖固定/随机多电线场景、有限线段距离、多线最近距离与所有距离、LiDAR 特征、synthetic 点云、正式后端无点云时的空特征路径、环境 observation shape，以及诊断后端 reset/step。

PX4/Gazebo 集成验收需要在仿真栈运行时执行：

```bash
ros2 run uav_px4_rl lidar_smoke_test
ros2 run uav_px4_rl offboard_smoke_test
```

## 可选：更换外部 PointCloud2 Topic

默认 3D LiDAR 启动和检查流程已在前面写完。本节只用于以后换成真实硬件 LiDAR 或其他 Gazebo LiDAR topic。外部 LiDAR 必须发布 ROS 2 `sensor_msgs/msg/PointCloud2`：

```bash
ros2 run uav_px4_rl lidar_smoke_test --lidar-topic /your/pointcloud2/topic
python train/train_ppo_px4.py --lidar-topic /your/pointcloud2/topic
```

不要在代码中写死用户本机 PX4 路径。PX4 路径继续通过环境变量和启动脚本管理。
