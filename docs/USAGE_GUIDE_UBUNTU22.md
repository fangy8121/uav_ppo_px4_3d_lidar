# Ubuntu 22.04 使用指南

## 1. 项目目标

本项目是 **3B-lite：多电线 + 3D LiDAR perception features** 分支。策略 observation 不再直接使用真值电线几何，而是使用固定长度 LiDAR 感知特征；reward、collision 和 info 仍可使用仿真真值距离作为训练反馈和诊断数据。

```text
PPO policy
  -> Px4GazeboWireEnv.step(action)
  -> ROS 2 /fmu/in/* Offboard messages
  -> PX4 SITL flight controller
  -> Gazebo Harmonic x500_3d_lidar physics + multi-wire world
  -> PX4 VehicleOdometry + PointCloud2
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
| `uav_px4_rl/gz_pointcloud_bridge.py` | Gazebo `PointCloudPacked` 到 ROS 2 `PointCloud2` 的点云 bridge |
| `launch/sim_bridge.launch.py` | 启动 Gazebo world、当前服务 bridge 和默认 LiDAR bridge |
| `models/x500_3d_lidar/model.sdf` | 本项目默认 PX4/Gazebo 3D LiDAR 机型 |
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
- 如果没有真实 PointCloud2，则返回空 LiDAR 特征；这只是排障边界，不是本项目默认训练路径。
- synthetic fallback 不用于伪装真实 Gazebo LiDAR。

## 7. 安装 PX4、Agent 与消息包

### 7.1 安装 PX4 Autopilot

```bash
cd ~
git clone --recursive https://github.com/PX4/PX4-Autopilot.git
cd PX4-Autopilot
git checkout v1.16.2
git submodule update --init --recursive
bash ./Tools/setup/ubuntu.sh
make px4_sitl gz_x500
```

`make px4_sitl gz_x500` 用于生成 PX4 SITL 可执行文件和默认 Gazebo 资源。本项目的 `x500_3d_lidar` 模型由 `tools/start_stack.sh` 加入 Gazebo 资源路径，不需要改 PX4 源码目录。

### 7.2 安装 Micro XRCE-DDS Agent

Micro XRCE-DDS Agent 是 PX4 与 ROS 2 之间的 DDS 通信代理。PX4 SITL 内部运行的是 `uxrce_dds_client`，它通过 UDP 连接到 `MicroXRCEAgent`；ROS 2 侧才能看到 `/fmu/out/*`，并把 `/fmu/in/*` Offboard 指令送回 PX4。没有它，Gazebo 里可能有飞机，但 ROS 2 后端收不到 PX4 里程计，也发不进控制指令。


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

### 7.3 放置 `px4_msgs`

`px4_msgs` 的分支必须和 PX4 v1.16 对齐：

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

## 9. 首次启动与冒烟验证

### 9.1 首次启动带 GUI 的默认 3D LiDAR 栈

第一次启动建议用 `GUI=true`，这样可以直接确认 Gazebo 打开、世界加载、无人机模型是 `x500_3d_lidar`，并在后续冒烟测试时看到多电线场景和飞机运动。

终端 A：

```bash
cd ~/uav_ppo_px4_3d_lidar
export PX4_AUTOPILOT_DIR=~/PX4-Autopilot
GUI=true bash tools/start_stack.sh
```

脚本启动 Micro XRCE-DDS Agent、Gazebo world、Gazebo 服务 bridge、LiDAR 点云 bridge 和 PX4 SITL。不要在代码中写死本机 PX4 路径。

本项目默认就是带 3D LiDAR 的 x500：

```text
PX4_SIM_MODEL=gz_x500_3d_lidar
BRIDGE_LIDAR=true
```

LiDAR 点云默认桥接到 ROS 2 topic：

```text
/x500/lidar/points
```

正常使用本项目不需要切换，默认就是 `gz_x500_3d_lidar`。只在排障或兼容旧流程时才覆盖为基本 `gz_x500`。

可以直接在终端执行下面这一行，它的含义是：只对这一次 `bash tools/start_stack.sh` 命令临时设置三个环境变量。

```bash
PX4_SIM_MODEL=gz_x500 BRIDGE_LIDAR=false GUI=false bash tools/start_stack.sh
```

这条命令会启动基本 `x500`，并关闭 LiDAR bridge；因此 `/x500/lidar/points` 不会作为可用训练点云。命令结束后这些临时变量不会自动保留。若已经有一个仿真栈在运行，先在终端 A 按 `Ctrl-C` 停止，再启动另一种模型。

### 9.2 检查 PX4 里程计、LiDAR 点云和 Gazebo 服务

终端 B：

```bash
cd ~/uav_ppo_px4_3d_lidar
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 topic echo /fmu/out/vehicle_odometry --once
ros2 topic echo /x500/lidar/points --once
ros2 service list | grep wire_training_world
```

### 9.3 LiDAR与Offboard 冒烟测试

LiDAR冒烟测试确认 `/x500/lidar/points` 能被 `Px4RosBackend` 收到、能从 `PointCloud2` 转成 numpy 点云，并能压缩成 13 维 LiDAR features
Offboard 冒烟测试会接管 PX4 Offboard，摆放固定多电线场景，并让飞机经过几个安全航点。首次启动使用 `GUI=true` 时，可以在 Gazebo 里看到模型、场景和运动过程。

```bash
ros2 run uav_px4_rl lidar_smoke_test
ros2 run uav_px4_rl offboard_smoke_test
```

`lidar_smoke_test` 会订阅 `/x500/lidar/points`，把 `PointCloud2` 转成 numpy 点云，并压缩成 13 维 LiDAR features。它验证真实点云接收和特征提取，不验证语义分类；需要额外走固定场景准备流程时再运行 `ros2 run uav_px4_rl lidar_smoke_test --prepare-fixed-scene`。

`offboard_smoke_test` 验证 PX4 Offboard、Gazebo 服务、多电线模型摆放和基础航点运动链路。

## 10. 在线 PPO 训练

### 10.1 是否需要重启仿真栈

如果第 9 节首次验证时使用的是 `GUI=true`，训练前建议在终端 A 按 `Ctrl-C` 停止当前栈，然后用 `GUI=false` 重启，减少 Gazebo GUI 的 CPU/GPU 负担。不要在旧栈还运行时再启动一个新栈。

如果当前已经是 `GUI=false bash tools/start_stack.sh` 启动的默认 `gz_x500_3d_lidar` 栈，并且第 9 节冒烟验证已通过，则不需要重新启动。

### 10.2 终端 A：启动无 GUI 训练栈

```bash
cd ~/uav_ppo_px4_3d_lidar
export PX4_AUTOPILOT_DIR=~/PX4-Autopilot
GUI=false bash tools/start_stack.sh
```

终端 A 要保持打开。这个脚本会启动 Micro XRCE-DDS Agent、Gazebo server、Gazebo 服务 bridge、LiDAR 点云 bridge 和 PX4 SITL。

默认训练栈仍然是：

```text
PX4_SIM_MODEL=gz_x500_3d_lidar
BRIDGE_LIDAR=true
GUI=false
```

### 10.3 终端 B：训练前快速检查

重启为无 GUI 栈后，建议至少确认一次里程计和 LiDAR 点云仍然正常：

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

如果 `ros2 topic echo /fmu/out/vehicle_odometry --once` 没有输出，或者训练时报 `No /fmu/out/vehicle_odometry received from PX4`，说明终端 A 的 PX4/Gazebo 栈没有运行、还没启动完成，或 ROS 2 没连上 Micro XRCE-DDS Agent。先回到 10.2 重新启动无 GUI 栈，等 PX4 日志出现 `Startup script returned successfully` 后再继续。

如果刚刚重启过仿真栈，但第 9 节已经完整跑过 `offboard_smoke_test`，这里通常不需要再次跑 Offboard 冒烟测试。更换 PX4、Gazebo 模型、world 或 bridge 配置后，再重新跑：

```bash
ros2 run uav_px4_rl offboard_smoke_test
```

### 10.4 终端 B：启动正式 PPO 训练

确认终端 A 的无 GUI 栈仍在运行后，继续在终端 B 启动正式训练：

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

正式训练默认使用实时闭环。不要为正式训练添加 `--synchronous`；该选项会在每个动作后暂停并步进 Gazebo，仅用于短时同步链路诊断，在 PX4 lockstep 和 3D LiDAR 同时运行时可能发生 Gazebo control 服务阻塞。

训练入口默认订阅：

```text
/x500/lidar/points
```

### 10.5 常用训练参数

```bash
# 固定多电线场景，仅用于链路排障
python train/train_ppo_px4.py --scenario fixed --num-wires 3 --timesteps 20000

# 指定模型名
python train/train_ppo_px4.py --model-name ppo_px4_3d_lidar_multiwire_seed7 --seed 7

# 接入其他真实 PointCloud2 topic
python train/train_ppo_px4.py --lidar-topic /your/pointcloud2/topic

# 仅诊断 Gazebo 暂停/步进链路，不用于正式训练
python train/train_ppo_px4.py --synchronous --scenario fixed --timesteps 64 --n-steps 64
```

`--perception lidar` 是本项目默认训练路径。`--perception empty` 或 `none` 只用于诊断，不用于正式 3D LiDAR 训练。

### 10.6 训练输出

默认模型输出：

```text
models/ppo_px4_3d_lidar_multiwire.zip
models/checkpoints/
logs/ppo_px4_3d_lidar_multiwire_monitor.csv
logs/tensorboard/
```

训练默认每 `2000` 步保存一次检查点。如果训练中断，入口还会保存：

```text
models/ppo_px4_3d_lidar_multiwire_interrupted.zip
```

从检查点继续训练时，`--timesteps` 表示本次额外训练的步数：

```bash
python train/train_ppo_px4.py \
  --resume-model models/checkpoints/ppo_px4_3d_lidar_multiwire_2000_steps.zip \
  --timesteps 98000
```

训练过程中不要关闭终端 A。训练结束或需要切换 GUI/模型时，先停止训练进程，再到终端 A 按 `Ctrl-C` 关闭仿真栈。

## 11. 在线评估

GUI 展示时先启动：

```bash
cd ~/uav_ppo_px4_3d_lidar
export PX4_AUTOPILOT_DIR=~/PX4-Autopilot
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

评估入口同样默认订阅 `/x500/lidar/points`。

输出：

```text
outputs/evaluation/actual_px4_trajectory.csv
outputs/evaluation/summary.json
```

CSV/summary 会记录 `num_wires`、`min_true_wire_distance`、`nearest_wire_index`、`lidar_confidence`、`goal_distance`、`collision` 和 `reached_goal`。

## 12. 可选：更换外部 PointCloud2 Topic

默认 3D LiDAR 启动和检查流程已经在第 9 节和第 10 节写完，不需要在这里重复。本节只用于以后换成真实硬件 LiDAR 或其他 Gazebo LiDAR topic。

当前代码使用这些接口接收真实点云：

- `Px4RosBackend(lidar_topic=...)`
- `Px4RosBackend.get_lidar_points()`
- `Px4RosBackend.latest_lidar_points`
- `Px4GazeboWireEnv` 中的 LiDAR feature extraction

外部 LiDAR 必须发布 ROS 2 `sensor_msgs/msg/PointCloud2`。先检查它是否有数据：

```bash
ros2 topic echo /your/pointcloud2/topic --once
ros2 run uav_px4_rl lidar_smoke_test --lidar-topic /your/pointcloud2/topic
```

训练和评估时覆盖默认 topic：

```bash
python train/train_ppo_px4.py --lidar-topic /your/pointcloud2/topic
python eval/evaluate_online.py \
  --model models/ppo_px4_3d_lidar_multiwire.zip \
  --lidar-topic /your/pointcloud2/topic
```

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
- 正式后端没有 LiDAR 样本时返回 `None`，环境使用空 LiDAR 特征。
- 环境 observation shape 为 22。
- observation 不再按旧结构直接包含真值 wire distance。
- `KinematicDiagnosticBackend` 在无 ROS/PX4 下仍能 reset 和 step。

PX4/Gazebo 集成验收需在 Ubuntu 仿真环境完成：

```bash
ros2 run uav_px4_rl lidar_smoke_test
ros2 run uav_px4_rl offboard_smoke_test
```

后续在线评估可再记录视频和 `outputs/evaluation/summary.json`。
