"""Train PPO online against PX4 SITL and the Gazebo wire scene."""

from pathlib import Path
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from uav_px4_rl.env import Px4GazeboWireEnv
from uav_px4_rl.px4_backend import Px4RosBackend


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument(
        "--scenario",
        choices=["fixed", "random"],
        default="random",
        help="'random' is the 3B-lite curriculum; 'fixed' is the pipeline check.",
    ) #场景模式 fixed 固定场景 random 随机场景
    parser.add_argument("--seed", type=int, default=0)  #
    parser.add_argument("--num-wires", type=int, default=3)
    parser.add_argument("--perception", default="lidar", choices=["lidar", "empty", "none"])
    parser.add_argument("--lidar-topic", default=None)
    parser.add_argument("--model-name", default="ppo_px4_3d_lidar_multiwire")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Do not pause/step Gazebo; useful for debugging, not recommended for training.",
    ) #切换到实时模式，不推荐用于训练。
    return parser.parse_args() #返回解析后的命令行参数


def main():
    args = parse_args() 
    model_dir = PROJECT_ROOT / "models" #模型保存目录
    log_dir = PROJECT_ROOT / "logs" #日志保存目录
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    backend = Px4RosBackend(
        synchronous=not args.realtime,
        lidar_topic=args.lidar_topic,
    ) #创建PX4ROS后端
    env = Px4GazeboWireEnv( #创建PX4GazeboWireEnv环境
        backend=backend,
        scenario_mode=args.scenario, #场景模式
        seed=args.seed, #随机种子
        num_wires=args.num_wires,
        perception_mode=args.perception,
    )
    env = Monitor(env, filename=str(log_dir / "monitor.csv"))

    checkpoint = CheckpointCallback( #创建检查点回调
        save_freq=10_000, #保存频率
        save_path=str(model_dir / "checkpoints"), #保存路径
        name_prefix=args.model_name, #模型名称
    )
    model = PPO(
        "MlpPolicy", #MLP策略
        env, 
        learning_rate=3e-4, 
        n_steps=1024, 
        batch_size=64,
        gamma=0.99, 
        gae_lambda=0.95, 
        clip_range=0.2, 
        verbose=1, #详细程度
        seed=args.seed, #随机种子
        tensorboard_log=str(log_dir / "tensorboard"), #TensorBoard日志路径
    ) 
    try: #尝试训练模型
        model.learn(total_timesteps=args.timesteps, callback=checkpoint, progress_bar=True)
        output_path = model_dir / args.model_name #模型保存路径
        model.save(str(output_path)) #保存模型
        print(f"Online PX4/Gazebo PPO model saved to {output_path}.zip") #打印模型保存路径
    finally: #最终关闭环境
        env.close() #关闭环境


if __name__ == "__main__":
    main()
